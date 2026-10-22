"""Stage2A joint updates and exact backbone inheritance into frozen Stage2B."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from fastwam.trainer import Wan22Trainer, _zeva_stage2a_schedule
from fastwam.zeva.checkpoint import (
    FOUNDATION_CONTRACT_SCHEMA, checkpoint_requires_foundation, checkpoint_sha256,
    resolve_joint_foundation_checkpoint,
)
from fastwam.zeva_memory_contract import pim_memory_contract
from test_ego_policy_alignment import predictive_sample
from test_foundation_mode_precision import POLICY_TASK, _compose, _tiny_runtime_model
from test_joint_foundation import joint_model, _inference_kwargs
from test_zero1_training import _assert_tree_equal


def _policy_model():
    model = joint_model()
    model.set_zeva_training_stage("policy_injection")
    model.zeva_training_recipe = "ego_joint_parent"
    model.configure_zeva_trainable_state()
    return model


def _frozen_pim_model():
    model = joint_model()
    model.zeva_training_recipe = "ego_adapter_warmup"
    model.configure_zeva_trainable_state()
    return model


def test_joint_stage2a_parent_keeps_fp32_backbone_in_frozen_stage2b_training_resume(tmp_path):
    source = _policy_model().bfloat16()
    source.torch_dtype = torch.bfloat16
    source.prepare_zeva_training_precision()
    with torch.no_grad():
        # Deliberately below BF16 resolution, as with a small foundation LR.
        source.video_expert.patch_embedding.weight.add_(1e-6)
        source.proprio_encoder.weight.add_(1e-6)
    assert not torch.equal(source.video_expert.patch_embedding.weight,
                           source.video_expert.patch_embedding.weight.bfloat16().float())
    parent_path = tmp_path / "stage2a_addon.pt"
    source.save_zeva_addon_checkpoint(parent_path, step=5000)
    parent = torch.load(parent_path, weights_only=False)
    assert parent["foundation_mode"] == "joint"
    assert checkpoint_requires_foundation(parent)
    parent_foundation = resolve_joint_foundation_checkpoint(parent_path, parent)

    target = _frozen_pim_model().bfloat16()
    target.torch_dtype = torch.bfloat16
    target.prepare_zeva_training_precision()
    assert target.video_expert.patch_embedding.weight.dtype == torch.bfloat16
    initial_reader = deepcopy(target.zeva_prompt_encoder.state_dict())
    target.load_zeva_addon_checkpoint(parent_path, load_scope="policy_injection")
    assert target.video_expert.patch_embedding.weight.dtype == torch.float32
    assert all(not p.requires_grad for p in target.zeva_foundation_parameters())
    assert target.zeva_foundation_mode == "frozen"
    assert target.zeva_requires_foundation_checkpoint
    for key, value in source.mot.state_dict().items():
        torch.testing.assert_close(target.mot.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(target.proprio_encoder.weight, source.proprio_encoder.weight, rtol=0, atol=0)
    _assert_tree_equal(target.zeva_prompt_encoder.state_dict(), initial_reader)

    output = tmp_path / "stage2b_addon.pt"
    target.save_zeva_addon_checkpoint(output, step=12000)
    child = torch.load(output, weights_only=False)
    assert child["foundation_mode"] == "frozen"
    assert child["foundation_checkpoint_required"] is True
    assert checkpoint_requires_foundation(child)
    assert child["foundation_parent_identity"] == {
        "addon_sha256": checkpoint_sha256(parent_path),
        "foundation_sha256": checkpoint_sha256(parent_foundation),
        "training_stage": "policy_injection", "training_recipe": "ego_joint_parent",
    }
    child_foundation = resolve_joint_foundation_checkpoint(output, child)
    assert child_foundation != parent_foundation
    # A frozen child is a self-contained pair; the source A files may be gone.
    parent_path.unlink()
    parent_foundation.unlink()
    restored = _frozen_pim_model().bfloat16()
    restored.torch_dtype = torch.bfloat16
    restored.prepare_zeva_training_precision()
    restored.load_zeva_addon_checkpoint(output)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    kwargs = _inference_kwargs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected_action = target.eval().infer_action_zeva(**kwargs)["action"]
        actual_action = restored.eval().infer_action_zeva(**kwargs)["action"]
    torch.testing.assert_close(actual_action, expected_action, rtol=0, atol=0)
    child_foundation.unlink()
    with pytest.raises(FileNotFoundError, match="requires its foundation checkpoint"):
        restored.load_zeva_addon_checkpoint(output)


def test_frozen_stage2b_bundle_loads_default_bf16_deployment_foundation(tmp_path):
    parent = _policy_model().bfloat16()
    parent.torch_dtype = torch.bfloat16
    parent.prepare_zeva_training_precision()
    with torch.no_grad():
        parent.video_expert.patch_embedding.weight.add_(1e-6)
        parent.action_expert.head.weight.add_(.1)
        parent.proprio_encoder.weight.add_(.1)
    parent_path = tmp_path / "parent_addon.pt"
    parent.save_zeva_addon_checkpoint(parent_path)
    child = _frozen_pim_model().bfloat16()
    child.torch_dtype = torch.bfloat16
    child.prepare_zeva_training_precision()
    child.load_zeva_addon_checkpoint(parent_path, load_scope="policy_injection")
    path = tmp_path / "child_addon.pt"
    child.save_zeva_addon_checkpoint(path)

    deployed = _frozen_pim_model().bfloat16().eval().requires_grad_(False)
    deployed.torch_dtype = torch.bfloat16
    # Production inference intentionally does not call training preparation.
    initial = deployed.action_expert.head.weight.detach().clone()
    deployed.load_zeva_addon_checkpoint(path)
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in deployed.parameters())
    assert not torch.equal(initial, deployed.action_expert.head.weight)
    assert deployed.zeva_requires_foundation_checkpoint
    assert deployed.zeva_foundation_mode == "frozen"
    for name, parameter in deployed.named_parameters():
        torch.testing.assert_close(parameter, child.get_parameter(name).bfloat16(), rtol=0, atol=0)
    # BF16 deployment rounds the FP32 training state. Compare against that
    # explicit cast, never claim that every 1e-6 update survives quantization.
    reference = _frozen_pim_model().bfloat16().eval().requires_grad_(False)
    reference.torch_dtype = torch.bfloat16
    reference.load_state_dict({key: value.to(dtype=torch.bfloat16) if value.is_floating_point() else value
                               for key, value in child.state_dict().items()}, strict=True)
    kwargs = _inference_kwargs()
    torch.testing.assert_close(deployed.infer_action_zeva(**kwargs)["action"],
                               reference.infer_action_zeva(**kwargs)["action"], rtol=0, atol=0)


def test_joint_parent_requires_explicit_partial_load_into_next_stage(tmp_path):
    path = tmp_path / "parent_addon.pt"
    _policy_model().save_zeva_addon_checkpoint(path)
    with pytest.raises(ValueError, match="training stage mismatch"):
        joint_model().load_zeva_addon_checkpoint(path)
    joint_model().load_zeva_addon_checkpoint(path, load_scope="policy_injection")


@pytest.mark.parametrize("problem", ["disabled", "disabled_without_sidecar", "invalid_parent", "missing_parent", "invalid_digest"])
def test_inherited_foundation_contract_cannot_be_silently_downgraded(tmp_path, problem):
    path = tmp_path / "parent_addon.pt"
    _policy_model().save_zeva_addon_checkpoint(path)
    target = _frozen_pim_model()
    target.load_zeva_addon_checkpoint(path, load_scope="policy_injection")
    child_path = tmp_path / "child_addon.pt"
    target.save_zeva_addon_checkpoint(child_path)
    payload = torch.load(child_path, weights_only=False)
    if problem in {"disabled", "disabled_without_sidecar"}:
        payload["foundation_checkpoint_required"] = False
        if problem == "disabled_without_sidecar":
            payload.pop("foundation_checkpoint")
    elif problem == "invalid_parent":
        payload["foundation_parent_identity"]["training_recipe"] = "frozen_parent"
    elif problem == "missing_parent":
        payload.pop("foundation_parent_identity")
    else:
        payload["foundation_parent_identity"]["foundation_sha256"] = "not-a-digest"
    with pytest.raises(ValueError):
        checkpoint_requires_foundation(payload)


def test_loading_original_base_addon_requires_discarding_previous_inherited_foundation(tmp_path):
    parent_path, base_path, child_path = (tmp_path / name for name in
        ("parent_addon.pt", "original_base.pt", "base_frozen_addon.pt"))
    parent = _policy_model()
    parent.save_zeva_addon_checkpoint(parent_path)
    target = _frozen_pim_model()
    target.save_checkpoint(base_path)
    target.save_zeva_addon_checkpoint(child_path)
    target.load_zeva_addon_checkpoint(parent_path, load_scope="policy_injection")
    with pytest.raises(ValueError, match="load the original FastWAM base"):
        target.load_zeva_addon_checkpoint(child_path)
    target.load_checkpoint(base_path)
    target.load_zeva_addon_checkpoint(child_path)
    assert not target.zeva_requires_foundation_checkpoint


@pytest.mark.parametrize("drop_marker", [False, True])
def test_new_frozen_child_cannot_be_downgraded_to_legacy_by_dropping_foundation_fields(tmp_path, drop_marker):
    parent_path, path = tmp_path / "parent_addon.pt", tmp_path / "child_addon.pt"
    _policy_model().save_zeva_addon_checkpoint(parent_path)
    child = _frozen_pim_model()
    child.zeva_pim_memory_contract = pim_memory_contract({
        "cross_attempt_only": True, "compression": "similarity_merge",
        "uniform_indexing": "reference_round", "merge_threshold": .95,
    })
    child.load_zeva_addon_checkpoint(parent_path, load_scope="policy_injection")
    child.save_zeva_addon_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    assert payload["foundation_contract_schema"] == FOUNDATION_CONTRACT_SCHEMA
    for name in ("foundation_checkpoint_required", "foundation_checkpoint", "foundation_parent_identity"):
        payload.pop(name)
    if drop_marker:
        payload.pop("foundation_contract_schema")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="foundation contract"):
        resolve_joint_foundation_checkpoint(path)
    with pytest.raises(ValueError, match="foundation contract"):
        _frozen_pim_model().load_zeva_addon_checkpoint(path)


@pytest.mark.parametrize("mode", ["frozen", "joint"])
def test_historical_uniform_floor_checkpoint_without_new_contract_still_loads(tmp_path, mode):
    path = tmp_path / "historical_addon.pt"
    model = joint_model() if mode == "joint" else _frozen_pim_model()
    model.save_zeva_addon_checkpoint(path, config={"model": {"zeva": {"memory": {
        "cross_attempt_only": True, "compression": "uniform", "uniform_indexing": "legacy_floor",
    }}}})
    payload = torch.load(path, weights_only=False)
    for name in ("foundation_contract_schema", "foundation_checkpoint_required"):
        payload.pop(name)
    if mode == "joint":
        sidecar = path.parent / payload["foundation_checkpoint"]["filename"]
        foundation = torch.load(sidecar, weights_only=False)
        for name in ("foundation_contract_schema", "foundation_checkpoint_required", "training_stage"):
            foundation.pop(name)
        torch.save(foundation, sidecar)
        payload["foundation_checkpoint"]["sha256"] = checkpoint_sha256(sidecar)
    torch.save(payload, path)
    restored = joint_model() if mode == "joint" else _frozen_pim_model()
    restored.load_zeva_addon_checkpoint(path)
    assert checkpoint_requires_foundation(payload) == (mode == "joint")
    _assert_tree_equal(restored.state_dict(), model.state_dict())


@pytest.mark.parametrize("problem", ["dit_only", "partial_mot", "unexpected_mot", "missing_proprio",
                                      "partial_proprio", "wrong_proprio_shape"])
def test_partial_base_restore_cannot_clear_loaded_foundation_or_mix_policy_weights(tmp_path, problem):
    parent_path, base_path, base_addon = (tmp_path / name for name in
        ("parent_addon.pt", "base.pt", "base_addon.pt"))
    model = _frozen_pim_model()
    model.save_checkpoint(base_path)
    model.save_zeva_addon_checkpoint(base_addon)
    original = deepcopy(model.state_dict())
    parent = _policy_model()
    with torch.no_grad():
        parent.video_expert.patch_embedding.weight.add_(.2)
        parent.action_expert.head.weight.add_(.3)
        parent.proprio_encoder.weight.add_(.4)
    parent.save_zeva_addon_checkpoint(parent_path)
    model.load_zeva_addon_checkpoint(parent_path, load_scope="policy_injection")
    inherited = deepcopy(model.state_dict())
    identity = deepcopy(model.zeva_foundation_identity)
    parent_identity = deepcopy(model.zeva_foundation_parent_identity)
    payload = torch.load(base_path, weights_only=False)
    if problem == "dit_only":
        payload = {"dit": {key: value.clone() for key, value in parent.video_expert.state_dict().items()}}
    elif problem == "partial_mot":
        payload["mot"].pop(next(iter(payload["mot"])))
    elif problem == "unexpected_mot":
        payload["mot"]["unrecognized_weight"] = torch.zeros(1)
    elif problem == "missing_proprio":
        payload.pop("proprio_encoder")
    elif problem == "partial_proprio":
        payload["proprio_encoder"].pop("bias")
    else:
        payload["proprio_encoder"]["weight"] = payload["proprio_encoder"]["weight"][:, :1]
    partial = tmp_path / "partial.pt"
    torch.save(payload, partial)
    with pytest.raises(ValueError, match="complete"):
        model.load_checkpoint(partial)
    _assert_tree_equal(model.state_dict(), inherited)
    assert model.zeva_foundation_identity == identity
    assert model.zeva_foundation_parent_identity == parent_identity
    with pytest.raises(ValueError, match="load the original FastWAM base"):
        model.load_zeva_addon_checkpoint(base_addon)
    model.load_checkpoint(base_path)
    model.load_zeva_addon_checkpoint(base_addon)
    assert model.zeva_foundation_identity is None
    assert model.zeva_foundation_parent_identity is None
    _assert_tree_equal(model.state_dict(), original)


class _PreparedPolicyDataset(Dataset):
    def __init__(self, root, name, sample):
        root = root / name
        root.mkdir()
        (root / "manifest.json").write_text('{"fixture": "predictive"}')
        (root / "static.pt").write_bytes(b"task context fixture")
        self.pim = SimpleNamespace(root=root)
        self.static_context = SimpleNamespace(path=root / "static.pt")
        self.sample = {key: value for key, value in sample.items() if isinstance(value, torch.Tensor)}

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {key: value[index].clone() for key, value in self.sample.items()}


@pytest.mark.parametrize("mode", ["joint", "frozen"])
def test_real_stage2a_trainer_constant_group_lrs_and_exact_resume(monkeypatch, tmp_path, mode):
    cfg = _compose(POLICY_TASK, f"foundation_mode={mode}", "batch_size=1",
                   "gradient_accumulation_steps=2", "num_workers=0", "max_steps=40",
                   "save_every=2", "eval_every=0", "log_every=0", "wandb.enabled=false")
    OmegaConf.update(cfg, "zeva_stage2a_schedule", "zeva_ego", force_add=True)
    cfg.output_dir = str(tmp_path / "full")
    for name in ("base", "cte"):
        (tmp_path / f"{name}.pt").write_bytes(name.encode())
    cfg.ckpt = str(tmp_path / "base.pt")
    cfg.model.zeva.cte.checkpoint = str(tmp_path / "cte.pt")
    sample = predictive_sample()
    train = _PreparedPolicyDataset(tmp_path, "train", sample)
    validation = _PreparedPolicyDataset(tmp_path, "validation", sample)
    torch.manual_seed(37)
    model = _tiny_runtime_model(monkeypatch, cfg)
    trainer = Wan22Trainer(model, train, validation, cfg=cfg)
    expected_lrs = {"eap": 5e-5, **({"foundation": 5e-6} if mode == "joint" else {})}
    assert {g["name"]: g["lr"] for g in trainer.optimizer.param_groups} == expected_lrs
    assert trainer.scheduler.state_dict()["last_epoch"] == 0
    # Exercise the start of a 40-update recipe, where the historical 5%
    # warmup would change its first updates. Stop after four and resume two.
    trainer.max_steps = 4
    original_backbone = model.video_expert.patch_embedding.weight.detach().clone()
    original_eap = model.zeva_behavior_prefix_adapter.behavior_effect_projector.weight.detach().clone()
    trainer.train()
    assert trainer.global_step == 4
    assert {g["name"]: g["lr"] for g in trainer.optimizer.param_groups} == expected_lrs
    assert not torch.equal(original_eap, model.zeva_behavior_prefix_adapter.behavior_effect_projector.weight)
    assert torch.equal(original_backbone, model.video_expert.patch_embedding.weight) == (mode == "frozen")
    assert all(p.grad is None and not p.requires_grad for p in model.vae.parameters())
    expected_model = deepcopy(model.state_dict())
    expected_optimizer = deepcopy(trainer.optimizer.state_dict())
    expected_scheduler = deepcopy(trainer.scheduler.state_dict())

    resume_cfg = deepcopy(cfg)
    resume_cfg.resume = str(tmp_path / "full/checkpoints/state/step_000002")
    resume_cfg.output_dir = str(tmp_path / "resumed")
    torch.manual_seed(37)
    resumed_model = _tiny_runtime_model(monkeypatch, resume_cfg)
    resumed = Wan22Trainer(resumed_model, train, validation, cfg=resume_cfg)
    assert resumed.global_step == 2
    resumed.max_steps = 4
    resumed.train()
    _assert_tree_equal(resumed_model.state_dict(), expected_model)
    _assert_tree_equal(resumed.optimizer.state_dict(), expected_optimizer)
    _assert_tree_equal(resumed.scheduler.state_dict(), expected_scheduler)


def test_legacy_stage2a_schedule_is_explicit_and_resume_cannot_change_it():
    old_cfg = {"lr_scheduler_type": "constant"}
    assert _zeva_stage2a_schedule(old_cfg, "frozen_parent") == "legacy_warmup"
    assert _zeva_stage2a_schedule(old_cfg, "ego_joint_parent") == "zeva_ego"
    new_cfg = {**old_cfg, "zeva_stage2a_schedule": "zeva_ego"}
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.cfg = old_cfg
    trainer.zeva_training_stage, trainer.zeva_training_recipe = "policy_injection", "frozen_parent"
    trainer.predictive_zeva = True
    saved = {"training_recipe": "frozen_parent", "config": old_cfg}
    trainer._validate_zeva_resume_recipe(saved)
    trainer.cfg = {**old_cfg, "zeva_stage2a_schedule": "legacy_warmup"}
    trainer._validate_zeva_resume_recipe(saved)
    trainer.cfg = new_cfg
    with pytest.raises(ValueError, match="zeva_stage2a_schedule"):
        trainer._validate_zeva_resume_recipe(saved)
    with pytest.raises(ValueError, match="constant learning-rate"):
        _zeva_stage2a_schedule({**new_cfg, "lr_scheduler_type": "cosine"}, "ego_joint_parent")
    with pytest.raises(ValueError, match="must be zeva_ego or legacy_warmup"):
        _zeva_stage2a_schedule({"zeva_stage2a_schedule": "unknown"}, "frozen_parent")
