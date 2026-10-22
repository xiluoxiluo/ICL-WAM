"""Run actual main-profile optimizer, checkpoint and exact-resume CPU paths."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import Dataset

from fastwam.trainer import Wan22Trainer
from fastwam.zeva.checkpoint import checkpoint_sha256, resolve_joint_foundation_checkpoint
from fastwam.zeva.main_data import validate_main_artifact
from fastwam.zeva.main_prepared import split_main_prepared_dataset
from test_ego_policy_alignment import predictive_sample
from test_foundation_mode_precision import _compose, _tiny_runtime_model
from test_main_model_contract import main_metadata
from test_zero1_training import _assert_tree_equal


MAIN_POLICY = "robotwin_zeva_main_policy_3cam_384"
MAIN_PIM = "robotwin_zeva_main_pim_3cam_384"


class _PreparedMainSource(Dataset):
    """Replace external cache payload IO, retaining the real episode split view."""

    def __init__(self, root, metadata):
        root.mkdir()
        manifest = {**deepcopy(metadata), "data_role": "all"}
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "shard").mkdir()
        self.episode_ids = sorted(metadata["episode_splits"]["train"] + metadata["episode_splits"]["validation"])
        np.save(root / "shard/episode_index.npy", np.arange(len(self.episode_ids), dtype=np.int32))
        self.pim = SimpleNamespace(root=root, manifest=manifest, episode_ids=self.episode_ids,
            _cross_attempt=None, shards=[SimpleNamespace(path="shard", global_start=0, size=len(self.episode_ids))])
        static = root / "static.pt"
        torch.save({"fixture": "precomputed initial image/text task context", **manifest}, static)
        self.static_context = SimpleNamespace(path=static)
        self.history_schedule = "legacy"
        self.sample = {key: value for key, value in predictive_sample().items()
                       if isinstance(value, torch.Tensor) and not key.startswith("current_effect")}
        self.sample["pim_mask"].fill_(True)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        return {key: value[index % value.shape[0]].clone() for key, value in self.sample.items()}


def _config(tmp_path, stage, mode):
    cfg = _compose(MAIN_POLICY if stage == "policy_injection" else MAIN_PIM,
        f"foundation_mode={mode}", "batch_size=1", "gradient_accumulation_steps=2",
        "num_workers=0", "max_steps=4", "save_every=1", "eval_every=0", "log_every=0",
        "wandb.enabled=false")
    cfg.output_dir = str(tmp_path / "full")
    cfg.ckpt = str(tmp_path / "base.pt")
    cfg.model.zeva.cte.checkpoint = str(tmp_path / "cte.pt")
    # The tiny real experts use the same two-BIT/three-PIM test shapes.
    cfg.model.zeva.memory.bit_size = 2
    cfg.model.zeva.memory.pim_top_k = 3
    return cfg


def _new_model(monkeypatch, cfg, metadata):
    torch.manual_seed(37)
    tiny_cfg = deepcopy(cfg)
    # Main omits some production-size adapter defaults; the shared test
    # helper explicitly supplies smaller dimensions for its real experts.
    OmegaConf.set_struct(tiny_cfg.model.zeva.adapter, False)
    model = _tiny_runtime_model(monkeypatch, tiny_cfg)
    model.zeva_main_artifact_metadata = deepcopy(metadata)
    model.zeva_success_data_identity = deepcopy(metadata["success_data_identity"])
    model.configure_zeva_trainable_state()
    return model


@pytest.mark.parametrize("stage", ["policy_injection", "pim_adapter"])
@pytest.mark.parametrize("mode", ["frozen", "joint"])
def test_main_real_trainer_optimizer_save_and_exact_resume(monkeypatch, tmp_path, stage, mode):
    metadata = main_metadata(tmp_path)
    for name in ("base", "cte"):
        torch.save({"fixture": name, **metadata}, tmp_path / f"{name}.pt")
    source = _PreparedMainSource(tmp_path / "prepared", metadata)
    train, validation = split_main_prepared_dataset(source)
    assert len(train) == 3 and len(validation) == 1
    assert train.split_identity["episode_split_sha256"] == metadata["success_data_identity"]["split_sha256"]
    assert set(train.indices).isdisjoint(validation.indices)
    cfg = _config(tmp_path, stage, mode)
    parent = None
    if stage == "pim_adapter":
        # Both B variants initialize from a joint A, so frozen B must also
        # carry its inherited trained backbone through save/resume.
        parent_cfg = _config(tmp_path, "policy_injection", "joint")
        parent = _new_model(monkeypatch, parent_cfg, metadata)
        parent.prepare_zeva_training_precision()
        with torch.no_grad():
            parent.video_expert.patch_embedding.weight.add_(1e-6)
            parent.zeva_behavior_prefix_adapter.behavior_global_projector.weight.add_(0.0123)
            parent.zeva_behavior_prefix_adapter.action_prior_adapter.weight.normal_(std=0.02)
        parent_path = tmp_path / "parent_addon.pt"
        parent.save_zeva_addon_checkpoint(parent_path, step=7,
            config=OmegaConf.to_container(parent_cfg, resolve=True),
            base_checkpoint_sha256=checkpoint_sha256(cfg.ckpt),
            cte_checkpoint_sha256=checkpoint_sha256(cfg.model.zeva.cte.checkpoint))
        cfg.model.zeva.policy_checkpoint = str(parent_path)

    model = _new_model(monkeypatch, cfg, metadata)
    trainer = Wan22Trainer(model, train, validation, cfg=cfg)
    assert trainer.main_zeva and trainer.strict_zeva_training and not trainer.predictive_zeva
    assert not trainer.ego_curriculum
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters() if parameter.requires_grad)
    assert all(parameter.dtype == torch.bfloat16 and not parameter.requires_grad for parameter in model.vae.parameters())
    eap = list(model.zeva_behavior_prefix_adapter.policy_injection_parameters())
    assert all(parameter.requires_grad == (stage == "policy_injection") for parameter in eap)
    assert all(parameter.requires_grad == (mode == "joint") for parameter in model.zeva_foundation_parameters())
    expected_lrs = ([float(cfg.learning_rate)] if stage == "policy_injection"
                    else [5 * float(cfg.learning_rate), 5 * float(cfg.learning_rate), float(cfg.learning_rate)])
    if mode == "joint":
        expected_lrs.append(float(cfg.zeva_foundation_learning_rate))
    assert [group["lr"] for group in trainer.optimizer.param_groups] == expected_lrs
    assert trainer.optimizer.defaults["betas"] == (0.9, 0.95)
    if parent is not None:
        for name, value in parent.mot.state_dict().items():
            torch.testing.assert_close(model.mot.state_dict()[name], value, rtol=0, atol=0)
        for name, value in parent.zeva_behavior_prefix_adapter.state_dict().items():
            if name.startswith(("prior.", "action_prior_adapter.", "behavior_global_projector.")):
                torch.testing.assert_close(model.zeva_behavior_prefix_adapter.state_dict()[name], value, rtol=0, atol=0)

    before = deepcopy(model.state_dict())
    trainer.max_steps = 2
    trainer.train()
    assert trainer.global_step == 2
    # The production cosine scheduler uses a four-update horizon in this
    # fixture; its 1% floor gives a 0.505 multiplier at the midpoint and
    # preserves the relative group scales.
    assert [group["lr"] for group in trainer.optimizer.param_groups] == pytest.approx([lr * 0.505 for lr in expected_lrs])
    changed_foundation = not torch.equal(model.action_expert.head.weight, before["action_expert.head.weight"])
    assert changed_foundation == (mode == "joint")
    adapter = model.zeva_behavior_prefix_adapter
    if stage == "policy_injection":
        assert not torch.equal(adapter.behavior_global_projector.weight,
                               before["zeva_behavior_prefix_adapter.behavior_global_projector.weight"])
    else:
        assert adapter.pim_gate.item() != 0
        assert not torch.equal(adapter.prefix_project.weight, before["zeva_behavior_prefix_adapter.prefix_project.weight"])
        for name, parameter in adapter.named_parameters():
            if name.startswith(("prior.", "action_prior_adapter.", "behavior_global_projector.")):
                torch.testing.assert_close(parameter, before[f"zeva_behavior_prefix_adapter.{name}"], rtol=0, atol=0)
    expected_model = deepcopy(model.state_dict())
    expected_optimizer = deepcopy(trainer.optimizer.state_dict())
    expected_scheduler = deepcopy(trainer.scheduler.state_dict())
    assert expected_optimizer["state"]
    assert all(state[key].dtype == torch.float32 for state in expected_optimizer["state"].values()
               for key in ("exp_avg", "exp_avg_sq"))

    state = tmp_path / "full/checkpoints/state/step_000001"
    assert all((state / name).is_file() for name in
               ("COMPLETE", "optimizer_scheduler.pt", "trainer_state.json", "rng_rank_0000.pt"))
    progress = json.loads((state / "trainer_state.json").read_text())
    assert progress["data_identity"] == trainer._zeva_data_identity
    assert torch.load(state / "rng_rank_0000.pt", weights_only=False)["format"] == "zeva_main_stage2_rng_v1"
    weights = tmp_path / "full/checkpoints/weights/step_000001_addon.pt"
    addon = torch.load(weights, weights_only=False)
    validate_main_artifact(addon, expected=metadata)
    assert (resolve_joint_foundation_checkpoint(weights) is not None) == (mode == "joint" or parent is not None)

    resume_cfg = deepcopy(cfg)
    resume_cfg.resume = str(state)
    resume_cfg.output_dir = str(tmp_path / "resumed")
    if stage == "pim_adapter":
        # Resume must use its own complete bundle, never re-run A init.
        resume_cfg.model.zeva.policy_checkpoint = str(tmp_path / "unavailable_parent.pt")
    resumed_model = _new_model(monkeypatch, resume_cfg, metadata)
    resumed = Wan22Trainer(resumed_model, train, validation, cfg=resume_cfg)
    assert resumed.global_step == 1
    resumed.max_steps = 2
    resumed.train()
    _assert_tree_equal(resumed_model.state_dict(), expected_model)
    _assert_tree_equal(resumed.optimizer.state_dict(), expected_optimizer)
    _assert_tree_equal(resumed.scheduler.state_dict(), expected_scheduler)
