"""Joint FastWAM/EAP/PIM updates and complete deployment checkpoint bundles."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import socket

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel

from fastwam.zeva.checkpoint import (
    checkpoint_sha256, load_addon_checkpoint, resolve_joint_foundation_checkpoint,
)
from fastwam.zeva.ego_pim_schedule import eap_lr_factor
from fastwam.trainer import Wan22Trainer
from test_ego_policy_alignment import predictive_model, predictive_sample


class TinyFrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(4))

    def encode(self, image, scale):
        return self.bias[None, :, None, None, None].expand(image.shape[0], 4, 1, 2, 2)


class TinyFrozenVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyFrozenEncoder()
        self.scale = None


def joint_model(*, checkpoint=False):
    model = predictive_model("pim_adapter")
    model.vae = TinyFrozenVAE()
    model.text_encoder = nn.Linear(6, 6)
    model.proprio_dim = 2
    model.proprio_encoder = nn.Linear(2, 6)
    model.zeva_training_recipe = "ego_joint_warmup"
    model.zeva_task_context_mode = "task_language"
    model.mot.mot_checkpoint_mixed_attn = checkpoint
    model.configure_zeva_trainable_state()
    return model


def joint_sample():
    sample = predictive_sample()
    sample["proprio"] = torch.randn(3, 2)
    return sample


def test_joint_recipe_updates_foundation_after_warmup_and_keeps_encoders_frozen():
    model = joint_model(checkpoint=True)
    adapter = model.zeva_behavior_prefix_adapter
    pim = list(model.zeva_prompt_encoder.parameters()) + list(adapter.pim_adapter_parameters())
    eap = list(adapter.policy_injection_parameters())
    foundation = model.zeva_foundation_parameters()
    all_parameters = pim + eap + foundation
    assert len(set(map(id, all_parameters))) == len(all_parameters)
    assert set(map(id, all_parameters)) == set(map(id, model.zeva_trainable_parameters()))
    assert all(p.requires_grad for p in foundation)
    assert all(not p.requires_grad for p in model.vae.parameters())
    assert all(not p.requires_grad for p in model.text_encoder.parameters())
    assert model.mot.training and not model.vae.training and not model.text_encoder.training
    report = model.zeva_parameter_report()
    assert "video_expert.patch_embedding.weight" in report["trainable_names"]
    optimizer = torch.optim.AdamW([
        {"params": pim, "lr": 5e-4}, {"params": eap, "lr": 1e-4},
        {"params": foundation, "lr": 1e-5},
    ])
    scheduler = LambdaLR(optimizer, [lambda s: 1., lambda s: eap_lr_factor(s, 2), lambda s: eap_lr_factor(s, 2)])
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(joint_sample())
        loss.backward()
        for parameter in (model.video_expert.patch_embedding.weight,
                          model.video_expert.text_embedding[0].weight,
                          model.action_expert.head.weight, model.proprio_encoder.weight):
            assert parameter.grad is not None and parameter.grad.norm() > 0
        assert model.vae.model.bias.grad is None
        assert all(p.grad is None for p in model.text_encoder.parameters())
        optimizer.step()
        scheduler.step()
        if step < 2:
            for name in ("video_expert.patch_embedding.weight", "action_expert.head.weight", "proprio_encoder.weight",
                         "zeva_behavior_prefix_adapter.behavior_effect_projector.weight"):
                torch.testing.assert_close(model.get_parameter(name), before[name], rtol=0, atol=0)
    for name in ("video_expert.patch_embedding.weight", "action_expert.head.weight", "proprio_encoder.weight",
                 "zeva_behavior_prefix_adapter.behavior_effect_projector.weight",
                 "zeva_behavior_prefix_adapter.pim_to_global.weight"):
        assert not torch.equal(model.get_parameter(name), before[name]), name


def test_cached_action_checkpointing_matches_loss_all_gradients_and_rng():
    plain = joint_model(checkpoint=False)
    checkpointed = deepcopy(plain)
    checkpointed.mot.mot_checkpoint_mixed_attn = True
    sample = joint_sample()
    torch.manual_seed(92)
    state = torch.get_rng_state()
    loss, _ = plain(sample)
    loss.backward()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    other_loss, _ = checkpointed(sample)
    other_loss.backward()
    torch.testing.assert_close(loss, other_loss, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for (name, parameter), (other_name, other) in zip(plain.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6, msg=name)


def _save_joint_bundle(path):
    model = joint_model()
    with torch.no_grad():
        model.action_expert.head.weight.add_(0.123)
        model.video_expert.patch_embedding.weight.sub_(0.234)
        model.proprio_encoder.weight.add_(0.345)
    model.save_zeva_addon_checkpoint(path, step=501, config={},
                                     base_checkpoint_sha256="base", cte_checkpoint_sha256="cte")
    return model


def _inference_kwargs():
    return dict(prompt=None, input_image=torch.zeros(1, 3, 16, 16),
                context=torch.randn(1, 3, 6), context_mask=torch.ones(1, 3, dtype=torch.bool),
                action_horizon=32, num_inference_steps=2, causal_prompt=torch.randn(1, 8),
                pim_mask=torch.ones(1, 3, dtype=torch.bool), zeva_task_tokens=torch.randn(1, 8),
                zeva_current_effect=torch.randn(1, 8), zeva_current_effect_mask=torch.ones(1, dtype=torch.bool),
                initial_action_noise=torch.randn(1, 32, 14), proprio=torch.randn(1, 2),
                zeva_mode="pim_on", enable_action_prior=False)


def test_joint_bundle_roundtrip_relocates_and_restores_action_not_original_base(tmp_path):
    source = tmp_path / "step_000501_addon.pt"
    model = _save_joint_bundle(source).eval()
    sidecar = resolve_joint_foundation_checkpoint(source)
    assert sidecar.name == "step_000501_foundation.pt"
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copyfile(source, relocated / source.name)
    shutil.copyfile(sidecar, relocated / sidecar.name)
    loaded = joint_model().eval()
    kwargs = _inference_kwargs()
    expected = model.infer_action_zeva(**kwargs)["action"]
    original_base = loaded.infer_action_zeva(**kwargs)["action"]
    assert not torch.allclose(expected, original_base)
    payload = loaded.load_zeva_addon_checkpoint(relocated / source.name,
                                               base_checkpoint_sha256="base", cte_checkpoint_sha256="cte")
    actual = loaded.infer_action_zeva(**kwargs)["action"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loaded.zeva_foundation_identity == payload["foundation_checkpoint"]
    for key, tensor in model.mot.state_dict().items():
        torch.testing.assert_close(loaded.mot.state_dict()[key], tensor, rtol=0, atol=0)
    torch.testing.assert_close(loaded.proprio_encoder.weight, model.proprio_encoder.weight, rtol=0, atol=0)
    with pytest.raises(ValueError, match="complete FastWAM foundation loader"):
        load_addon_checkpoint(source, loaded.zeva_prompt_encoder, loaded.zeva_behavior_prefix_adapter)


@pytest.mark.parametrize("problem", ["missing", "hash", "step", "base", "path", "recipe"])
def test_joint_bundle_rejects_missing_or_mismatched_foundation(tmp_path, problem):
    path = tmp_path / "step_000501_addon.pt"
    _save_joint_bundle(path)
    payload = torch.load(path, weights_only=False)
    sidecar = resolve_joint_foundation_checkpoint(path, payload)
    if problem == "missing":
        sidecar.unlink()
    elif problem == "hash":
        with sidecar.open("ab") as handle:
            handle.write(b"changed")
    elif problem == "step":
        payload["foundation_checkpoint"]["step"] = 500
    elif problem == "base":
        payload["foundation_checkpoint"]["base_checkpoint_sha256"] = "other"
    elif problem == "path":
        payload["foundation_checkpoint"]["filename"] = "../other.pt"
    elif problem == "recipe":
        payload["training_recipe"] = "ego_adapter_warmup"
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_joint_foundation_checkpoint(path, payload)


def test_joint_bundle_checks_foundation_internal_lineage_even_when_hash_matches(tmp_path):
    path = tmp_path / "step_000501_addon.pt"
    model = _save_joint_bundle(path)
    addon = torch.load(path, weights_only=False)
    foundation_path = resolve_joint_foundation_checkpoint(path, addon)
    foundation = torch.load(foundation_path, weights_only=False)
    foundation["step"] -= 1
    torch.save(foundation, foundation_path)
    addon["foundation_checkpoint"]["sha256"] = checkpoint_sha256(foundation_path)
    torch.save(addon, path)
    with pytest.raises(ValueError, match="lineage mismatch: step"):
        model.load_zeva_addon_checkpoint(path)


def test_joint_resume_rejects_changed_foundation_lr():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.zeva_training_recipe = "ego_joint_warmup"
    trainer.predictive_zeva = True
    trainer.cfg = {"zeva_foundation_learning_rate": 1e-6}
    payload = {"training_recipe": "ego_joint_warmup", "config": dict(trainer.cfg)}
    trainer._validate_zeva_resume_recipe(payload)
    payload["config"]["zeva_foundation_learning_rate"] = 2e-6
    with pytest.raises(ValueError, match="zeva_foundation_learning_rate"):
        trainer._validate_zeva_resume_recipe(payload)


def test_direct_loader_requires_matching_joint_recipe_but_allows_stage2a_parent(tmp_path):
    joint = joint_model()
    frozen = predictive_model("pim_adapter")
    frozen_path = tmp_path / "frozen_addon.pt"
    frozen.save_zeva_addon_checkpoint(frozen_path)
    with pytest.raises(ValueError, match="joint/frozen foundation recipe mismatch"):
        joint.load_zeva_addon_checkpoint(frozen_path)
    joint_path = tmp_path / "joint_addon.pt"
    joint.save_zeva_addon_checkpoint(joint_path)
    with pytest.raises(ValueError, match="joint/frozen foundation recipe mismatch"):
        frozen.load_zeva_addon_checkpoint(joint_path)
    parent = predictive_model("policy_injection")
    with torch.no_grad():
        parent.zeva_behavior_prefix_adapter.behavior_effect_projector.weight.add_(0.25)
    parent_path = tmp_path / "stage2a_addon.pt"
    parent.save_zeva_addon_checkpoint(parent_path)
    before = joint.video_expert.patch_embedding.weight.detach().clone()
    joint.load_zeva_addon_checkpoint(parent_path, load_scope="policy_injection")
    torch.testing.assert_close(joint.video_expert.patch_embedding.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(joint.zeva_behavior_prefix_adapter.behavior_effect_projector.weight,
                               parent.zeva_behavior_prefix_adapter.behavior_effect_projector.weight, rtol=0, atol=0)


def _mixed_precision_joint_model():
    model = joint_model(checkpoint=True).bfloat16()
    model.torch_dtype = torch.bfloat16
    model.prepare_zeva_joint_training_precision()
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    assert all(p.dtype == torch.bfloat16 for p in model.vae.parameters())
    assert all(p.dtype == torch.bfloat16 for p in model.text_encoder.parameters())
    return model


def _joint_optimizer_and_scheduler(model):
    adapter = model.zeva_behavior_prefix_adapter
    optimizer = torch.optim.AdamW([
        {"params": list(model.zeva_prompt_encoder.parameters()) + list(adapter.pim_adapter_parameters()), "lr": 5e-5},
        {"params": list(adapter.policy_injection_parameters()), "lr": 1e-5},
        {"params": model.zeva_foundation_parameters(), "lr": 1e-6},
    ], betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, [lambda s: 1., lambda s: eap_lr_factor(s, 1), lambda s: eap_lr_factor(s, 1)])
    return optimizer, scheduler


def _mixed_precision_update(model, optimizer, scheduler, sample):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = model(sample)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


def test_joint_bf16_autocast_preserves_small_updates_and_full_precision_resume(tmp_path):
    source = _mixed_precision_joint_model()
    optimizer, scheduler = _joint_optimizer_and_scheduler(source)
    sample = joint_sample()
    initial = source.video_expert.patch_embedding.weight.detach().clone()
    _mixed_precision_update(source, optimizer, scheduler, sample)
    torch.testing.assert_close(source.video_expert.patch_embedding.weight, initial, rtol=0, atol=0)
    _mixed_precision_update(source, optimizer, scheduler, sample)
    assert not torch.equal(source.video_expert.patch_embedding.weight, initial)
    # Both Adam moments and parameters stay FP32 even though compute is bf16.
    assert all(state[key].dtype == torch.float32 for state in optimizer.state.values()
               for key in ("exp_avg", "exp_avg_sq"))

    base, cte = tmp_path / "base.pt", tmp_path / "cte.pt"
    base.write_bytes(b"base")
    cte.write_bytes(b"cte")
    weights = tmp_path / "checkpoints" / "weights"
    weights.mkdir(parents=True)
    resume = tmp_path / "checkpoints" / "state" / "step_000002"
    resume.mkdir(parents=True)
    cfg = OmegaConf.create({"ckpt": str(base), "mixed_precision": "bf16",
                           "zeva_foundation_learning_rate": 1e-6,
                           "model": {"zeva": {"cte": {"checkpoint": str(cte)}}}})
    source.save_zeva_addon_checkpoint(weights / "step_000002_addon.pt", step=2,
        config=OmegaConf.to_container(cfg), base_checkpoint_sha256=checkpoint_sha256(base),
        cte_checkpoint_sha256=checkpoint_sha256(cte))
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
               resume / "optimizer_scheduler.pt")
    (resume / "trainer_state.json").write_text(json.dumps({
        "global_step": 2, "epoch": 0, "batch_in_epoch": 0, "data_identity": {},
    }))
    (resume / "COMPLETE").write_text("complete\n")
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = _mixed_precision_joint_model()
    trainer.optimizer, trainer.scheduler = _joint_optimizer_and_scheduler(trainer.model)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1,
                                         process_index=0, unwrap_model=lambda model: model)
    trainer.cfg, trainer.resume = cfg, str(resume)
    trainer.zeva_training = trainer.predictive_zeva = True
    trainer.zeva_training_stage, trainer.zeva_training_recipe = "pim_adapter", "ego_joint_warmup"
    trainer.global_step, trainer.epoch, trainer.batch_in_epoch = 2, 0, 0
    trainer._zeva_data_identity = {}
    trainer.train_sampler = SimpleNamespace(set_epoch_offset=lambda _: None, set_resume_batch_offset=lambda _: None)
    rng = trainer._rank_rng_state()
    torch.save(rng, resume / "rng_rank_0000.pt")
    trainer.global_step = 0
    trainer._resume_or_load_checkpoint()
    assert trainer.global_step == 2
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters() if p.requires_grad)
    assert all(state[key].dtype == torch.float32 for state in trainer.optimizer.state.values()
               for key in ("exp_avg", "exp_avg_sq"))
    trainer._restore_rank_rng(trainer._zeva_resume_rng)
    expected_loss = _mixed_precision_update(source, optimizer, scheduler, sample)
    trainer._restore_rank_rng(trainer._zeva_resume_rng)
    actual_loss = _mixed_precision_update(trainer.model, trainer.optimizer, trainer.scheduler, sample)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for (name, expected), (other_name, actual) in zip(source.named_parameters(), trainer.model.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)


def _joint_ddp_worker(rank, root):
    distributed.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank,
                                   world_size=2, timeout=timedelta(seconds=45))
    try:
        model = joint_model(checkpoint=True)
        wrapped = DistributedDataParallel(model, find_unused_parameters=True, broadcast_buffers=False)
        adapter = model.zeva_behavior_prefix_adapter
        pim = list(model.zeva_prompt_encoder.parameters()) + list(adapter.pim_adapter_parameters())
        optimizer = torch.optim.AdamW([
            {"params": pim, "lr": 5e-4},
            {"params": list(adapter.policy_injection_parameters()), "lr": 1e-4},
            {"params": model.zeva_foundation_parameters(), "lr": 1e-5},
        ])
        scheduler = LambdaLR(optimizer, [lambda s: 1., lambda s: eap_lr_factor(s, 1), lambda s: eap_lr_factor(s, 1)])
        initial = model.action_expert.head.weight.detach().clone()
        torch.manual_seed(91 + rank)
        for step in range(3):
            sample = joint_sample()
            if step == 2:
                sample["pim_mask"].zero_()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = wrapped(sample)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step == 0:
                torch.testing.assert_close(model.action_expert.head.weight, initial, rtol=0, atol=0)
        assert not torch.equal(model.action_expert.head.weight, initial)
        peers = [torch.empty_like(model.action_expert.head.weight) for _ in range(2)]
        distributed.all_gather(peers, model.action_expert.head.weight.detach())
        torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
        (Path(root) / f"rank{rank}.complete").write_text("warmup, joint update and empty PIM passed\n")
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not distributed.is_available() or not distributed.is_gloo_available(), reason="requires CPU Gloo")
def test_two_rank_ddp_joint_warmup_with_checkpointing_and_empty_history(tmp_path, monkeypatch):
    try:
        interfaces = dict(socket.if_nameindex()).values()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except OSError:
        pytest.skip("CPU DDP requires a permitted local loopback socket")
    if "lo" not in interfaces:
        pytest.skip("CPU DDP test requires the local Linux loopback interface")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    multiprocessing.spawn(_joint_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert (tmp_path / "rank0.complete").is_file() and (tmp_path / "rank1.complete").is_file()
