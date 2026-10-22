"""Exercise real Accelerate/DDP ZeRO-1 training, consolidation and exact resume.

The model is deliberately tiny; the production trainer, accumulation, sampler,
history curriculum, optimizer, checkpoint and rank RNG paths are unchanged.
"""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from fastwam.trainer import Wan22Trainer
from fastwam.zeva.distributed_training import (
    create_training_optimizer, load_optimizer_checkpoint, optimizer_betas,
    optimizer_sharding_mode, unwrap_optimizer,
)


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(effect_source="current_prediction", pim_variant="cross_attempt")
        self.pim = nn.Linear(4, 4)
        self.eap = nn.Linear(4, 4)

    def pim_adapter_parameters(self):
        return self.pim.parameters()

    def policy_injection_parameters(self):
        return self.eap.parameters()

    def reset_pim_parameters(self):
        with torch.no_grad():
            self.pim.weight.zero_()
            self.pim.bias.zero_()


class TinyPredictiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.zeva_enabled = True
        self.zeva_training_stage = "pim_adapter"
        self.zeva_training_recipe = "ego_joint_warmup"
        self.zeva_prompt_encoder = nn.Linear(4, 4)
        self.zeva_behavior_prefix_adapter = TinyAdapter()
        self.foundation = nn.Linear(4, 4)
        self.unused_foundation = nn.Linear(4, 4)
        self.frozen_encoder = nn.Linear(4, 4)
        self.losses, self.seen, self.history = [], [], []

    def configure_zeva_trainable_state(self):
        self.train().requires_grad_(True)
        self.frozen_encoder.eval().requires_grad_(False)
        return self

    def prepare_zeva_training_precision(self):
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

    def zeva_foundation_parameters(self):
        return list(self.foundation.parameters()) + list(self.unused_foundation.parameters())

    def zeva_trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def zeva_parameter_report(self):
        return {"trainable_names": [name for name, p in self.named_parameters() if p.requires_grad],
                "trainable_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "frozen_count": sum(p.numel() for p in self.parameters() if not p.requires_grad)}

    def forward(self, sample):
        x = sample["x"]
        adapter = self.zeva_behavior_prefix_adapter
        context = adapter.pim(self.zeva_prompt_encoder(x))
        # Empty history remains a valid DDP graph, while the unused video-like
        # branch above exercises find_unused_parameters throughout training.
        context = context * sample["pim_mask"].any(-1, keepdim=True)
        prediction = self.foundation(x) + adapter.eap(x) + context
        noise = torch.randn_like(prediction) * .01
        loss = ((prediction + noise - sample["target"]) ** 2).mean()
        self.losses.append(float(loss.detach()))
        self.seen.extend(sample["index"].tolist())
        self.history.extend(sample["pim_history_kind"].tolist())
        return loss, {"action_loss": loss.detach()}

    def save_zeva_addon_checkpoint(self, path, *, step, config, **lineage):
        torch.save({"model": {key: value.detach().cpu() for key, value in self.state_dict().items()},
                    "step": step, "config": config, "training_stage": self.zeva_training_stage,
                    "training_recipe": self.zeva_training_recipe, **lineage}, path)

    def load_zeva_addon_checkpoint(self, path, *, load_scope, **lineage):
        del lineage
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if load_scope == "policy_injection":
            state = {key.removeprefix("zeva_behavior_prefix_adapter.eap."): value
                     for key, value in payload["model"].items()
                     if key.startswith("zeva_behavior_prefix_adapter.eap.")}
            self.zeva_behavior_prefix_adapter.eap.load_state_dict(state)
        else:
            self.load_state_dict(payload["model"])
        return payload


class TinyPreparedDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        self.history_schedule = "ego_optimizer_step"
        self.pim = SimpleNamespace(root=root / split, manifest={"globally_available_history_kinds": ["matched"]})
        self.static_context = SimpleNamespace(path=root / "static.pt")

    def __len__(self):
        return 5  # Pads to 6 globally: 3 local microbatches, accumulation=4.

    def __getitem__(self, index):
        x = torch.tensor([1., .1 * index, -.3, .7])
        return {"index": index, "x": x, "target": x * .3,
                "pim_mask": torch.ones(2, dtype=torch.bool),
                "pim_phases": torch.ones(2, 4), "pim_effects": torch.ones(2, 4),
                "pim_history_available": torch.tensor([True, False, False]),
                "pim_history_phases": torch.ones(3, 2, 4), "pim_history_effects": torch.ones(3, 2, 4),
                "pim_history_masks": torch.ones(3, 2, dtype=torch.bool),
                "pim_history_scores": torch.ones(3, 2)}


def _config(root, output, sharding, resume=None):
    root = Path(root)
    return OmegaConf.create({
        "output_dir": str(root / output), "learning_rate": 5e-4, "weight_decay": 1e-4,
        "batch_size": 1, "num_workers": 0, "num_epochs": 1, "max_steps": 8,
        "log_every": 0, "save_every": 2, "eval_every": 0, "eval_num_inference_steps": 1,
        "gradient_accumulation_steps": 4, "max_grad_norm": .5, "seed": 197,
        "resume": None if resume is None else str(resume), "mixed_precision": "bf16",
        "wandb": {"enabled": False}, "lr_scheduler_type": "constant",
        "zeva_eap_learning_rate": 1e-4, "zeva_foundation_learning_rate": 1e-6,
        "zeva_pim_warmup_steps": 2, "zeva_optimizer_sharding": sharding, "optimizer_betas": None,
        "ckpt": str(root / "base.pt"),
        "model": {"zeva": {"policy_checkpoint": str(root / "parent.pt"),
                           "cte": {"checkpoint": str(root / "cte.pt")}}},
    })


def _make_trainer(root, output, sharding, resume=None):
    torch.manual_seed(29)
    model = TinyPredictiveModel().bfloat16()
    return Wan22Trainer(model, TinyPreparedDataset(root, "train"), TinyPreparedDataset(root, "validation"),
                        cfg=_config(root, output, sharding, resume))


def _local_moments(trainer):
    raw = unwrap_optimizer(trainer.optimizer)
    local = getattr(raw, "optim", raw)
    names = {id(parameter): name for name, parameter in trainer.accelerator.unwrap_model(trainer.model).named_parameters()}
    return {names[id(parameter)]: {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                                  for key, value in state.items()}
            for parameter, state in local.state.items()}


def _assert_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            _assert_tree_equal(first, second)
    else:
        assert actual == expected


def _zero1_worker(rank, root):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT="29589", ACCELERATE_USE_CPU="true")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        root = Path(root)
        full = _make_trainer(root, "full", "zero1")
        model = full.accelerator.unwrap_model(full.model)
        initial_foundation = model.foundation.weight.detach().clone()
        parent = torch.load(root / "parent.pt", weights_only=False)
        torch.testing.assert_close(model.zeva_behavior_prefix_adapter.eap.weight,
                                   parent["model"]["zeva_behavior_prefix_adapter.eap.weight"], rtol=0, atol=0)
        assert full.model.gradient_as_bucket_view
        assert full.optimizer.device_placement is False
        assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
        assert model.frozen_encoder.weight.dtype == torch.bfloat16
        full.train()
        expected_weights = deepcopy(model.state_dict())
        expected_moments = _local_moments(full)
        expected_scheduler = deepcopy(full.scheduler.state_dict())
        expected_losses, expected_seen, expected_history = model.losses[:], model.seen[:], model.history[:]
        assert len(expected_losses) == 32  # Eight full windows, crossing epoch boundaries.
        assert 3 in expected_history  # Empty-PIM branch after warmup.
        assert not torch.equal(model.foundation.weight, initial_foundation)
        warmup = torch.load(root / "full/checkpoints/weights/step_000002_addon.pt", weights_only=False)
        torch.testing.assert_close(warmup["model"]["foundation.weight"], initial_foundation, rtol=0, atol=0)
        raw = unwrap_optimizer(full.optimizer)
        assert raw._all_state_dicts == []  # No full CPU snapshot retained after save.
        assert 0 < len(expected_moments) < len(list(model.zeva_trainable_parameters()))
        owners = [None, None]
        dist.all_gather_object(owners, sorted(expected_moments))
        assert not set(owners[0]) & set(owners[1])
        assert all(value.device.type == "cpu" and value.dtype == torch.float32
                   for state in expected_moments.values() for key, value in state.items()
                   if key in {"exp_avg", "exp_avg_sq", "step"})

        resume = root / "full/checkpoints/state/step_000002"
        restored = _make_trainer(root, "resumed", "zero1", resume)
        resumed_optimizer = unwrap_optimizer(restored.optimizer)
        assert resumed_optimizer.state is resumed_optimizer.optim.state
        assert (restored.global_step, restored.epoch, restored.batch_in_epoch) == (2, 2, 2)
        restored.train()
        actual = restored.accelerator.unwrap_model(restored.model)
        _assert_tree_equal(actual.state_dict(), expected_weights)
        _assert_tree_equal(_local_moments(restored), expected_moments)
        _assert_tree_equal(restored.scheduler.state_dict(), expected_scheduler)
        assert actual.losses == expected_losses[8:]
        assert actual.seen == expected_seen[8:]
        assert actual.history == expected_history[8:]
        peers = [torch.empty_like(actual.foundation.weight) for _ in range(2)]
        dist.all_gather(peers, actual.foundation.weight.detach())
        torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)

        # State sharding must preserve the Adam update; compare with ordinary
        # DDP using identical inputs, seeds, group LRs and accumulation.
        replicated = _make_trainer(root, "replicated", "none")
        replicated.train()
        plain = replicated.accelerator.unwrap_model(replicated.model)
        _assert_tree_equal(plain.state_dict(), expected_weights)
        assert plain.losses == expected_losses
        for name, state in expected_moments.items():
            _assert_tree_equal(_local_moments(replicated)[name], state)
        checkpoint = torch.load(root / "full/checkpoints/state/step_000008/optimizer_scheduler.pt", weights_only=False)
        assert checkpoint["optimizer_contract"]["sharding"] == "zero1"
        assert checkpoint["optimizer_contract"]["world_size"] == 2
        assert len(checkpoint["optimizer"]["state"]) == len(owners[0]) + len(owners[1])
        assert all(value.device.type == "cpu" for state in checkpoint["optimizer"]["state"].values()
                   for value in state.values() if isinstance(value, torch.Tensor))
        (root / f"rank{rank}.complete").write_text("zero1, accumulated DDP, warmup, empty PIM and exact resume passed\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires CPU Gloo")
def test_two_rank_zero1_matches_ddp_and_resumes_exactly_across_warmup_and_epochs(tmp_path, monkeypatch):
    try:
        interfaces = dict(socket.if_nameindex()).values()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except OSError:
        pytest.skip("CPU DDP requires a permitted local loopback socket")
    if "lo" not in interfaces:
        pytest.skip("CPU DDP test requires the local Linux loopback interface")
    for split in ("train", "validation"):
        (tmp_path / split).mkdir()
        (tmp_path / split / "manifest.json").write_text(json.dumps({"split": split}))
    for name in ("base.pt", "cte.pt", "static.pt"):
        (tmp_path / name).write_bytes(name.encode())
    parent = TinyPredictiveModel()
    torch.save({"model": parent.state_dict(), "training_stage": "policy_injection"}, tmp_path / "parent.pt")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    multiprocessing.spawn(_zero1_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert (tmp_path / "rank0.complete").is_file() and (tmp_path / "rank1.complete").is_file()


def test_zero1_rejects_single_process_and_invalid_settings():
    with pytest.raises(ValueError, match="none or zero1"):
        optimizer_sharding_mode("zero2")
    with pytest.raises(ValueError, match="at least two processes"):
        create_training_optimizer([nn.Parameter(torch.ones(1))], sharding="zero1", predictive=True,
                                  lr=1e-4, weight_decay=0, betas=(.9, .999))
    with pytest.raises(ValueError, match="predictive"):
        create_training_optimizer([nn.Parameter(torch.ones(1))], sharding="zero1", predictive=False,
                                  lr=1e-4, weight_decay=0, betas=(.9, .999))
    assert optimizer_betas(None, predictive=True) == (.9, .999)
    assert optimizer_betas(None, predictive=False) == (.9, .95)
    with pytest.raises(ValueError, match="two finite"):
        optimizer_betas((.9, float("nan")), predictive=True)


def test_predictive_checkpoint_cannot_drop_partial_accumulated_gradients():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.predictive_zeva = True
    trainer.accelerator = SimpleNamespace(sync_gradients=False)
    with pytest.raises(ValueError, match="completed optimizer boundary"):
        trainer.save_checkpoint()


def test_predictive_resume_rejects_bf16_moments_and_changed_betas(tmp_path):
    parameter = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    parameter.sum().backward()
    optimizer.step()
    scheduler.step()
    payload = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
    path = tmp_path / "state.pt"
    payload["optimizer"]["state"][0]["exp_avg"] = payload["optimizer"]["state"][0]["exp_avg"].bfloat16()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="FP32 Adam moments"):
        load_optimizer_checkpoint(path, optimizer, scheduler, world_size=1, predictive=True)
    payload["optimizer"]["state"][0]["exp_avg"] = payload["optimizer"]["state"][0]["exp_avg"].float()
    payload["optimizer"]["param_groups"][0]["betas"] = (.9, .95)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="optimizer_betas"):
        load_optimizer_checkpoint(path, optimizer, scheduler, world_size=1, predictive=True)
