import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from fastwam.trainer import Wan22Trainer


class _FakeExactAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.prior = nn.Linear(2, 2)
        self.action_prior_adapter = nn.Linear(2, 2)
        self.behavior_global_projector = nn.Linear(2, 2)
        self.prefix_project = nn.Linear(2, 2)
        self.pim_gate = nn.Parameter(torch.zeros(()))
        self.reset_calls = 0

    def policy_injection_parameters(self):
        yield from self.prior.parameters()
        yield from self.action_prior_adapter.parameters()
        yield from self.behavior_global_projector.parameters()

    def pim_adapter_parameters(self):
        yield from self.prefix_project.parameters()
        yield self.pim_gate

    def reset_pim_parameters(self):
        self.reset_calls += 1
        with torch.no_grad():
            self.prefix_project.weight.zero_()
            self.prefix_project.bias.zero_()
            self.pim_gate.zero_()


class _FakeZevaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.zeva_enabled = True
        self.zeva_training_stage = "pim_adapter"
        self.base = nn.Linear(2, 2)
        self.zeva_prompt_encoder = nn.Linear(2, 2)
        self.zeva_behavior_prefix_adapter = _FakeExactAdapter()

    def configure_zeva_trainable_state(self):
        self.eval()
        self.requires_grad_(False)
        self.zeva_prompt_encoder.train().requires_grad_(True)
        adapter = self.zeva_behavior_prefix_adapter
        adapter.prefix_project.train().requires_grad_(True)
        adapter.pim_gate.requires_grad_(True)

    def zeva_trainable_parameters(self):
        yield from self.zeva_prompt_encoder.parameters()
        yield from self.zeva_behavior_prefix_adapter.pim_adapter_parameters()

    def zeva_parameter_report(self):
        trainable_names = [name for name, p in self.named_parameters() if p.requires_grad]
        return {
            "training_stage": self.zeva_training_stage,
            "trainable_names": trainable_names,
            "frozen_names": [name for name, p in self.named_parameters() if not p.requires_grad],
            "trainable_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen_count": sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }

    def load_zeva_addon_checkpoint(self, path, *, base_checkpoint_sha256, cte_checkpoint_sha256, load_scope):
        assert load_scope == "all"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.zeva_prompt_encoder.load_state_dict(payload["causal_prompt_encoder"])
        self.zeva_behavior_prefix_adapter.load_state_dict(payload["behavior_prefix_adapter"])
        return payload


class _FakeAccelerator:
    def unwrap_model(self, model):
        return model


def _write_pim_resume_checkpoint(tmp_path: Path, model):
    checkpoint_root = tmp_path / "checkpoints"
    weights_dir = checkpoint_root / "weights"
    state_dir = checkpoint_root / "state"
    weights_dir.mkdir(parents=True)
    resume_dir = state_dir / "step_000123"
    resume_dir.mkdir(parents=True)
    addon_path = weights_dir / "step_000123_addon.pt"
    torch.save(
        {
            "causal_prompt_encoder": model.zeva_prompt_encoder.state_dict(),
            "behavior_prefix_adapter": model.zeva_behavior_prefix_adapter.state_dict(),
            "training_stage": "pim_adapter",
        },
        addon_path,
    )
    return addon_path, resume_dir


def test_pim_stage_resume_restores_all_state(tmp_path):
    source = _FakeZevaModel()
    source.configure_zeva_trainable_state()
    with torch.no_grad():
        source.zeva_prompt_encoder.weight.fill_(1.25)
        source.zeva_prompt_encoder.bias.fill_(1.5)
        source.zeva_behavior_prefix_adapter.prefix_project.weight.fill_(2.25)
        source.zeva_behavior_prefix_adapter.prefix_project.bias.fill_(2.5)
        source.zeva_behavior_prefix_adapter.pim_gate.fill_(0.75)
    _, resume_dir = _write_pim_resume_checkpoint(tmp_path, source)

    target = _FakeZevaModel()
    target.configure_zeva_trainable_state()
    optimizer = torch.optim.AdamW([
        {"params": list(target.zeva_prompt_encoder.parameters()), "lr": 5e-4},
        {"params": list(target.zeva_behavior_prefix_adapter.prefix_project.parameters()), "lr": 5e-4},
        {"params": [target.zeva_behavior_prefix_adapter.pim_gate], "lr": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    loss = sum(p.sum() for p in target.parameters() if p.requires_grad)
    loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    torch.save({"optimizer": optimizer_state, "scheduler": scheduler_state}, resume_dir / "optimizer_scheduler.pt")
    (resume_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": 123, "epoch": 4, "batch_in_epoch": 17}), encoding="utf-8"
    )

    base_path = tmp_path / "base.pt"; base_path.write_bytes(b"base")
    cte_path = tmp_path / "cte.pt"; cte_path.write_bytes(b"cte")
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = target
    trainer.optimizer = torch.optim.AdamW([
        {"params": list(target.zeva_prompt_encoder.parameters()), "lr": 5e-4},
        {"params": list(target.zeva_behavior_prefix_adapter.prefix_project.parameters()), "lr": 5e-4},
        {"params": [target.zeva_behavior_prefix_adapter.pim_gate], "lr": 1e-4},
    ])
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=lambda _: 1.0)
    trainer.accelerator = _FakeAccelerator()
    trainer.zeva_training = True
    trainer.zeva_training_stage = "pim_adapter"
    trainer.resume = str(resume_dir)
    trainer.cfg = OmegaConf.create({"ckpt": str(base_path), "model": {"zeva": {"cte": {"checkpoint": str(cte_path)}}}})
    trainer.global_step = trainer.epoch = trainer.batch_in_epoch = 0
    trainer._resume_or_load_checkpoint()

    assert target.zeva_behavior_prefix_adapter.reset_calls == 0
    torch.testing.assert_close(target.zeva_prompt_encoder.weight, source.zeva_prompt_encoder.weight)
    torch.testing.assert_close(target.zeva_prompt_encoder.bias, source.zeva_prompt_encoder.bias)
    torch.testing.assert_close(target.zeva_behavior_prefix_adapter.prefix_project.weight, source.zeva_behavior_prefix_adapter.prefix_project.weight)
    torch.testing.assert_close(target.zeva_behavior_prefix_adapter.prefix_project.bias, source.zeva_behavior_prefix_adapter.prefix_project.bias)
    torch.testing.assert_close(target.zeva_behavior_prefix_adapter.pim_gate, source.zeva_behavior_prefix_adapter.pim_gate)
    restored_optimizer = trainer.optimizer.state_dict()
    assert restored_optimizer["param_groups"] == optimizer_state["param_groups"]
    assert restored_optimizer["state"].keys() == optimizer_state["state"].keys()
    for parameter_id, restored_state in restored_optimizer["state"].items():
        expected_state = optimizer_state["state"][parameter_id]
        assert restored_state.keys() == expected_state.keys()
        for name, value in restored_state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected_state[name])
            else:
                assert value == expected_state[name]
    restored_scheduler = trainer.scheduler.state_dict()
    assert restored_scheduler.keys() == scheduler_state.keys()
    for name, value in restored_scheduler.items():
        assert value == scheduler_state[name]
    assert (trainer.global_step, trainer.epoch, trainer.batch_in_epoch) == (123, 4, 17)


def test_pim_stage_initialization_uses_policy_checkpoint_only(tmp_path):
    source = _FakeZevaModel()
    with torch.no_grad():
        source.zeva_behavior_prefix_adapter.prior.weight.fill_(1.0)
        source.zeva_behavior_prefix_adapter.action_prior_adapter.weight.fill_(2.0)
        source.zeva_behavior_prefix_adapter.behavior_global_projector.weight.fill_(3.0)
    policy_path = tmp_path / "stage2a.pt"
    torch.save({
        "causal_prompt_encoder": source.zeva_prompt_encoder.state_dict(),
        "behavior_prefix_adapter": source.zeva_behavior_prefix_adapter.state_dict(),
        "training_stage": "policy_injection",
    }, policy_path)
    target = _FakeZevaModel()
    scopes = []

    def loader(path, *, base_checkpoint_sha256, cte_checkpoint_sha256, load_scope):
        scopes.append(load_scope)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source_state = payload["behavior_prefix_adapter"]
        target_state = target.zeva_behavior_prefix_adapter.state_dict()
        for name, value in source_state.items():
            if name.startswith(("prior.", "action_prior_adapter.", "behavior_global_projector.")):
                target_state[name] = value
        target.zeva_behavior_prefix_adapter.load_state_dict(target_state)
        return payload

    target.load_zeva_addon_checkpoint = loader
    base_path = tmp_path / "base.pt"; base_path.write_bytes(b"base")
    cte_path = tmp_path / "cte.pt"; cte_path.write_bytes(b"cte")
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = target
    trainer.accelerator = _FakeAccelerator()
    trainer.zeva_training = True; trainer.zeva_training_stage = "pim_adapter"; trainer.resume = None
    trainer.cfg = OmegaConf.create({"ckpt": str(base_path), "model": {"zeva": {"policy_checkpoint": str(policy_path), "cte": {"checkpoint": str(cte_path)}}}})
    trainer._initialize_pim_stage_from_policy_checkpoint()
    assert scopes == ["policy_injection"]
    assert target.zeva_behavior_prefix_adapter.reset_calls == 1
