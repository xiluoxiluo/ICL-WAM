"""Prevent silent reinterpretation of merged-prefix checkpoints at deployment."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from fastwam.trainer import Wan22Trainer
from fastwam.zeva.behavior_prefix_adapter import (
    ExactZevaPolicyInjectionAdapter,
    ExactZevaPolicyInjectionConfig,
)
from fastwam.zeva.causal_prompt import CausalPromptConfig, CausalPromptEncoder
from fastwam.zeva.checkpoint import (
    addon_layout_metadata,
    checkpoint_sha256,
    load_addon_checkpoint,
    save_addon_checkpoint,
)


def _pair(layout="zeva_ego", variant="legacy"):
    prompt = CausalPromptEncoder(CausalPromptConfig(
        global_dim=8, phase_dim=4, effect_dim=4, brief_length=2,
        persistent_length=4, hidden_dim=8, num_heads=2,
        pim_context_dropout=0.0, pim_support_dropout=0.0,
        pim_variant=variant,
    ))
    adapter = ExactZevaPolicyInjectionAdapter(ExactZevaPolicyInjectionConfig(
        memory_dim=8, context_dim=8, global_dim=8, phase_dim=4,
        effect_dim=4, effect_history_length=2, action_dim=14,
        action_horizon=32, prior_hidden_dim=8, prior_num_heads=2,
        action_hidden_dim=8, prior_dropout_rate=0.0,
        pim_variant=variant, prefix_layout=layout,
    ))
    return prompt, adapter


def _save(path, pair, stage="policy_injection", **hashes):
    save_addon_checkpoint(
        path, *pair, step=12, config={},
        base_checkpoint_sha256=hashes.get("base", "base"),
        cte_checkpoint_sha256=hashes.get("cte", "cte"),
        training_stage=stage,
    )


@pytest.mark.parametrize("variant", ["legacy", "cross_attempt"])
def test_new_layout_checkpoint_roundtrip_records_effect_contract(tmp_path, variant):
    source = _pair(variant=variant)
    path = tmp_path / "new.pt"
    _save(path, source)
    target = _pair(variant=variant)
    payload = load_addon_checkpoint(path, *target)
    assert addon_layout_metadata(target[1]) == {
        "pim_variant": variant, "prefix_layout": "zeva_ego",
        "effect_source": "latest_completed", "addon_schema": "zeva_ego_prefix_v1",
    }
    assert payload["prefix_layout"] == "zeva_ego"
    assert payload["effect_source"] == "latest_completed"
    assert payload["addon_schema"] == "zeva_ego_prefix_v1"
    for source_module, target_module in zip(source, target):
        for key, expected in source_module.state_dict().items():
            torch.testing.assert_close(target_module.state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["legacy", "cross_attempt"])
def test_old_checkpoint_without_layout_remains_merged_only(tmp_path, variant):
    pair = _pair("legacy_merged", variant)
    path = tmp_path / "old.pt"
    _save(path, pair)
    payload = torch.load(path, weights_only=False)
    payload.pop("prefix_layout")
    payload.pop("effect_source")
    if variant == "legacy":
        payload.pop("addon_schema")
    torch.save(payload, path)
    load_addon_checkpoint(path, *_pair("legacy_merged", variant))
    with pytest.raises(ValueError, match="prefix_layout mismatch"):
        load_addon_checkpoint(path, *_pair("zeva_ego", variant))
    with pytest.raises(ValueError, match="prefix_layout mismatch"):
        load_addon_checkpoint(path, *_pair("zeva_ego", variant), load_scope="policy_injection")


@pytest.mark.parametrize("scope", ["all", "policy_injection"])
@pytest.mark.parametrize("key,value,match", [
    ("addon_schema", "cross_attempt_v1", "zeva_ego_prefix_v1"),
    ("effect_source", None, "effect_source"),
    ("effect_source", "future_ground_truth", "effect_source"),
    ("prefix_layout", None, "prefix_layout"),
])
def test_new_layout_rejects_missing_or_wrong_contract(tmp_path, scope, key, value, match):
    pair = _pair()
    path = tmp_path / "bad.pt"
    _save(path, pair)
    payload = torch.load(path, weights_only=False)
    if value is None:
        payload.pop(key)
    else:
        payload[key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        load_addon_checkpoint(path, *_pair(), load_scope=scope)


def test_new_stage2a_parent_imports_effect_without_loading_reader_or_pim(tmp_path):
    source = _pair(variant="legacy")
    target = _pair(variant="cross_attempt")
    path = tmp_path / "parent.pt"
    with torch.no_grad():
        source[1].behavior_effect_projector.weight.fill_(0.73)
    _save(path, source)
    reader_before = deepcopy(target[0].state_dict())
    prefix_before = deepcopy(target[1].prefix_project.state_dict())
    global_map_before = deepcopy(target[1].pim_to_global.state_dict())
    load_addon_checkpoint(path, *target, load_scope="policy_injection")
    for name, expected in source[1].state_dict().items():
        if name.startswith(("prior.", "action_prior_adapter.", "behavior_global_projector.", "behavior_effect_projector.")):
            torch.testing.assert_close(target[1].state_dict()[name], expected, rtol=0, atol=0)
    for module, before in ((target[0], reader_before), (target[1].prefix_project, prefix_before), (target[1].pim_to_global, global_map_before)):
        for name, expected in before.items():
            torch.testing.assert_close(module.state_dict()[name], expected, rtol=0, atol=0)
    payload = torch.load(path, weights_only=False)
    payload["behavior_prefix_adapter"].pop("behavior_effect_projector.weight")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incomplete policy-injection"):
        load_addon_checkpoint(path, *target, load_scope="policy_injection")


def test_explicit_warmstart_copies_only_old_parent_and_keeps_new_effect(tmp_path):
    source = _pair("legacy_merged")
    target = _pair()
    path = tmp_path / "old_stage2a.pt"
    _save(path, source)
    effect_before = deepcopy(target[1].behavior_effect_projector.state_dict())
    reader_before = deepcopy(target[0].state_dict())
    prefix_before = deepcopy(target[1].prefix_project.state_dict())
    load_addon_checkpoint(
        path, *target, load_scope="policy_warmstart",
        target_training_stage="policy_injection",
        base_checkpoint_sha256="base", cte_checkpoint_sha256="cte",
    )
    for name, expected in source[1].state_dict().items():
        if name.startswith(("prior.", "action_prior_adapter.", "behavior_global_projector.")):
            torch.testing.assert_close(target[1].state_dict()[name], expected, rtol=0, atol=0)
    for module, before in ((target[1].behavior_effect_projector, effect_before), (target[0], reader_before), (target[1].prefix_project, prefix_before)):
        for name, expected in before.items():
            torch.testing.assert_close(module.state_dict()[name], expected, rtol=0, atol=0)
    for stage in (None, "pim_adapter", "eval"):
        with pytest.raises(ValueError, match="only valid when training a new Stage2A"):
            load_addon_checkpoint(path, *target, load_scope="policy_warmstart", target_training_stage=stage)
    for key in ("base_checkpoint_sha256", "cte_checkpoint_sha256"):
        with pytest.raises(ValueError, match=key):
            load_addon_checkpoint(path, *target, load_scope="policy_warmstart", target_training_stage="policy_injection", **{key: "wrong"})


@pytest.mark.parametrize("source_layout,source_stage,target_layout", [
    ("legacy_merged", "pim_adapter", "zeva_ego"),
    ("zeva_ego", "policy_injection", "zeva_ego"),
    ("legacy_merged", "policy_injection", "legacy_merged"),
])
def test_warmstart_rejects_wrong_source_stage_or_layout(tmp_path, source_layout, source_stage, target_layout):
    path = tmp_path / "source.pt"
    _save(path, _pair(source_layout), source_stage)
    with pytest.raises(ValueError, match="Stage2A"):
        load_addon_checkpoint(path, *_pair(target_layout), load_scope="policy_warmstart", target_training_stage="policy_injection")


class _WarmstartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.zeva_prompt_encoder, self.zeva_behavior_prefix_adapter = _pair()
        self.zeva_enabled = True
        self.zeva_training_stage = "policy_injection"
        self.scopes = []

    def load_zeva_addon_checkpoint(self, path, **kwargs):
        self.scopes.append(kwargs["load_scope"])
        return load_addon_checkpoint(
            path, self.zeva_prompt_encoder, self.zeva_behavior_prefix_adapter,
            target_training_stage=self.zeva_training_stage, **kwargs,
        )

    def configure_zeva_trainable_state(self):
        self.requires_grad_(False)
        for parameter in self.zeva_behavior_prefix_adapter.policy_injection_parameters():
            parameter.requires_grad_(True)


def _warmstart_trainer(tmp_path):
    base_path = tmp_path / "base.pt"
    cte_path = tmp_path / "cte.pt"
    base_path.write_bytes(b"fixed FastWAM")
    cte_path.write_bytes(b"fixed CTE")
    old = _pair("legacy_merged")
    path = tmp_path / "stage2a.pt"
    _save(path, old, base=checkpoint_sha256(base_path), cte=checkpoint_sha256(cte_path))
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = _WarmstartModel()
    trainer.zeva_training = True
    trainer.zeva_training_stage = "policy_injection"
    trainer.resume = None
    trainer.cfg = OmegaConf.create({
        "ckpt": str(base_path),
        "model": {"zeva": {
            "policy_checkpoint": None,
            "policy_warmstart_checkpoint": str(path),
            "cte": {"checkpoint": str(cte_path)},
        }},
    })
    return trainer, old


def test_trainer_warmstart_is_stage2a_initialization_with_effect_trainable(tmp_path):
    trainer, old = _warmstart_trainer(tmp_path)
    effect_before = deepcopy(trainer.model.zeva_behavior_prefix_adapter.behavior_effect_projector.state_dict())
    trainer._initialize_policy_stage_from_warmstart()
    assert trainer.model.scopes == ["policy_warmstart"]
    adapter = trainer.model.zeva_behavior_prefix_adapter
    for name, expected in effect_before.items():
        torch.testing.assert_close(adapter.behavior_effect_projector.state_dict()[name], expected, rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in adapter.behavior_effect_projector.parameters())
    assert all(not parameter.requires_grad for parameter in trainer.model.zeva_prompt_encoder.parameters())
    torch.testing.assert_close(adapter.behavior_global_projector.weight, old[1].behavior_global_projector.weight, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["stage2b", "resume", "policy_checkpoint"])
def test_trainer_rejects_ambiguous_warmstart_before_loading_weights(tmp_path, invalid):
    trainer, _ = _warmstart_trainer(tmp_path)
    if invalid == "stage2b":
        trainer.zeva_training_stage = "pim_adapter"
    elif invalid == "resume":
        trainer.resume = "/some/resume/state"
    else:
        trainer.cfg.model.zeva.policy_checkpoint = "/some/other/parent.pt"
    with pytest.raises(ValueError, match="policy_warmstart_checkpoint"):
        trainer._initialize_policy_stage_from_warmstart()
    assert trainer.model.scopes == []


def test_trainer_warmstart_verifies_artifact_hashes(tmp_path):
    trainer, _ = _warmstart_trainer(tmp_path)
    Path(trainer.cfg.model.zeva.cte.checkpoint).write_bytes(b"a different CTE")
    with pytest.raises(ValueError, match="cte_checkpoint_sha256"):
        trainer._initialize_policy_stage_from_warmstart()
