"""Exercise config propagation through the real factory without loading weights."""

from types import SimpleNamespace

import pytest
import torch

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.runtime import create_fastwam


def _create(monkeypatch, **overrides):
    model = SimpleNamespace()

    def attach(prompt, adapter):
        model.prompt = prompt
        model.adapter = adapter

    model.attach_zeva_addon = attach
    model.set_zeva_training_stage = lambda stage: setattr(model, "stage", stage)
    monkeypatch.setattr(FastWAM, "from_wan22_pretrained", lambda **kwargs: model)
    zeva = {
        "enabled": True,
        "training_stage": "pim_adapter",
        "prompt": {
            "global_dim": 8, "phase_dim": 4, "effect_dim": 4,
            "brief_length": 2, "persistent_length": 6,
            "hidden_dim": 8, "num_heads": 2,
            "pim_context_dropout": 0.0, "pim_support_dropout": 0.0,
        },
        "adapter": {
            "mode": "exact_zeva", "memory_dim": 8, "global_dim": 8,
            "phase_dim": 4, "effect_dim": 4, "effect_history_length": 2,
            "action_horizon": 4, "prior_hidden_dim": 8, "prior_num_heads": 2,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict):
            zeva[key].update(value)
        else:
            zeva[key] = value
    return create_fastwam(
        model_id="unused", tokenizer_model_id="unused",
        video_dit_config={"hidden_dim": 8},
        action_dit_config={"hidden_dim": 8, "action_dim": 14},
        action_scheduler={"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 10},
        device="cpu", model_dtype=torch.float32, zeva=zeva,
    )


def test_top_level_cross_attempt_variant_reaches_both_real_modules(monkeypatch):
    model = _create(monkeypatch, pim_variant="cross_attempt")
    assert model.prompt.config.pim_variant == "cross_attempt"
    assert model.adapter.config.pim_variant == "cross_attempt"
    assert model.adapter.config.context_dim == 8
    assert model.stage == "pim_adapter"


def test_legacy_factory_configuration_keeps_legacy_modules(monkeypatch):
    model = _create(monkeypatch)
    assert model.prompt.config.pim_variant == "legacy"
    assert model.adapter.config.pim_variant == "legacy"


@pytest.mark.parametrize("overrides", [
    {"pim_variant": "cross_attempt", "adapter": {"pim_variant": "legacy"}},
    {"prompt": {"pim_variant": "cross_attempt"}},
    {"pim_variant": "misspelled"},
    {"pim_variant": "cross_attempt", "adapter": {"mode": "memory_residual"}},
])
def test_factory_rejects_mixed_or_unsupported_variants(monkeypatch, overrides):
    with pytest.raises(ValueError, match="variant|cross_attempt"):
        _create(monkeypatch, **overrides)
