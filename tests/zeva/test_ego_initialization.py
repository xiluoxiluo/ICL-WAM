"""Reference fan-in initialization without reinterpreting trained legacy weights."""

from dataclasses import replace
import math

import torch

from fastwam.zeva.behavior_prefix_adapter import ExactZevaPolicyInjectionAdapter, ExactZevaPolicyInjectionConfig
from fastwam.zeva.causal_prompt import CausalPromptConfig, CausalPromptEncoder


def _configs(initialization):
    return (
        CausalPromptConfig(global_dim=64, phase_dim=64, effect_dim=64, hidden_dim=64,
                           pim_variant="cross_attempt", initialization=initialization),
        ExactZevaPolicyInjectionConfig(memory_dim=64, context_dim=128, global_dim=64,
            phase_dim=64, effect_dim=64, prior_hidden_dim=64, action_hidden_dim=32,
            pim_variant="cross_attempt", prefix_layout="zeva_ego",
            effect_source="current_prediction", prior_variant="zeva_ego",
            initialization=initialization),
    )


def test_reference_initialization_keeps_default_fan_in_bounds_and_stable_residuals():
    torch.manual_seed(527)
    prompt_cfg, adapter_cfg = _configs("zeva_ego")
    prompt, adapter = CausalPromptEncoder(prompt_cfg), ExactZevaPolicyInjectionAdapter(adapter_cfg)
    for module in (prompt.pim_query, prompt.pim_key, prompt.pim_value,
                   adapter.behavior_global_projector, adapter.behavior_effect_projector):
        bound = 1 / math.sqrt(module.in_features)
        assert module.weight.abs().max() <= bound
        torch.testing.assert_close(module.weight.square().mean(), torch.tensor(bound ** 2 / 3), rtol=.06, atol=0)
        if module.bias is not None:
            assert module.bias.count_nonzero() == module.bias.numel()
            assert module.bias.abs().max() <= bound
    assert adapter.pim_to_global.weight.count_nonzero() == 0
    assert adapter.pim_to_global.bias.count_nonzero() == 0
    assert adapter.action_prior_adapter.weight.count_nonzero() == 0
    assert adapter.action_prior_adapter.bias.count_nonzero() == 0
    torch.testing.assert_close(adapter.prefix_project.weight.std(), torch.tensor(.002), rtol=.05, atol=0)


def test_reference_initialization_loads_existing_predictive_weights_exactly():
    torch.manual_seed(223)
    prompt_cfg, adapter_cfg = _configs("legacy_xavier")
    legacy_prompt = CausalPromptEncoder(prompt_cfg)
    legacy_adapter = ExactZevaPolicyInjectionAdapter(adapter_cfg)
    assert legacy_prompt.pim_value.weight.abs().max() > 1 / math.sqrt(64)
    assert legacy_adapter.behavior_global_projector.bias.count_nonzero() == 0
    prompt = CausalPromptEncoder(replace(prompt_cfg, initialization="zeva_ego"))
    adapter = ExactZevaPolicyInjectionAdapter(replace(adapter_cfg, initialization="zeva_ego"))
    prompt.load_state_dict(legacy_prompt.state_dict(), strict=True)
    adapter.load_state_dict(legacy_adapter.state_dict(), strict=True)
    for old, loaded in ((legacy_prompt, prompt), (legacy_adapter, adapter)):
        for key, value in old.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
