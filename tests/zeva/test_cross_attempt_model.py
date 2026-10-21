"""CPU regressions for the opt-in completed-attempt PIM model path."""

from dataclasses import replace

import pytest
import torch

from fastwam.zeva.behavior_prefix_adapter import (
    ExactZevaPolicyInjectionAdapter,
    ExactZevaPolicyInjectionConfig,
    gaussian_prior_nll,
)
from fastwam.zeva.causal_prompt import CausalPromptConfig, CausalPromptEncoder
from fastwam.zeva.checkpoint import load_addon_checkpoint, save_addon_checkpoint


def _configs():
    prompt = CausalPromptConfig(
        global_dim=8, phase_dim=4, effect_dim=4, brief_length=2,
        persistent_length=12, hidden_dim=8, num_heads=2,
        pim_variant="cross_attempt",
    )
    adapter = ExactZevaPolicyInjectionConfig(
        memory_dim=8, context_dim=8, global_dim=8, phase_dim=4,
        effect_dim=4, effect_history_length=2, action_dim=14,
        action_horizon=32, prior_hidden_dim=8, prior_num_heads=2,
        action_hidden_dim=8, prior_dropout_rate=0.0,
        pim_variant="cross_attempt",
    )
    return prompt, adapter


def _inputs(batch=2, entries=7):
    return dict(
        task_tokens=torch.randn(batch, 8),
        current_phase=torch.randn(batch, 4),
        bit_effects=torch.randn(batch, 2, 4),
        bit_mask=torch.ones(batch, 2, dtype=torch.bool),
        pim_phases=torch.randn(batch, entries, 4),
        pim_effects=torch.randn(batch, entries, 4),
        pim_mask=torch.ones(batch, entries, dtype=torch.bool),
    )


def _prior(adapter, inputs, prompt, *, enable_pim=True):
    return adapter.prior_for_prompt(
        inputs["task_tokens"], inputs["current_phase"], inputs["bit_effects"],
        inputs["bit_mask"], prompt, inputs["pim_mask"], enable_pim=enable_pim,
    )


def test_readout_handles_variable_candidates_empty_rows_and_nan_padding():
    torch.manual_seed(17)
    cfg, _ = _configs()
    encoder = CausalPromptEncoder(cfg)
    inputs = _inputs()
    inputs["pim_mask"][0] = False
    inputs["pim_mask"][1, -2:] = False
    valid_inputs = {name: value.clone() for name, value in inputs.items()}
    inputs["pim_phases"][~inputs["pim_mask"]] = torch.nan
    inputs["pim_effects"][~inputs["pim_mask"]] = torch.nan
    actual = encoder(**inputs)
    torch.testing.assert_close(actual, encoder(**valid_inputs), rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0]) == 0
    actual.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert torch.count_nonzero(encoder(**_inputs(entries=0))) == 0
    with pytest.raises(ValueError, match="capacity"):
        encoder(**_inputs(entries=13))


def test_readout_uses_candidates_beyond_old_top4_and_is_set_order_invariant():
    torch.manual_seed(23)
    cfg, _ = _configs()
    encoder = CausalPromptEncoder(cfg)
    inputs = _inputs()
    original = encoder(**inputs)
    permutation = torch.tensor([6, 4, 2, 0, 1, 5, 3])
    reordered = dict(inputs)
    for name in ("pim_phases", "pim_effects", "pim_mask"):
        reordered[name] = inputs[name][:, permutation]
    torch.testing.assert_close(encoder(**reordered), original, rtol=1e-5, atol=1e-6)
    changed = dict(inputs)
    changed["pim_effects"] = inputs["pim_effects"].clone()
    changed["pim_effects"][:, 6] += 20.0
    assert not torch.allclose(encoder(**changed), original)


def test_empty_and_disabled_memory_preserve_parent_after_training_weights_move():
    torch.manual_seed(31)
    prompt_cfg, adapter_cfg = _configs()
    encoder = CausalPromptEncoder(prompt_cfg)
    adapter = ExactZevaPolicyInjectionAdapter(adapter_cfg)
    inputs = _inputs()
    prompt = encoder(**inputs)
    with torch.no_grad():
        adapter.pim_to_global.weight.normal_(std=0.3)
        adapter.prefix_project.bias.fill_(0.8)
    parent_prefix = adapter.behavior_global_projector(inputs["task_tokens"])[:, None]
    parent_mean, parent_std = adapter.prior(
        inputs["task_tokens"], inputs["current_phase"], inputs["bit_effects"], inputs["bit_mask"],
    )
    for enabled in (False, True):
        local = dict(inputs)
        if enabled:
            local["pim_mask"] = torch.zeros_like(inputs["pim_mask"])
        prefix = adapter.behavior_prefix_slot(
            local["task_tokens"], prompt, local["pim_mask"], enable_pim=enabled,
        )
        mean, std = _prior(adapter, local, prompt, enable_pim=enabled)
        torch.testing.assert_close(prefix, parent_prefix, rtol=0, atol=0)
        torch.testing.assert_close(mean, parent_mean, rtol=0, atol=0)
        torch.testing.assert_close(std, parent_std, rtol=0, atol=0)
    active_mean, _ = _prior(adapter, inputs, prompt)
    assert not torch.allclose(active_mean, parent_mean)
    # The same original global token is conditioned exactly once per path.
    expected_prefix = adapter.behavior_global_projector(
        adapter.condition_task_tokens(inputs["task_tokens"], prompt, inputs["pim_mask"])
    )[:, None] + adapter.prefix_project(prompt)[:, None]
    torch.testing.assert_close(
        adapter.behavior_prefix_slot(inputs["task_tokens"], prompt, inputs["pim_mask"]),
        expected_prefix,
    )


def test_initial_prefix_trains_readout_while_prior_gradients_reach_zero_global_map():
    torch.manual_seed(37)
    prompt_cfg, adapter_cfg = _configs()
    encoder = CausalPromptEncoder(prompt_cfg)
    adapter = ExactZevaPolicyInjectionAdapter(adapter_cfg)
    for p in adapter.policy_injection_parameters():
        p.requires_grad_(False)
    inputs = _inputs()
    prompt = encoder(**inputs)
    parent_mean, parent_std = _prior(adapter, inputs, prompt, enable_pim=False)
    mean, std = _prior(adapter, inputs, prompt)
    torch.testing.assert_close(mean, parent_mean, rtol=0, atol=0)
    torch.testing.assert_close(std, parent_std, rtol=0, atol=0)
    nll = gaussian_prior_nll(torch.randn_like(mean), mean, std)
    nll.backward(retain_graph=True)
    assert adapter.pim_to_global.weight.grad is not None
    assert adapter.pim_to_global.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in adapter.prior.parameters())
    encoder.zero_grad(set_to_none=True)
    prefix = adapter.behavior_prefix_slot(inputs["task_tokens"], prompt, inputs["pim_mask"])
    (prefix * torch.randn_like(prefix)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    assert not adapter.pim_gate.requires_grad
    assert adapter.pim_gate.grad is None


def test_cross_attempt_rejects_hidden_encoder_dropout():
    cfg, _ = _configs()
    with pytest.raises(ValueError, match="explicit sampled memory masks"):
        replace(cfg, pim_context_dropout=0.2)


def test_checkpoint_variant_is_explicit_and_stage2a_import_is_complete(tmp_path):
    prompt_cfg, adapter_cfg = _configs()
    legacy_prompt = CausalPromptEncoder(replace(prompt_cfg, pim_variant="legacy"))
    legacy_adapter = ExactZevaPolicyInjectionAdapter(replace(adapter_cfg, pim_variant="legacy"))
    prompt = CausalPromptEncoder(prompt_cfg)
    adapter = ExactZevaPolicyInjectionAdapter(adapter_cfg)
    path = tmp_path / "stage2a.pt"
    save_addon_checkpoint(path, legacy_prompt, legacy_adapter, 10, {}, "base", "cte")
    with pytest.raises(ValueError, match="pim_variant mismatch"):
        load_addon_checkpoint(path, prompt, adapter)
    prefix_before = adapter.prefix_project.weight.detach().clone()
    load_addon_checkpoint(path, prompt, adapter, load_scope="policy_injection")
    for source, target in zip(legacy_adapter.policy_injection_parameters(), adapter.policy_injection_parameters()):
        torch.testing.assert_close(source, target, rtol=0, atol=0)
    torch.testing.assert_close(adapter.prefix_project.weight, prefix_before, rtol=0, atol=0)
    payload = torch.load(path, weights_only=False)
    payload["behavior_prefix_adapter"].pop("prior.global_to_anchors.weight")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incomplete policy-injection"):
        load_addon_checkpoint(path, prompt, adapter, load_scope="policy_injection")
    new_path = tmp_path / "new_stage2b.pt"
    save_addon_checkpoint(new_path, prompt, adapter, 20, {}, "base", "cte", training_stage="pim_adapter")
    restored = load_addon_checkpoint(new_path, prompt, adapter)
    assert restored["addon_schema"] == "cross_attempt_v1"
    with pytest.raises(ValueError, match="Stage2A"):
        load_addon_checkpoint(new_path, prompt, adapter, load_scope="policy_injection")
