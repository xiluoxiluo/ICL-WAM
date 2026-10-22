"""Independent prefix topology and causal completed-effect regressions."""

from dataclasses import replace

import pytest
import torch

from fastwam.zeva.behavior_prefix_adapter import (
    ExactZevaPolicyInjectionAdapter,
    ExactZevaPolicyInjectionConfig,
    select_latest_completed_effect,
)


def _config(**overrides):
    cfg = ExactZevaPolicyInjectionConfig(
        memory_dim=8, context_dim=12, global_dim=8, phase_dim=4,
        effect_dim=4, effect_history_length=3, action_dim=14,
        action_horizon=32, prior_hidden_dim=8, prior_num_heads=2,
        action_hidden_dim=8, prior_dropout_rate=0.0,
        pim_variant="cross_attempt", prefix_layout="zeva_ego",
    )
    return replace(cfg, **overrides)


def _inputs(batch=3):
    return dict(
        task_tokens=torch.randn(batch, 8),
        causal_prompt=torch.randn(batch, 8),
        pim_mask=torch.ones(batch, 5, dtype=torch.bool),
        bit_effects=torch.randn(batch, 3, 4),
        bit_mask=torch.ones(batch, 3, dtype=torch.bool),
    )


def test_latest_completed_effect_handles_sparse_masks_and_nan_padding():
    effects = torch.arange(4 * 3 * 2, dtype=torch.float32).view(4, 3, 2)
    mask = torch.tensor([
        [True, False, True],
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ])
    expected = torch.stack((effects[0, 2], effects[1, 1], effects[2, 0], torch.zeros(2)))
    effects[~mask] = torch.nan
    actual, valid = select_latest_completed_effect(effects, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert valid.tolist() == [True, True, True, False]
    empty, empty_valid = select_latest_completed_effect(
        torch.empty(2, 0, 4), torch.empty(2, 0, dtype=torch.bool),
    )
    assert empty.shape == (2, 4)
    assert torch.count_nonzero(empty) == 0
    assert not empty_valid.any()


def test_independent_global_effect_and_pim_positions_and_one_global_conditioning():
    torch.manual_seed(101)
    adapter = ExactZevaPolicyInjectionAdapter(_config())
    inputs = _inputs()
    with torch.no_grad():
        adapter.pim_to_global.weight.normal_(std=0.2)
    tokens, valid = adapter.behavior_prefix_tokens(**inputs)
    assert tokens.shape == (3, 3, 12)
    assert valid.all()
    expected_global = adapter.behavior_global_projector(
        inputs["task_tokens"] + adapter.pim_to_global(inputs["causal_prompt"])
    )
    torch.testing.assert_close(tokens[:, 0], expected_global, rtol=0, atol=0)
    torch.testing.assert_close(
        tokens[:, 1], adapter.behavior_effect_projector(inputs["bit_effects"][:, -1]),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        tokens[:, 2], adapter.prefix_project(inputs["causal_prompt"]), rtol=0, atol=0,
    )
    # With the global-conditioning map at zero, PIM can change only its own
    # prefix slot. A combined global/PIM slot would fail this independence.
    with torch.no_grad():
        adapter.pim_to_global.weight.zero_()
    before, _ = adapter.behavior_prefix_tokens(**inputs)
    changed = dict(inputs, causal_prompt=inputs["causal_prompt"] + 2.0)
    after, _ = adapter.behavior_prefix_tokens(**changed)
    torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
    assert not torch.allclose(before[:, 2], after[:, 2])
    changed_effect = dict(inputs, bit_effects=inputs["bit_effects"].clone())
    changed_effect["bit_effects"][:, -1] += 3.0
    effect_after, _ = adapter.behavior_prefix_tokens(**changed_effect)
    torch.testing.assert_close(before[:, [0, 2]], effect_after[:, [0, 2]], rtol=0, atol=0)
    assert not torch.allclose(before[:, 1], effect_after[:, 1])


def test_empty_effect_and_pim_slots_are_invalid_even_after_biases_move():
    torch.manual_seed(103)
    adapter = ExactZevaPolicyInjectionAdapter(_config())
    inputs = _inputs()
    inputs["pim_mask"][0] = False
    inputs["bit_mask"][0] = False
    inputs["bit_mask"][1] = torch.tensor([True, False, False])
    inputs["causal_prompt"][0] = torch.nan
    inputs["bit_effects"][~inputs["bit_mask"]] = torch.nan
    with torch.no_grad():
        adapter.pim_to_global.weight.normal_(std=0.3)
        adapter.prefix_project.bias.fill_(2.0)
        adapter.behavior_effect_projector.bias.fill_(3.0)
    tokens, valid = adapter.behavior_prefix_tokens(**inputs)
    assert valid.tolist() == [[True, False, False], [True, True, True], [True, True, True]]
    assert torch.isfinite(tokens).all()
    assert torch.count_nonzero(tokens[0, 1:]) == 0
    torch.testing.assert_close(
        tokens[0, 0], adapter.behavior_global_projector(inputs["task_tokens"])[0],
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        tokens[1, 1], adapter.behavior_effect_projector(inputs["bit_effects"][1, 0]),
        rtol=1e-6, atol=1e-6,
    )
    # Disabling PIM suppresses its key as well as both of its value paths.
    off, off_valid = adapter.behavior_prefix_tokens(**inputs, enable_pim=False)
    assert not off_valid[:, 2].any()
    assert torch.count_nonzero(off[:, 2]) == 0
    torch.testing.assert_close(
        off[:, 0], adapter.behavior_global_projector(inputs["task_tokens"]),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(tokens[:, 1], off[:, 1], rtol=0, atol=0)
    # A trained nonzero projector bias must not resurrect an empty token.
    assert not off_valid[0, 1]
    assert torch.count_nonzero(off[0, 1]) == 0


def test_masked_padding_cannot_poison_prefix_projection_gradients():
    torch.manual_seed(107)
    adapter = ExactZevaPolicyInjectionAdapter(_config())
    inputs = _inputs()
    inputs["pim_mask"][0] = False
    inputs["bit_mask"][0] = False
    inputs["bit_mask"][1, 1:] = False
    inputs["causal_prompt"][0] = torch.nan
    inputs["bit_effects"][~inputs["bit_mask"]] = torch.nan
    tokens, valid = adapter.behavior_prefix_tokens(**inputs)
    tokens.square().sum().backward()
    for module in (
        adapter.behavior_global_projector,
        adapter.behavior_effect_projector,
        adapter.prefix_project,
        adapter.pim_to_global,
    ):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    assert adapter.behavior_effect_projector.weight.grad.abs().sum() > 0
    assert adapter.prefix_project.weight.grad.abs().sum() > 0
    assert adapter.pim_to_global.weight.grad.abs().sum() > 0
    assert not adapter.pim_gate.requires_grad
    assert adapter.pim_gate.grad is None
    assert valid[:, 0].all()


def test_effect_projection_belongs_to_stage2a_parent_not_pim_training():
    for variant in ("legacy", "cross_attempt"):
        adapter = ExactZevaPolicyInjectionAdapter(_config(pim_variant=variant))
        policy_ids = {id(p) for p in adapter.policy_injection_parameters()}
        pim_ids = {id(p) for p in adapter.pim_adapter_parameters()}
        effect_ids = {id(p) for p in adapter.behavior_effect_projector.parameters()}
        assert effect_ids <= policy_ids
        assert effect_ids.isdisjoint(pim_ids)
        assert policy_ids.isdisjoint(pim_ids)
        assert id(adapter.pim_gate) not in pim_ids


def test_legacy_wrapper_preserves_combined_slot_and_legacy_gate():
    torch.manual_seed(109)
    adapter = ExactZevaPolicyInjectionAdapter(
        _config(prefix_layout="legacy_merged", pim_variant="legacy")
    )
    inputs = _inputs()
    inputs["pim_mask"][0] = False
    with torch.no_grad():
        adapter.pim_gate.fill_(0.2)
    old = adapter.behavior_prefix_slot(
        inputs["task_tokens"], inputs["causal_prompt"], inputs["pim_mask"],
    )
    tokens, valid = adapter.behavior_prefix_tokens(**inputs)
    expected = adapter.behavior_global_projector(inputs["task_tokens"])[:, None]
    expected = expected + (
        torch.tanh(adapter.pim_gate) * adapter.prefix_project(inputs["causal_prompt"])[:, None]
        * inputs["pim_mask"].any(dim=-1)[:, None, None]
    )
    assert tokens.shape == (3, 1, 12)
    assert valid.shape == (3, 1)
    assert valid.all()
    torch.testing.assert_close(tokens, old, rtol=0, atol=0)
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
    assert not hasattr(adapter, "behavior_effect_projector")
    assert adapter.pim_gate.requires_grad


def test_new_layout_requires_explicit_completed_effect_inputs():
    adapter = ExactZevaPolicyInjectionAdapter(_config())
    inputs = _inputs()
    with pytest.raises(ValueError, match="requires bit_effects and bit_mask"):
        adapter.behavior_prefix_tokens(
            inputs["task_tokens"], inputs["causal_prompt"], inputs["pim_mask"],
        )
    with pytest.raises(ValueError, match="independent prefix tokens"):
        adapter.behavior_prefix_slot(
            inputs["task_tokens"], inputs["causal_prompt"], inputs["pim_mask"],
        )
    with pytest.raises(ValueError, match="prefix_layout"):
        _config(prefix_layout="unknown")
    with pytest.raises(ValueError, match="effect_source"):
        _config(effect_source="predicted_current")
