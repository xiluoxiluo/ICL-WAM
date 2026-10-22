"""Predictive effects reach action through real, differently-sized experts."""
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace
import random

import pytest
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from fastwam.zeva import CausalPromptConfig, CausalPromptEncoder
from fastwam.zeva.behavior_prefix_adapter import ExactZevaPolicyInjectionAdapter, gaussian_prior_nll
from fastwam.zeva.checkpoint import save_addon_checkpoint, load_addon_checkpoint
from fastwam.zeva.ego_pim_schedule import apply_history_kind, select_history_kind, eap_lr_factor
from fastwam.trainer import Wan22Trainer
from test_separate_prefix_integration import _model, _sample


def predictive_model(stage="policy_injection"):
    model = _model(stage=stage)
    config = replace(model.zeva_behavior_prefix_adapter.config, phase_dim=8, effect_dim=8,
                     prior_variant="zeva_ego", effect_source="current_prediction", pim_variant="cross_attempt")
    model.attach_zeva_addon(CausalPromptEncoder(CausalPromptConfig(
        global_dim=8, phase_dim=8, effect_dim=8, brief_length=2, persistent_length=3,
        hidden_dim=8, num_heads=2, pim_variant="cross_attempt", pim_context_dropout=0, pim_support_dropout=0,
    )), ExactZevaPolicyInjectionAdapter(config))
    model.zeva_training_recipe = "ego_adapter_warmup" if stage == "pim_adapter" else "frozen_parent"
    model.set_zeva_training_stage(stage).configure_zeva_trainable_state()
    return model


def predictive_sample():
    sample = _sample()
    sample.update(phase=torch.randn(3, 8), bit_effects=torch.randn(3, 2, 8),
                  pim_phases=torch.randn(3, 3, 8), pim_effects=torch.randn(3, 3, 8),
                  current_effect=torch.randn(3, 8), current_effect_mask=torch.ones(3, dtype=torch.bool))
    return sample


def test_first_boundary_has_predicted_effect_and_eap_prior_does_not_read_bit():
    model, sample = predictive_model(), predictive_sample()
    sample["bit_mask"].zero_()
    adapter = model.zeva_behavior_prefix_adapter
    args = (sample["task_context"], sample["phase"], sample["bit_effects"], sample["bit_mask"],
            torch.zeros(3, 8), sample["pim_mask"])
    mean, std = adapter.prior_for_prompt(*args, enable_pim=False)
    changed = list(args)
    changed[2] = torch.full_like(changed[2], float("nan"))
    other_mean, other_std = adapter.prior_for_prompt(*changed, enable_pim=False)
    torch.testing.assert_close(mean, other_mean, rtol=0, atol=0)
    torch.testing.assert_close(std, other_std, rtol=0, atol=0)
    loss, metrics = model(sample)
    assert metrics["prefix/effect_valid_fraction"] == 1.0
    assert metrics["memory/bit_count"] == 0.0
    loss.backward()
    assert adapter.behavior_effect_projector.weight.grad.norm() > 0
    assert adapter.prior.temporal_waypoint[0].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.mot.parameters())


def test_same_explicit_noise_effect_and_memory_change_action_without_advancing_rng():
    model = predictive_model("pim_adapter").eval()
    kwargs = dict(prompt=None, input_image=torch.zeros(1, 3, 16, 16),
                  context=torch.zeros(1, 3, 6), context_mask=torch.ones(1, 3, dtype=torch.bool),
                  action_horizon=32, num_inference_steps=2, causal_prompt=torch.randn(1, 8),
                  pim_mask=torch.ones(1, 3, dtype=torch.bool), zeva_task_tokens=torch.randn(1, 8),
                  zeva_current_effect=torch.randn(1, 8), zeva_current_effect_mask=torch.ones(1, dtype=torch.bool),
                  initial_action_noise=torch.randn(1, 32, 14), enable_action_prior=False)
    noise = kwargs["initial_action_noise"].clone()
    rng = torch.get_rng_state().clone()
    on = model.infer_action_zeva(**kwargs, zeva_mode="pim_on")["action"]
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(kwargs["initial_action_noise"], noise, rtol=0, atol=0)
    off = model.infer_action_zeva(**kwargs, zeva_mode="zeva_stage2")["action"]
    assert not torch.allclose(on, off, rtol=1e-6, atol=1e-7)
    changed = {**kwargs, "zeva_current_effect": kwargs["zeva_current_effect"] + 3}
    changed_action = model.infer_action_zeva(**changed, zeva_mode="pim_on")["action"]
    assert not torch.allclose(on, changed_action, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="mutually exclusive"):
        model.infer_action_zeva(**kwargs, seed=1)
    with pytest.raises(ValueError, match="shape"):
        model.infer_action_zeva(**{**kwargs, "initial_action_noise": noise[:, :24]})


def test_reference_prior_nll_sums_action_coordinates_then_averages_valid_times():
    target, mean = torch.randn(2, 32, 14), torch.randn(2, 32, 14)
    std = torch.rand_like(mean) + 0.1
    valid = torch.rand(2, 32) > 0.5
    actual = gaussian_prior_nll(target, mean, std, valid, reference_reduction=True)
    expected = -torch.distributions.Normal(mean, std).log_prob(target).sum(-1)[valid].mean()
    torch.testing.assert_close(actual, expected)


def test_predictive_checkpoint_cannot_be_reinterpreted_as_latest_completed(tmp_path):
    model = predictive_model()
    path = tmp_path / "predictive.pt"
    pair = model.zeva_prompt_encoder, model.zeva_behavior_prefix_adapter
    save_addon_checkpoint(path, *pair, step=10, config={}, base_checkpoint_sha256="base", cte_checkpoint_sha256="new")
    payload = load_addon_checkpoint(path, *pair)
    assert payload["addon_schema"] == "zeva_ego_prediction_v1"
    assert payload["memory_semantics"] == "predicted_effect_trace"
    old = _model()
    with pytest.raises(ValueError, match="prior_variant"):
        load_addon_checkpoint(path, old.zeva_prompt_encoder, old.zeva_behavior_prefix_adapter)
    payload["memory_semantics"] = "completed_effect"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="semantics"):
        load_addon_checkpoint(path, *pair)


def test_pim_warmup_freezes_eap_updates_then_jointly_updates_with_frozen_foundation():
    model = predictive_model("pim_adapter")
    adapter = model.zeva_behavior_prefix_adapter
    pim = list(model.zeva_prompt_encoder.parameters()) + list(adapter.pim_adapter_parameters())
    eap = list(adapter.policy_injection_parameters())
    optimizer = torch.optim.AdamW([{"params": pim, "lr": 5e-4}, {"params": eap, "lr": 1e-4}])
    scheduler = LambdaLR(optimizer, [lambda step: 1.0, lambda step: eap_lr_factor(step, 2)])
    original = deepcopy(adapter.behavior_effect_projector.state_dict())
    original_pim = adapter.pim_to_global.weight.detach().clone()
    for step in range(3):
        optimizer.zero_grad()
        loss, _ = model(predictive_sample())
        loss.backward()
        assert all(p.grad is None for p in model.mot.parameters())
        optimizer.step()
        scheduler.step()
        if step < 2:
            for key, value in original.items():
                torch.testing.assert_close(adapter.behavior_effect_projector.state_dict()[key], value, rtol=0, atol=0)
    assert not torch.equal(adapter.behavior_effect_projector.weight, original["weight"])
    assert not torch.equal(adapter.pim_to_global.weight, original_pim)
    model.zeva_parameter_report()


def test_history_curriculum_is_step_based_and_empty_preserves_current_prediction():
    kinds = ("matched", "far", "cross_condition")
    rng = torch.get_rng_state().clone()

    def modes(steps, available=kinds, seed=1):
        return [select_history_kind(i, seed=seed, warmup_steps=500, available_kinds=available) for i in steps]

    assert modes(range(16)) == [0, 0, 0, 0, 0, 1, 1, 2] * 2
    assert 3 not in modes(range(500))
    assert modes(range(500, 520)) == [0, 0, 0, 0, 0, 1, 1, 2, 3, 3] * 2
    # Filter unavailable kinds before cycling. The post-warmup cycle still
    # indexes global step 500 (500 % 9 == 5), rather than restarting at zero.
    assert modes(range(14), kinds[:2]) == [0, 0, 0, 0, 0, 1, 1] * 2
    assert modes(range(500, 518), kinds[:2]) == [1, 1, 3, 3, 0, 0, 0, 0, 0] * 2
    assert 2 not in modes(range(560), kinds[:2])
    assert modes(range(500, 520), seed=999) == modes(range(500, 520))
    assert torch.equal(torch.get_rng_state(), rng)
    sample = predictive_sample()
    sample.update(pim_history_available=torch.ones(3, 3, dtype=torch.bool),
                  pim_history_phases=torch.randn(3, 3, 3, 8), pim_history_effects=torch.randn(3, 3, 3, 8),
                  pim_history_masks=torch.ones(3, 3, 3, dtype=torch.bool), pim_history_scores=torch.randn(3, 3, 3))
    selected = apply_history_kind(sample, 1)
    assert selected["pim_history_kind"].tolist() == [1, 1, 1]
    assert selected["pim_history_count"].tolist() == [1, 1, 1]
    empty = apply_history_kind(sample, 3)
    assert not empty["pim_mask"].any()
    assert torch.equal(empty["current_effect"], sample["current_effect"])
    assert torch.equal(empty["current_effect_mask"], sample["current_effect_mask"])


def test_stage2_rank_rng_restores_next_draws_and_rejects_different_topology():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=2, process_index=1)
    trainer.global_step = 500
    state = trainer._rank_rng_state()
    expected = (torch.randn(2, 32, 14), np.random.rand(4), random.random())
    torch.rand(17)
    np.random.rand(13)
    random.random()
    trainer._restore_rank_rng(state)
    torch.testing.assert_close(torch.randn(2, 32, 14), expected[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(4), expected[1])
    assert random.random() == expected[2]
    for key in ("step", "world_size", "rank"):
        bad = {**state, key: state[key] + 1}
        with pytest.raises(ValueError, match=key):
            trainer._restore_rank_rng(bad)


def test_new_curriculum_resume_rejects_changed_warmup_or_effective_batch():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.zeva_training_recipe = "ego_adapter_warmup"
    trainer.cfg = {"zeva_pim_warmup_steps": 500, "zeva_eap_learning_rate": 1e-5,
                   "learning_rate": 5e-5, "seed": 20260919, "batch_size": 1,
                   "gradient_accumulation_steps": 64}
    payload = {"training_recipe": "ego_adapter_warmup", "config": dict(trainer.cfg)}
    trainer._validate_zeva_resume_recipe(payload)
    for key in trainer.cfg:
        changed = {**payload, "config": {**trainer.cfg, key: trainer.cfg[key] * 2}}
        with pytest.raises(ValueError, match=key):
            trainer._validate_zeva_resume_recipe(changed)
