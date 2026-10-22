"""Causal/indexing/streaming contracts for the new CTE, on explicit CPU test backends."""

from dataclasses import replace
import math
import sys

import pytest
import torch
from torch.nn import functional as F

from fastwam.zeva.ego_cte import ZevaEgoCTE, ZevaEgoCTEConfig, zeva_ego_cte_loss


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    values = dict(hidden_dim=16, phase_dim=16, effect_dim=16, retrieval_dim=8,
                  aux_phase_dim=8, num_layers=2, image_size=8, dropout=0,
                  temporal_backend="gru_test", vision_backbone="tiny_test", vision_pretrained=False)
    values.update(kwargs)
    return ZevaEgoCTEConfig(**values)


def inputs(batch=2, length=3):
    generator = torch.Generator().manual_seed(12)
    return (torch.rand(batch, length + 1, 3, 3, 8, 8, generator=generator) * 2 - 1,
            torch.randn(batch, length, 24, 14, generator=generator))


def test_forward_and_online_are_the_same_causal_computation():
    model = ZevaEgoCTE(config()).eval()
    images, actions = inputs()
    outputs = model(images, actions)
    assert outputs["phase"].shape == (2, 4, 16)
    assert outputs["predicted_effect"].shape == (2, 4, 16)
    assert outputs["retrieval"].shape == (2, 4, 8)
    assert outputs["pred_act"].shape == actions.shape
    assert model.cfg.transition_steps == 24 and model.cfg.effect_window_transitions == 1
    assert model.config is model.cfg
    cache = None
    for boundary in range(images.shape[1]):
        phase, effect, cache = model.step(images[:, boundary], None if boundary == 0 else actions[:, boundary - 1], cache)
        torch.testing.assert_close(phase, outputs["phase"][:, boundary], atol=5e-6, rtol=5e-5)
        torch.testing.assert_close(effect, outputs["predicted_effect"][:, boundary], atol=5e-6, rtol=5e-5)
        assert cache.offset == boundary + 1
    assert len(cache.temporal) == 3 * model.cfg.num_layers
    reset_phase, reset_effect, reset = model.step(images[:, 0])
    torch.testing.assert_close(reset_phase, outputs["phase"][:, 0], atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(reset_effect, outputs["predicted_effect"][:, 0], atol=5e-6, rtol=5e-5)
    assert reset.offset == 1


def test_current_action_and_future_images_cannot_affect_current_prediction_in_train():
    model = ZevaEgoCTE(config()).train()
    assert not any(module.training for module in model.vision_encoder.modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    assert not model.target_vision_encoder.training
    images, actions = inputs()
    baseline = model(images, actions)
    changed_images, changed_actions = images.clone(), actions.clone()
    changed_images[:, 3:] = -changed_images[:, 3:]
    changed_actions[:, 2:] += 20
    changed = model(changed_images, changed_actions)
    for key in ("phase", "predicted_effect", "retrieval"):
        torch.testing.assert_close(baseline[key][:, :3], changed[key][:, :3])
    assert not torch.allclose(baseline["target_effect"][:, 2], changed["target_effect"][:, 2])
    actions = actions.requires_grad_()
    images = images.requires_grad_()
    output = model(images, actions)["predicted_effect"][:, 1].square().sum()
    action_gradient, image_gradient = torch.autograd.grad(output, (actions, images))
    assert action_gradient[:, 0].abs().sum() > 0
    assert torch.count_nonzero(action_gradient[:, 1:]) == 0
    assert image_gradient[:, :2].abs().sum() > 0
    assert torch.count_nonzero(image_gradient[:, 2:]) == 0


def test_effect_targets_are_detached_ema_deltas_and_the_predictor_trains():
    model = ZevaEgoCTE(config())
    images, actions = inputs()
    outputs = model(images, actions)
    with torch.no_grad():
        target = model._visual(images, target=True)
    torch.testing.assert_close(outputs["target_effect"], target[:, 1:] - target[:, :-1])
    assert not outputs["target_effect"].requires_grad
    losses = zeva_ego_cte_loss(outputs, actions, outputs["valid"], torch.tensor([0, 0]))
    losses["total"].backward()
    assert model.effect_predictor[-1].weight.grad.abs().sum() > 0
    assert model.action_sos.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.target_vision_encoder.parameters())


def test_padding_cannot_poison_history_or_losses():
    model = ZevaEgoCTE(config()).eval()
    images, actions = inputs()
    valid = torch.tensor([[True, False, False], [True, True, True]])
    clean = model(images, actions, valid)
    images[0, 2:] = float("nan")
    actions[0, 1:] = float("nan")
    output = model(images, actions, valid)
    for key in ("phase", "predicted_effect", "retrieval"):
        torch.testing.assert_close(output[key], clean[key])
        assert torch.count_nonzero(output[key][0, 2:]) == 0
    losses = zeva_ego_cte_loss(output, actions, valid, torch.tensor([0, 0]))
    assert all(torch.isfinite(value) for value in losses.values())
    with pytest.raises(ValueError, match="contiguous suffix"):
        model(images, actions, torch.tensor([[True, False, True], [True, True, True]]))
    with pytest.raises(ValueError, match="nonempty"):
        zeva_ego_cte_loss(output, actions, torch.zeros_like(valid), torch.tensor([0, 0]))


def test_loss_matches_reference_dimension_sums_and_contrastive_denominator():
    valid = torch.ones(2, 2, dtype=torch.bool)
    actions = torch.zeros(2, 2, 24, 14)
    output = {
        "pred_act": torch.ones_like(actions), "pred_vis": torch.ones(2, 2, 3),
        "target_vis": torch.zeros(2, 2, 3), "pred_effect": torch.full((2, 2, 3), 2.),
        "target_effect": torch.zeros(2, 2, 3), "z_global": torch.ones(2, 2, 2),
        "z_local": F.normalize(torch.ones(2, 2, 2), dim=-1), "logit_scale": torch.tensor(0.),
    }
    losses = zeva_ego_cte_loss(output, actions, valid, torch.tensor([5, 5]))
    assert float(losses["action"]) == 14.0
    assert float(losses["vision"]) == 3.0
    assert float(losses["effect"]) == 12.0
    assert float(losses["global"]) == pytest.approx(math.log(2))
    assert float(losses["phase"]) == pytest.approx(math.log(4))
    assert float(losses["total"]) == pytest.approx(1.4 + 0.6 + 2 * math.log(2) + math.log(4) + 2.4)


def test_ema_updates_parameters_and_copies_buffers():
    model = ZevaEgoCTE(config(ema_decay=0.5))
    online = next(model.vision_encoder.parameters())
    target = next(model.target_vision_encoder.parameters())
    before = target.detach().clone()
    with torch.no_grad():
        online.add_(2)
        next(model.vision_encoder.buffers()).fill_(3)
    model.update_ema_target()
    torch.testing.assert_close(target, before + 1)
    for online_buffer, target_buffer in zip(model.vision_encoder.buffers(), model.target_vision_encoder.buffers(), strict=True):
        torch.testing.assert_close(online_buffer, target_buffer)


def test_online_contract_rejects_missing_partial_and_initial_actions():
    model = ZevaEgoCTE(config()).eval()
    images, actions = inputs()
    with pytest.raises(ValueError, match="initial boundary"):
        model.step(images[:, 0], actions[:, 0])
    _, _, cache = model.step(images[:, 0])
    with pytest.raises(ValueError, match="complete executed chunk"):
        model.step(images[:, 1], actions[:, 0, :-1], cache)
    with pytest.raises(ValueError, match="complete executed chunk"):
        model.step(images[:, 1], None, cache)
    model.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        model.step(images[:, 0])


def test_mamba_dependency_never_silently_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "mamba_ssm", None)
    with pytest.raises(ImportError, match="no GRU fallback"):
        ZevaEgoCTE(replace(config(), temporal_backend="mamba"))


def test_legacy_fused_or_wrong_horizon_input_is_rejected():
    model = ZevaEgoCTE(config())
    images, actions = inputs()
    with pytest.raises(ValueError, match="independently resized"):
        model(images[:, :, 0], actions)
    with pytest.raises(ValueError, match="complete chunks"):
        model(images, actions[:, :, :16])
