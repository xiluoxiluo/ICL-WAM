import math
from types import MethodType

import torch
from torch import nn

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.zeva import (
    BehaviorPrefixAdapter,
    BehaviorPrefixAdapterConfig,
    CausalPromptEncoder,
)


class _ActionExpert(nn.Module):
    action_dim = 14

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(14, 8)

    def prepare(self, action_tokens, timestep, context, context_mask):
        tokens = self.encoder(action_tokens)
        return tokens, timestep, torch.ones_like(tokens), context, context_mask, torch.zeros(tokens.shape[1], 1)

    def post(self, tokens):
        return tokens


class _MoT(nn.Module):
    def forward_action_with_video_cache_tensor(self, action_tokens, **kwargs):
        return action_tokens * 2.0 + 1.0


def _model():
    model = FastWAM.__new__(FastWAM)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.action_expert = _ActionExpert()
    model.mot = _MoT()
    model.zeva_enabled = True
    model.zeva_prompt_encoder = CausalPromptEncoder()
    model.zeva_behavior_prefix_adapter = BehaviorPrefixAdapter(
        BehaviorPrefixAdapterConfig(action_hidden_dim=8, num_heads=2)
    )
    return model


def test_gate_zero_action_path_matches_base_and_addon_gets_gradients():
    model = _model()
    memory = torch.randn(2, 4, 256)
    memory_mask = torch.ones(2, 4, dtype=torch.bool)
    action = torch.randn(2, 32, 14)
    timestep = torch.ones(2)
    context = torch.randn(2, 3, 4)
    context_mask = torch.ones(2, 3, dtype=torch.bool)
    attention = torch.zeros(2, 32 + 1)
    cache = [torch.zeros(2, 1, 1, 1)]

    model.eval()
    base = model._denoise_action_with_video_cache(
        action, timestep, context, context_mask, cache, cache, attention
    )
    shadow = model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention,
        memory, memory_mask, gate_override=0.0,
    )
    torch.testing.assert_close(base, shadow, rtol=0.0, atol=0.0)

    model.zeva_prompt_encoder.train()
    model.zeva_behavior_prefix_adapter.train()
    model.requires_grad_(False)
    model.zeva_prompt_encoder.requires_grad_(True)
    model.zeva_behavior_prefix_adapter.requires_grad_(True)
    task_args = (
        torch.randn(2, 256), torch.randn(2, 128), torch.randn(2, 4, 128),
        torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 4, 128),
        torch.randn(2, 4, 128), torch.ones(2, 4, dtype=torch.bool),
    )
    memory, memory_mask = model.zeva_prompt_encoder(*task_args)
    pred = model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention,
        memory, memory_mask,
    )
    pred.square().mean().backward()
    assert model.zeva_behavior_prefix_adapter.pim_gate.grad is not None
    # The exact zero gate intentionally makes the first residual update a
    # gate-only step; once the gate opens, the projector and prompt receive
    # ordinary upstream gradients.
    assert model.zeva_behavior_prefix_adapter.output.weight.grad is not None
    assert model.zeva_behavior_prefix_adapter.output.weight.grad.norm() == 0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.zeva_prompt_encoder.parameters())
    assert all(p.grad is None for p in model.action_expert.parameters())


def test_nonzero_addon_gate_changes_action_path():
    """A loaded/trained addon must influence ActionDiT before the MoT block."""
    model = _model()
    memory = torch.randn(1, 4, 256)
    memory_mask = torch.ones(1, 4, dtype=torch.bool)
    action = torch.randn(1, 32, 14)
    timestep = torch.ones(1)
    context = torch.randn(1, 3, 4)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    attention = torch.zeros(1, 33)
    cache = [torch.zeros(1, 1, 1, 1)]

    model.eval()
    with torch.no_grad():
        # Simulate a checkpoint whose residual head and scalar gate have moved
        # away from their exact no-op initialization.
        model.zeva_behavior_prefix_adapter.output.bias.fill_(1.0)
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(math.atanh(0.5))
    base = model._denoise_action_with_video_cache(
        action, timestep, context, context_mask, cache, cache, attention
    )
    conditioned = model._denoise_action_with_video_cache_zeva(
        action,
        timestep,
        context,
        context_mask,
        cache,
        cache,
        attention,
        memory,
        memory_mask,
    )
    assert not torch.equal(base, conditioned)


def test_conditioned_action_path_reports_residual_debug_metrics():
    model = _model()
    memory = torch.randn(1, 4, 256)
    memory_mask = torch.ones(1, 4, dtype=torch.bool)
    action = torch.randn(1, 32, 14)
    timestep = torch.ones(1)
    context = torch.randn(1, 3, 4)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    attention = torch.zeros(1, 33)
    cache = [torch.zeros(1, 1, 1, 1)]
    with torch.no_grad():
        model.zeva_behavior_prefix_adapter.output.weight.normal_(std=0.01)
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(math.atanh(0.5))
    debug = {}
    model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention,
        memory, memory_mask, debug=debug,
    )
    assert debug["base_action_hidden_norm"] > 0
    assert debug["memory_delta_hidden_norm"] > 0
    assert debug["conditioned_action_hidden_norm"] > 0
    assert torch.isfinite(debug["memory_residual_ratio"])


def test_memory_content_changes_action_prediction():
    """Different PIM content must affect actions, not merely a constant bias."""
    torch.manual_seed(7)
    model = _model().eval()
    with torch.no_grad():
        model.zeva_behavior_prefix_adapter.output.weight.normal_(mean=0.0, std=0.01)
        model.zeva_behavior_prefix_adapter.pim_gate.fill_(math.atanh(0.5))

    action = torch.randn(1, 32, 14)
    timestep = torch.ones(1)
    context = torch.randn(1, 3, 4)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    attention = torch.zeros(1, 33)
    cache = [torch.zeros(1, 1, 1, 1)]
    task = torch.randn(1, 256)
    phase = torch.randn(1, 128)
    bit = torch.randn(1, 4, 128)
    bit_mask = torch.ones(1, 4, dtype=torch.bool)
    pim_phase_a = torch.randn(1, 4, 128)
    pim_effect_a = torch.randn(1, 4, 128)
    pim_phase_b = pim_phase_a.clone()
    pim_effect_b = -pim_effect_a
    pim_mask = torch.ones(1, 4, dtype=torch.bool)

    memory_a, mask_a = model.zeva_prompt_encoder(
        task, phase, bit, bit_mask, pim_phase_a, pim_effect_a, pim_mask
    )
    memory_b, mask_b = model.zeva_prompt_encoder(
        task, phase, bit, bit_mask, pim_phase_b, pim_effect_b, pim_mask
    )
    assert torch.linalg.vector_norm(memory_a - memory_b) > 1e-6

    action_a = model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention, memory_a, mask_a
    )
    action_b = model._denoise_action_with_video_cache_zeva(
        action, timestep, context, context_mask, cache, cache, attention, memory_b, mask_b
    )
    delta = torch.linalg.vector_norm(action_a - action_b)
    assert torch.isfinite(delta)
    assert delta > 1e-5


def test_parameter_report_rejects_trainable_base():
    model = _model()
    model.base_probe = nn.Linear(2, 2)
    model.requires_grad_(False)
    model.zeva_prompt_encoder.requires_grad_(True)
    model.zeva_behavior_prefix_adapter.requires_grad_(True)
    report = model.zeva_parameter_report()
    assert report["trainable_count"] > 0
    model.base_probe.weight.requires_grad_(True)
    try:
        model.zeva_parameter_report()
    except AssertionError:
        pass
    else:
        raise AssertionError("base parameter was not rejected by Zeva whitelist")


def test_attach_zeva_addon_freezes_base_modules():
    model = _model()
    # _model() emulates an already-constructed FastWAM with an attached addon;
    # exercise the same invariant directly on its public attach method.
    model.zeva_enabled = False
    model.zeva_prompt_encoder = None
    model.zeva_behavior_prefix_adapter = None
    model.attach_zeva_addon(CausalPromptEncoder(), BehaviorPrefixAdapter(
        BehaviorPrefixAdapterConfig(action_hidden_dim=8, num_heads=2)
    ))
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith(("zeva_prompt_encoder.", "zeva_behavior_prefix_adapter."))
    )
    assert any(parameter.requires_grad for parameter in model.zeva_prompt_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.zeva_behavior_prefix_adapter.parameters())


def test_forward_routes_stage2_through_model_entrypoint():
    model = _model()
    called = {}

    def fake_forward(self, **kwargs):
        called.update(kwargs)
        return torch.tensor(2.0), {"loss_action": 2.0}

    model.forward_zeva_action_train = MethodType(fake_forward, model)
    sample = {
        "_training_mode": "zeva_stage2",
        "video": torch.zeros(1, 3, 1, 8, 8),
        "action": torch.zeros(1, 32, 14),
        "context": torch.zeros(1, 3, 256),
        "context_mask": torch.ones(1, 3, dtype=torch.bool),
        "phase": torch.zeros(1, 128),
        "bit_effects": torch.zeros(1, 4, 128),
        "bit_mask": torch.zeros(1, 4, dtype=torch.bool),
        "pim_phases": torch.zeros(1, 4, 128),
        "pim_effects": torch.zeros(1, 4, 128),
        "pim_mask": torch.zeros(1, 4, dtype=torch.bool),
    }
    loss, metrics = model(sample)
    assert loss.item() == 2.0
    assert metrics["memory/bit_count"] == 0.0
    assert metrics["memory/pim_count"] == 0.0
    assert called["behavior_memory"].shape == (1, 4, 256)
