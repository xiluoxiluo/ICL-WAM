import math

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
    # The epsilon training gate makes the zero-initialized output projection
    # identifiable on the first optimizer step; without this gradient the
    # addon would remain a permanent no-op.
    assert model.zeva_behavior_prefix_adapter.output.weight.grad is not None
    assert model.zeva_behavior_prefix_adapter.output.weight.grad.norm() > 0
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
