"""Map causal memory tokens to FastWAM action hidden-space residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BehaviorPrefixAdapterConfig:
    memory_dim: int = 256
    action_horizon: int = 32
    action_hidden_dim: int = 1024
    num_heads: int = 8
    mlp_ratio: float = 2.0
    gate_init: float = 0.0
    # A tiny training-only offset lets the zero-initialized residual projection
    # receive gradients while alpha is still zero. Evaluation remains exactly
    # gate-controlled.
    train_gate_epsilon: float = 1.0e-3

    def __post_init__(self) -> None:
        if min(self.memory_dim, self.action_horizon, self.action_hidden_dim, self.num_heads) < 1:
            raise ValueError("adapter dimensions and action horizon must be positive")
        if self.action_hidden_dim % self.num_heads:
            raise ValueError("action_hidden_dim must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.train_gate_epsilon < 0:
            raise ValueError("train_gate_epsilon cannot be negative")


class BehaviorPrefixAdapter(nn.Module):
    def __init__(self, config: BehaviorPrefixAdapterConfig | None = None) -> None:
        super().__init__(); self.config = config or BehaviorPrefixAdapterConfig(); cfg = self.config
        self.memory_norm = nn.LayerNorm(cfg.memory_dim)
        self.memory_project = nn.Linear(cfg.memory_dim, cfg.action_hidden_dim)
        self.behavior_queries = nn.Parameter(torch.randn(1, cfg.action_horizon, cfg.action_hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(cfg.action_hidden_dim, cfg.num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(cfg.action_hidden_dim), nn.Linear(cfg.action_hidden_dim, int(cfg.mlp_ratio * cfg.action_hidden_dim)), nn.GELU(), nn.Linear(int(cfg.mlp_ratio * cfg.action_hidden_dim), cfg.action_hidden_dim))
        self.output = nn.Linear(cfg.action_hidden_dim, cfg.action_hidden_dim)
        # Required by the V1 contract: the addon is an exact no-op at its
        # initialization (tanh(alpha)=0 and residual=0).
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.pim_gate = nn.Parameter(torch.tensor(float(cfg.gate_init)))

    def forward(self, memory_tokens: Tensor, memory_mask: Tensor, action_horizon: int | None = None) -> Tensor:
        cfg = self.config; horizon = cfg.action_horizon if action_horizon is None else int(action_horizon)
        if memory_tokens.ndim != 3 or memory_tokens.shape[0:2] != memory_mask.shape or memory_tokens.shape[-1] != cfg.memory_dim:
            raise ValueError("memory_tokens/memory_mask shape mismatch")
        if horizon != cfg.action_horizon:
            raise ValueError(f"V1 action horizon is fixed at {cfg.action_horizon}")
        memory_tokens = memory_tokens.to(
            device=self.memory_project.weight.device,
            dtype=self.memory_project.weight.dtype,
        )
        memory_mask = memory_mask.to(device=memory_tokens.device, dtype=torch.bool)
        queries = self.behavior_queries.expand(memory_tokens.shape[0], -1, -1)
        keys = self.memory_project(self.memory_norm(memory_tokens))
        padding = ~memory_mask.bool(); empty = ~memory_mask.any(dim=-1); padding = padding.clone(); padding[empty, 0] = False
        attended, _ = self.cross_attention(queries, keys, keys, key_padding_mask=padding, need_weights=False)
        return self.output(attended + self.mlp(attended))

    def gated(self, memory_tokens: Tensor, memory_mask: Tensor, action_horizon: int | None = None) -> Tensor:
        gate = torch.tanh(self.pim_gate)
        if self.training and self.config.train_gate_epsilon:
            gate = gate + float(self.config.train_gate_epsilon)
        return gate * self.forward(memory_tokens, memory_mask, action_horizon)
