"""Zeva behavior-prefix and action-prior adapters for FastWAM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .precision import ReferenceFloat32Module, reference_inference_precision


@dataclass(frozen=True)
class BehaviorPrefixAdapterConfig:
    memory_dim: int = 256
    action_horizon: int = 32
    action_hidden_dim: int = 1024
    num_heads: int = 8
    mlp_ratio: float = 2.0
    gate_init: float = 0.0

    def __post_init__(self) -> None:
        if min(self.memory_dim, self.action_horizon, self.action_hidden_dim, self.num_heads) < 1:
            raise ValueError("adapter dimensions and action horizon must be positive")
        if self.action_hidden_dim % self.num_heads:
            raise ValueError("action_hidden_dim must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")


class BehaviorPrefixAdapter(nn.Module):
    def __init__(self, config: BehaviorPrefixAdapterConfig | None = None) -> None:
        super().__init__(); self.config = config or BehaviorPrefixAdapterConfig(); cfg = self.config
        self.memory_norm = nn.LayerNorm(cfg.memory_dim)
        self.memory_project = nn.Linear(cfg.memory_dim, cfg.action_hidden_dim)
        self.behavior_queries = nn.Parameter(torch.randn(1, cfg.action_horizon, cfg.action_hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(cfg.action_hidden_dim, cfg.num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(cfg.action_hidden_dim), nn.Linear(cfg.action_hidden_dim, int(cfg.mlp_ratio * cfg.action_hidden_dim)), nn.GELU(), nn.Linear(int(cfg.mlp_ratio * cfg.action_hidden_dim), cfg.action_hidden_dim))
        self.output = nn.Linear(cfg.action_hidden_dim, cfg.action_hidden_dim)
        # Match Zeva's stable initialization: the residual projector is
        # expressive from the first step, while tanh(gate_init)=0 keeps the
        # complete addon an exact no-op until the scalar gate moves.
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
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
        return torch.tanh(self.pim_gate) * self.forward(memory_tokens, memory_mask, action_horizon)


@dataclass(frozen=True)
class ExactZevaPolicyInjectionConfig:
    """FastWAM adaptation of Zeva's prefix and action-prior injection paths."""

    memory_dim: int = 256
    context_dim: int = 4096
    global_dim: int = 256
    phase_dim: int = 128
    effect_dim: int = 128
    effect_history_length: int = 4
    action_dim: int = 14
    action_horizon: int = 32
    prior_hidden_dim: int = 256
    prior_num_heads: int = 4
    action_hidden_dim: int = 1024
    leading_condition_steps: int = 0
    prior_loss_weight: float = 0.01
    prior_dropout_rate: float = 0.4
    prior_inference_guidance_scale: float = 0.5
    prompt_gate_init: float = 0.0
    min_std: float = 1.0e-3
    # Opt in explicitly: existing checkpoints retain their original topology.
    pim_variant: str = "legacy"
    pim_prefix_init_std: float = 0.002
    # Reader identity and prefix topology are separate checkpoint contracts.
    prefix_layout: str = "legacy_merged"
    # The existing ICL-WAM CTE encodes completed interactions. It does not
    # provide Zeva-Ego's prediction of the current state's future effect.
    effect_source: str = "latest_completed"
    # The predictive CTE has a different policy contract. Keep old weights
    # explicit rather than silently changing their BIT-conditioned prior.
    prior_variant: str = "legacy_bit"
    initialization: str = "legacy_xavier"
    numerical_contract: str = "legacy_v1"

    def __post_init__(self) -> None:
        values = (
            self.memory_dim, self.context_dim, self.global_dim, self.phase_dim,
            self.effect_dim, self.effect_history_length, self.action_dim,
            self.action_horizon, self.prior_hidden_dim, self.prior_num_heads,
            self.action_hidden_dim,
        )
        if min(values) < 1:
            raise ValueError("Zeva policy-injection dimensions must be positive")
        if self.prior_hidden_dim % self.prior_num_heads:
            raise ValueError("prior_hidden_dim must be divisible by prior_num_heads")
        if not 0 <= self.leading_condition_steps <= self.action_horizon:
            raise ValueError("leading_condition_steps must be within action_horizon")
        if self.prior_loss_weight < 0 or self.prior_dropout_rate < 0 or self.prior_dropout_rate > 1:
            raise ValueError("invalid Zeva prior loss/dropout configuration")
        if self.min_std <= 0:
            raise ValueError("min_std must be positive")
        if self.pim_variant not in {"legacy", "cross_attempt"}:
            raise ValueError("pim_variant must be legacy or cross_attempt")
        if self.pim_prefix_init_std <= 0:
            raise ValueError("pim_prefix_init_std must be positive")
        if self.prefix_layout not in {"legacy_merged", "zeva_ego"}:
            raise ValueError("prefix_layout must be legacy_merged or zeva_ego")
        if self.effect_source not in {"latest_completed", "current_prediction"}:
            raise ValueError("effect_source must be latest_completed or current_prediction")
        if self.prior_variant not in {"legacy_bit", "zeva_ego"}:
            raise ValueError("prior_variant must be legacy_bit or zeva_ego")
        if self.initialization not in {"legacy_xavier", "zeva_ego"}:
            raise ValueError("adapter initialization must be legacy_xavier or zeva_ego")
        if self.initialization == "zeva_ego" and self.prior_variant != "zeva_ego":
            raise ValueError("zeva_ego initialization requires the predictive Zeva-Ego prior")
        if self.numerical_contract not in {"legacy_v1", "zeva_ego_v1"}:
            raise ValueError("numerical_contract must be legacy_v1 or zeva_ego_v1")
        if self.numerical_contract == "zeva_ego_v1" and self.prior_variant != "zeva_ego":
            raise ValueError("zeva_ego_v1 numerical_contract requires the predictive Zeva-Ego prior")
        if self.effect_source == "current_prediction":
            if self.prefix_layout != "zeva_ego" or self.prior_variant != "zeva_ego":
                raise ValueError("current_prediction requires zeva_ego prefix_layout and prior_variant")
        if self.prior_variant == "zeva_ego":
            if self.effect_source != "current_prediction":
                raise ValueError("zeva_ego prior requires the predictive CTE contract")
            if self.phase_dim != self.prior_hidden_dim or self.global_dim != self.prior_hidden_dim:
                raise ValueError("zeva_ego prior requires phase_dim=global_dim=prior_hidden_dim")


class FastWAMPolicyInjectionPrior(nn.Module):
    """FastWAM action prior using phase and completed BIT effect context."""

    def __init__(self, config: ExactZevaPolicyInjectionConfig) -> None:
        super().__init__()
        self.config = config
        self.global_to_anchors = nn.Linear(
            config.global_dim, 8 * config.prior_hidden_dim
        )
        self.anchor_position = nn.Parameter(
            torch.randn(1, 8, config.prior_hidden_dim) * 0.02
        )
        self.phase_query = nn.Sequential(
            nn.LayerNorm(config.phase_dim), nn.Linear(config.phase_dim, config.prior_hidden_dim)
        )
        self.effect_project = nn.Sequential(
            nn.LayerNorm(config.effect_dim), nn.Linear(config.effect_dim, config.prior_hidden_dim)
        )
        self.effect_position = nn.Parameter(
            torch.randn(1, config.effect_history_length, config.prior_hidden_dim) * 0.02
        )
        self.bos_effect = nn.Parameter(
            torch.randn(1, 1, config.effect_dim) * 0.02
        )
        self.effect_attention = nn.MultiheadAttention(
            config.prior_hidden_dim, config.prior_num_heads, batch_first=True
        )
        self.effect_norm = nn.LayerNorm(config.prior_hidden_dim)
        self.effect_gate = nn.Sequential(
            nn.LayerNorm(2 * config.prior_hidden_dim),
            nn.Linear(2 * config.prior_hidden_dim, config.prior_hidden_dim),
            nn.Sigmoid(),
        )
        self.progress_attention = nn.MultiheadAttention(
            config.prior_hidden_dim, config.prior_num_heads, batch_first=True
        )
        self.context_norm = nn.LayerNorm(config.prior_hidden_dim)
        self.distribution_head = nn.Sequential(
            nn.Linear(config.prior_hidden_dim, config.prior_hidden_dim),
            nn.GELU(),
            nn.Linear(config.prior_hidden_dim, 2 * config.action_horizon * config.action_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.anchor_position, std=0.02)
        nn.init.normal_(self.effect_position, std=0.02)
        nn.init.normal_(self.bos_effect, std=0.02)

    def forward(
        self,
        task_tokens: Tensor,
        phase: Tensor,
        bit_effects: Tensor,
        bit_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        batch = task_tokens.shape[0]
        expected = {
            "task_tokens": (batch, cfg.global_dim),
            "phase": (batch, cfg.phase_dim),
            "bit_effects": (batch, cfg.effect_history_length, cfg.effect_dim),
            "bit_mask": (batch, cfg.effect_history_length),
        }
        for name, shape in expected.items():
            if tuple(locals()[name].shape) != shape:
                raise ValueError(f"Expected {name} {shape}, got {tuple(locals()[name].shape)}")
        anchors = self.global_to_anchors(task_tokens).view(batch, 8, cfg.prior_hidden_dim)
        anchors = anchors + self.anchor_position
        phase_query = self.phase_query(phase)
        bit_mask = bit_mask.to(torch.bool)
        bit_input = torch.where(
            bit_mask.unsqueeze(-1),
            bit_effects,
            self.bos_effect.expand(batch, cfg.effect_history_length, -1),
        )
        bit = self.effect_project(bit_input) + self.effect_position
        effect_context, _ = self.effect_attention(
            phase_query[:, None], bit, bit, need_weights=False
        )
        effect_context = self.effect_norm(effect_context[:, 0])
        gate = self.effect_gate(torch.cat((phase_query, effect_context), dim=-1))
        local_query = gate * phase_query + (1.0 - gate) * effect_context
        context, _ = self.progress_attention(local_query[:, None], anchors, anchors, need_weights=False)
        params = self.distribution_head(self.context_norm(context[:, 0]))
        mean, raw_scale = params.chunk(2, dim=-1)
        mean = mean.view(batch, cfg.action_horizon, cfg.action_dim)
        std = (F.softplus(raw_scale) + cfg.min_std).view(batch, cfg.action_horizon, cfg.action_dim)
        return mean, std


class ZevaEgoActionPrior(ReferenceFloat32Module):
    """Reference EAP: global waypoints attended by the current raw CTE phase.

    Effects condition the independent visual prefix. They are deliberately
    not folded into the local query of this prior (unlike legacy ICL-WAM).
    Shapes retain FastWAM's joint14 / horizon32 action contract.
    """

    def __init__(self, config: ExactZevaPolicyInjectionConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.prior_hidden_dim
        self.temporal_waypoint = nn.Sequential(
            nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, dim * 8),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 8, dim) * 0.02)
        self.progress_attention = nn.MultiheadAttention(dim, config.prior_num_heads, batch_first=True)
        self.norm_global = nn.LayerNorm(dim)
        self.norm_local = nn.LayerNorm(dim)
        self.dist_head = nn.Sequential(
            nn.Linear(dim, 512), nn.GELU(),
            nn.Linear(512, config.action_horizon * config.action_dim * 2),
        )
        nn.init.normal_(self.dist_head[-1].weight, std=0.01)
        nn.init.zeros_(self.dist_head[-1].bias)

    @reference_inference_precision
    def forward(self, task_tokens: Tensor, phase: Tensor,
                bit_effects: Tensor | None = None, bit_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if task_tokens.ndim != 2 or task_tokens.shape[-1] != cfg.global_dim:
            raise ValueError("task_tokens must be [B,global_dim]")
        if phase.shape != (task_tokens.shape[0], cfg.phase_dim):
            raise ValueError("phase must be [B,phase_dim]")
        anchors = self.temporal_waypoint(task_tokens).reshape(-1, 8, cfg.prior_hidden_dim) + self.pos_emb
        keys = self.norm_global(anchors)
        local, _ = self.progress_attention(
            self.norm_local(phase)[:, None], keys, keys,
            need_weights=(cfg.numerical_contract == "zeva_ego_v1"),
        )
        params = self.dist_head(local[:, 0]).reshape(-1, cfg.action_horizon, cfg.action_dim, 2)
        return params[..., 0], params[..., 1].clamp(-5.0, 2.0).exp()


def gaussian_prior_nll(target: Tensor, mean: Tensor, std: Tensor, valid: Tensor | None = None,
                       *, reference_reduction: bool = False) -> Tensor:
    """Diagonal-Gaussian objective, with versioned action-dimension scaling.

    Zeva-Ego sums log probabilities across action dimensions, then averages
    time/batch. Preserve the old mean reduction for legacy trained weights.
    """
    if target.shape != mean.shape or mean.shape != std.shape:
        raise ValueError("target, mean, and std must have the same shape")
    if reference_reduction:
        nll = -torch.distributions.Normal(mean, std).log_prob(target).sum(dim=-1)
    else:
        nll = 0.5 * (((target - mean) / std).square() + 2.0 * std.log()).mean(dim=-1)
    if valid is None:
        return nll.mean()
    if valid.shape != nll.shape:
        raise ValueError("valid must have shape [B,H]")
    valid = valid.to(device=nll.device, dtype=nll.dtype)
    return (nll * valid).sum() / valid.sum().clamp_min(1.0)


def select_latest_completed_effect(
    bit_effects: Tensor,
    bit_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Select the latest BIT effect without assuming right-aligned valid slots.

    BIT positions must be ordered from oldest to newest. Masked entries are
    cleared before selection so NaN padding cannot enter a learned projection.
    Empty rows return an exact zero and a false validity flag. This is an
    observed, completed interaction, not Zeva-Ego's predicted current effect.
    """
    if bit_effects.ndim != 3 or bit_mask.shape != bit_effects.shape[:2]:
        raise ValueError("bit_effects/bit_mask must be [B,L,D]/[B,L]")
    valid = bit_mask.to(device=bit_effects.device, dtype=torch.bool)
    has_effect = valid.any(dim=-1)
    if bit_effects.shape[1] == 0:
        return bit_effects.new_zeros((bit_effects.shape[0], bit_effects.shape[2])), has_effect
    clean = torch.where(valid[..., None], bit_effects, torch.zeros_like(bit_effects))
    positions = torch.arange(valid.shape[1], device=valid.device)[None].expand_as(valid)
    latest = positions.masked_fill(~valid, -1).max(dim=-1).values.clamp_min(0)
    selected = clean.gather(
        1, latest[:, None, None].expand(-1, 1, clean.shape[-1])
    )[:, 0]
    selected = torch.where(has_effect[:, None], selected, torch.zeros_like(selected))
    return selected, has_effect


class ExactZevaPolicyInjectionAdapter(ReferenceFloat32Module):
    """Separate behavior-prefix and action-prior injection branches.

    ``legacy_merged`` preserves ICL-WAM's original combined global/PIM slot.
    ``zeva_ego`` exposes independent global, effect, and PIM slots. The
    versioned effect_source distinguishes legacy completed effects from the
    predictive CTE's current-boundary effect; their checkpoints cannot mix.
    Both layouts project the action-prior mean into ActionDiT hidden space.
    """

    is_exact_zeva = True

    def __init__(self, config: ExactZevaPolicyInjectionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExactZevaPolicyInjectionConfig()
        cfg = self.config
        self.prior = (ZevaEgoActionPrior(cfg) if cfg.prior_variant == "zeva_ego"
                      else FastWAMPolicyInjectionPrior(cfg))
        self.action_prior_adapter = nn.Linear(cfg.action_dim, cfg.action_hidden_dim)
        self.behavior_global_projector = nn.Linear(cfg.global_dim, cfg.context_dim)
        self.prefix_project = nn.Linear(cfg.memory_dim, cfg.context_dim)
        self.pim_gate = nn.Parameter(torch.tensor(float(cfg.prompt_gate_init)))
        if cfg.pim_variant == "cross_attempt":
            self.pim_to_global = nn.Linear(
                cfg.memory_dim, cfg.global_dim, bias=(cfg.prior_variant == "zeva_ego"),
            )
            # Retain the scalar in the state dict for legacy diagnostics, but
            # the new path neither multiplies by it nor trains it.
            self.pim_gate.requires_grad_(False)
        if cfg.prefix_layout == "zeva_ego":
            self.behavior_effect_projector = nn.Linear(cfg.effect_dim, cfg.context_dim)
            # The independent PIM token has no legacy scalar gate, including
            # the legacy-reader Stage2A configuration used to train its parent.
            self.pim_gate.requires_grad_(False)
        nn.init.zeros_(self.action_prior_adapter.weight)
        nn.init.zeros_(self.action_prior_adapter.bias)
        if cfg.initialization == "legacy_xavier":
            nn.init.xavier_uniform_(self.behavior_global_projector.weight)
            nn.init.zeros_(self.behavior_global_projector.bias)
        nn.init.xavier_uniform_(self.prefix_project.weight)
        nn.init.zeros_(self.prefix_project.bias)
        if cfg.prefix_layout == "zeva_ego" and cfg.initialization == "legacy_xavier":
            nn.init.xavier_uniform_(self.behavior_effect_projector.weight)
            nn.init.zeros_(self.behavior_effect_projector.bias)
        if cfg.pim_variant == "cross_attempt":
            self.reset_pim_parameters()

    def policy_injection_parameters(self):
        yield from self.prior.parameters()
        yield from self.action_prior_adapter.parameters()
        yield from self.behavior_global_projector.parameters()
        if self.config.prefix_layout == "zeva_ego":
            yield from self.behavior_effect_projector.parameters()

    def pim_adapter_parameters(self):
        yield from self.prefix_project.parameters()
        if self.config.pim_variant == "cross_attempt":
            yield from self.pim_to_global.parameters()
        elif self.config.prefix_layout == "legacy_merged":
            yield self.pim_gate

    @torch.no_grad()
    def reset_pim_parameters(self) -> None:
        if self.config.pim_variant == "cross_attempt":
            nn.init.normal_(self.prefix_project.weight, std=self.config.pim_prefix_init_std)
            nn.init.zeros_(self.pim_to_global.weight)
            if self.pim_to_global.bias is not None:
                nn.init.zeros_(self.pim_to_global.bias)
        else:
            nn.init.xavier_uniform_(self.prefix_project.weight)
        nn.init.zeros_(self.prefix_project.bias)
        self.pim_gate.fill_(float(self.config.prompt_gate_init))

    @reference_inference_precision
    def condition_task_tokens(
        self,
        task_tokens: Tensor,
        causal_prompt: Tensor,
        pim_mask: Tensor,
        *,
        enable_pim: bool = True,
    ) -> Tensor:
        """Condition the original global token on completed-attempt memory.

        Callers must pass the original task token to both this prior path and
        the prefix builder; both branches apply the same conditioning once.
        Empty/disabled memory exactly preserves the parent global token.
        """
        if task_tokens.ndim != 2 or task_tokens.shape[-1] != self.config.global_dim:
            raise ValueError("task_tokens must be [B,global_dim]")
        if causal_prompt.shape != (task_tokens.shape[0], self.config.memory_dim):
            raise ValueError("causal_prompt must be [B,memory_dim]")
        if pim_mask.ndim != 2 or pim_mask.shape[0] != task_tokens.shape[0]:
            raise ValueError("pim_mask must be [B,N]")
        if self.config.pim_variant != "cross_attempt" or not enable_pim:
            return task_tokens
        parameter = self.pim_to_global.weight
        original = task_tokens.to(device=parameter.device, dtype=parameter.dtype)
        prompt = causal_prompt.to(parameter)
        valid = pim_mask.to(device=parameter.device, dtype=torch.bool).any(dim=-1, keepdim=True)
        if self.config.prefix_layout == "zeva_ego":
            prompt = torch.where(valid, prompt, torch.zeros_like(prompt))
        delta = self.pim_to_global(prompt)
        return original + torch.where(valid, delta, torch.zeros_like(delta))

    @reference_inference_precision
    def prior_for_prompt(
        self,
        task_tokens: Tensor,
        phase: Tensor,
        bit_effects: Tensor,
        bit_mask: Tensor,
        causal_prompt: Tensor,
        pim_mask: Tensor,
        *,
        enable_pim: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Shared train/deploy entry for the PIM-conditioned frozen prior.

        Freezing prior parameters must not introduce ``no_grad`` here: Stage2B
        gradients need to reach ``pim_to_global`` through the parent prior.
        """
        conditioned = self.condition_task_tokens(
            task_tokens, causal_prompt, pim_mask, enable_pim=enable_pim,
        )
        parameter = next(self.prior.parameters())
        return self.prior(
            conditioned.to(parameter),
            phase.to(parameter),
            bit_effects.to(parameter),
            bit_mask.to(device=parameter.device, dtype=torch.bool),
        )

    def causal_prompt_prefix(
        self,
        causal_prompt: Tensor,
        pim_mask: Tensor,
        *,
        enable_pim: bool = True,
    ) -> Tensor:
        if causal_prompt.ndim != 2:
            raise ValueError("causal_prompt must be [B,D]")
        if causal_prompt.shape[-1] != self.config.memory_dim:
            raise ValueError(
                f"causal_prompt last dim must be {self.config.memory_dim}"
            )
        if pim_mask.ndim != 2 or pim_mask.shape[0] != causal_prompt.shape[0]:
            raise ValueError("pim_mask must be [B,K]")
        prompt_input = causal_prompt.to(self.prefix_project.weight)
        if self.config.prefix_layout == "zeva_ego":
            valid_input = pim_mask.to(
                device=prompt_input.device, dtype=torch.bool
            ).any(dim=-1, keepdim=True)
            if not enable_pim:
                valid_input = torch.zeros_like(valid_input)
            prompt_input = torch.where(valid_input, prompt_input, torch.zeros_like(prompt_input))
        prompt = self.prefix_project(prompt_input)[:, None]
        if not enable_pim:
            return torch.zeros_like(prompt)
        has_pim = (
            pim_mask.any(dim=-1, keepdim=True)
            .to(device=prompt.device, dtype=prompt.dtype)
            .unsqueeze(-1)
        )
        if self.config.pim_variant == "cross_attempt" or self.config.prefix_layout == "zeva_ego":
            return torch.where(has_pim.bool(), prompt, torch.zeros_like(prompt))
        return torch.tanh(self.pim_gate).to(prompt.dtype) * prompt * has_pim

    def behavior_prefix_slot(
        self,
        task_tokens: Tensor,
        causal_prompt: Tensor,
        pim_mask: Tensor,
        *,
        enable_pim: bool = True,
    ) -> Tensor:
        """Build the original ICL-WAM combined slot for legacy checkpoints."""
        if self.config.prefix_layout != "legacy_merged":
            raise ValueError(
                "zeva_ego has independent prefix tokens; use behavior_prefix_tokens"
            )
        if task_tokens.ndim != 2 or task_tokens.shape[-1] != self.config.global_dim:
            raise ValueError(
                "task_tokens shape does not match the exact Zeva adapter: "
                f"expected [B,{self.config.global_dim}], got {tuple(task_tokens.shape)}"
            )
        if task_tokens.shape[0] != causal_prompt.shape[0]:
            raise ValueError("task_tokens and causal_prompt must have the same batch size")
        conditioned_task = self.condition_task_tokens(
            task_tokens, causal_prompt, pim_mask, enable_pim=enable_pim,
        )
        base_prefix = self.behavior_global_projector(
            conditioned_task.to(
                device=self.behavior_global_projector.weight.device,
                dtype=self.behavior_global_projector.weight.dtype,
            )
        )[:, None]
        pim_delta = self.causal_prompt_prefix(
            causal_prompt,
            pim_mask,
            enable_pim=enable_pim,
        )
        return base_prefix + pim_delta.to(
            device=base_prefix.device,
            dtype=base_prefix.dtype,
        )

    @reference_inference_precision
    def behavior_prefix_tokens(
        self,
        task_tokens: Tensor,
        causal_prompt: Tensor,
        pim_mask: Tensor,
        *,
        bit_effects: Tensor | None = None,
        bit_mask: Tensor | None = None,
        current_effect: Tensor | None = None,
        current_effect_mask: Tensor | None = None,
        enable_pim: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Build prefix embeddings and their attention-key validity mask.

        The Zeva-Ego layout is always ordered ``[global, effect, PIM]``. PIM
        is a separate token; only ``pim_to_global`` conditions the global
        token. Empty effect/PIM slots are zeroed and invalid, so callers must
        also exclude them from attention rather than attending to zero keys.
        """
        if self.config.prefix_layout == "legacy_merged":
            tokens = self.behavior_prefix_slot(
                task_tokens, causal_prompt, pim_mask, enable_pim=enable_pim,
            )
            return tokens, torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)

        cfg = self.config
        if task_tokens.ndim != 2 or task_tokens.shape[-1] != cfg.global_dim:
            raise ValueError("task_tokens must be [B,global_dim]")
        batch = task_tokens.shape[0]
        if causal_prompt.shape != (batch, cfg.memory_dim):
            raise ValueError("causal_prompt must be [B,memory_dim]")
        if pim_mask.ndim != 2 or pim_mask.shape[0] != batch:
            raise ValueError("pim_mask must be [B,N]")
        if cfg.effect_source == "latest_completed":
            if current_effect is not None or current_effect_mask is not None:
                raise ValueError("latest_completed checkpoint cannot consume current_prediction inputs")
            if bit_effects is None or bit_mask is None:
                raise ValueError("zeva_ego prefix requires bit_effects and bit_mask for latest_completed effect")
            if bit_effects.shape != (batch, cfg.effect_history_length, cfg.effect_dim):
                raise ValueError("bit_effects must match [B,effect_history_length,effect_dim]")
            if bit_mask.shape != bit_effects.shape[:2]:
                raise ValueError("bit_mask must match bit_effects [B,effect_history_length]")
        else:
            if current_effect is None or current_effect_mask is None:
                raise ValueError("current_prediction requires current_effect and current_effect_mask")
            if current_effect.shape != (batch, cfg.effect_dim) or current_effect_mask.shape != (batch,):
                raise ValueError("current_effect/current_effect_mask must be [B,effect_dim]/[B]")

        parameter = self.prefix_project.weight
        has_pim = pim_mask.to(device=parameter.device, dtype=torch.bool).any(dim=-1)
        if not enable_pim:
            has_pim = torch.zeros_like(has_pim)
        clean_prompt = causal_prompt.to(parameter)
        clean_prompt = torch.where(
            has_pim[:, None], clean_prompt, torch.zeros_like(clean_prompt)
        )
        conditioned_task = self.condition_task_tokens(
            task_tokens, clean_prompt, pim_mask, enable_pim=enable_pim,
        )
        global_token = self.behavior_global_projector(
            conditioned_task.to(self.behavior_global_projector.weight)
        )
        if cfg.effect_source == "latest_completed":
            effect, has_effect = select_latest_completed_effect(
                bit_effects.to(self.behavior_effect_projector.weight), bit_mask,
            )
        else:
            effect = current_effect.to(self.behavior_effect_projector.weight)
            has_effect = current_effect_mask.to(device=effect.device, dtype=torch.bool)
            effect = torch.where(has_effect[:, None], effect, torch.zeros_like(effect))
            if not torch.isfinite(effect).all():
                raise ValueError("valid current_effect must be finite")
        effect_token = self.behavior_effect_projector(effect)
        effect_token = torch.where(
            has_effect[:, None], effect_token, torch.zeros_like(effect_token)
        )
        pim_token = self.prefix_project(clean_prompt)
        pim_token = torch.where(has_pim[:, None], pim_token, torch.zeros_like(pim_token))
        tokens = torch.stack(
            (global_token, effect_token.to(global_token), pim_token.to(global_token)), dim=1,
        )
        valid = torch.stack(
            (torch.ones_like(has_pim), has_effect.to(has_pim.device), has_pim), dim=1,
        )
        return tokens, valid.to(device=tokens.device)

    @reference_inference_precision
    def action_prior_residual(self, prior_mean: Tensor, *, training: bool) -> Tensor:
        if self.config.numerical_contract == "zeva_ego_v1":
            prior_mean = prior_mean.to(self.action_prior_adapter.weight)
        residual = self.action_prior_adapter(prior_mean)
        if training and self.config.prior_dropout_rate:
            draw = torch.rand((residual.shape[0], 1, 1), device=residual.device)
            keep = (draw < 1.0 - self.config.prior_dropout_rate
                    if self.config.numerical_contract == "zeva_ego_v1"
                    else draw >= self.config.prior_dropout_rate)
            residual = residual * keep.to(residual.dtype)
        elif not training:
            residual = residual * self.config.prior_inference_guidance_scale
        if self.config.leading_condition_steps:
            residual = torch.cat(
                (
                    residual.new_zeros((residual.shape[0], self.config.leading_condition_steps, residual.shape[-1])),
                    residual,
                ),
                dim=1,
            )
        return residual

    def add_prefix_to_context(self, context: Tensor, prefix: Tensor) -> Tensor:
        """Backward-compatible legacy helper.

        Production injection uses main-stream prefix tokens and leaves the
        instruction sequence untouched. This former text-residual helper
        remains available only for legacy callers.
        """
        if context.ndim != 3 or prefix.shape != (context.shape[0], 1, context.shape[-1]):
            raise ValueError("context/prefix shapes do not match")
        result = context.clone()
        result[:, :1] = result[:, :1] + prefix.to(device=context.device, dtype=context.dtype)
        return result

    def prepend_behavior_prefix_slot(
        self,
        context: Tensor,
        context_mask: Tensor,
        causal_prompt: Tensor,
        pim_mask: Tensor,
        task_tokens: Tensor,
        *,
        enable_pim: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Legacy compatibility helper for the former context-token path.

        This helper is intentionally NOT used by ``exact_zeva`` anymore.
        Production injection consumes :meth:`behavior_prefix_tokens` as main
        MoT prefix tokens and leaves instruction context unchanged.
        """
        if context.ndim != 3 or context_mask.shape != context.shape[:2]:
            raise ValueError("context/context_mask shapes do not match")
        if causal_prompt.ndim != 2 or causal_prompt.shape[0] != context.shape[0]:
            raise ValueError("causal_prompt must have the same batch size as context")
        prefix = self.behavior_prefix_slot(
            task_tokens,
            causal_prompt,
            pim_mask,
            enable_pim=enable_pim,
        )
        if prefix.shape != (context.shape[0], 1, context.shape[-1]):
            raise ValueError(
                "Causal Prompt prefix dimension must match the raw FastWAM context dimension"
            )
        # Empty PIM disables only the legacy residual; the task base prefix
        # remains a valid behavior condition.
        slot_mask = torch.ones(
            (context.shape[0], 1), device=context.device, dtype=torch.bool
        )
        return (
            torch.cat(
                (prefix.to(device=context.device, dtype=context.dtype), context),
                dim=1,
            ),
            torch.cat(
                (slot_mask, context_mask.to(device=context.device, dtype=torch.bool)),
                dim=1,
            ),
        )
