"""Causal Transition Encoder adapted from Zeva for RoboTwin/FastWAM."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class CausalTransitionEncoderConfig:
    action_dim: int = 14
    hidden_dim: int = 256
    retrieval_dim: int = 128
    phase_dim: int = 128
    effect_dim: int = 128
    transition_steps: int = 4
    effect_window_transitions: int = 4
    effect_target_grid: tuple[int, int] = (4, 6)
    num_layers: int = 4
    num_heads: int = 8
    image_channels: int = 3
    ema_decay: float = 0.996
    use_mamba: bool = False
    vision_chunk_size: int = 256
    vision_gradient_checkpointing: bool = True
    vision_checkpoint_threshold: int = 1024

    def __post_init__(self) -> None:
        if min(self.action_dim, self.hidden_dim, self.retrieval_dim, self.phase_dim, self.effect_dim,
               self.transition_steps, self.effect_window_transitions, self.num_layers, self.num_heads) < 1:
            raise ValueError("CTE dimensions and layer counts must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.phase_dim > self.hidden_dim:
            raise ValueError("phase_dim cannot exceed hidden_dim for visual_key projection")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _VisionStem(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, chunk_size: int, gradient_checkpointing: bool, checkpoint_threshold: int) -> None:
        super().__init__()
        width = max(hidden_dim // 4, 32)
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=7, stride=4, padding=3), nn.GELU(),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width * 2, hidden_dim, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.chunk_size = chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_threshold = checkpoint_threshold

    def _encode_flat(self, flat: Tensor) -> Tensor:
        return self.proj(self.net(flat).flatten(1))

    def forward(self, frames: Tensor) -> Tensor:
        batch, steps = frames.shape[:2]
        flat = frames.flatten(0, 1)
        use_checkpoint = self.training and self.gradient_checkpointing and torch.is_grad_enabled() and flat.shape[0] > self.checkpoint_threshold
        if not use_checkpoint:
            return self._encode_flat(flat).unflatten(0, (batch, steps))
        chunks = [checkpoint(self._encode_flat, chunk, use_reentrant=False) for chunk in flat.split(self.chunk_size, dim=0)]
        return torch.cat(chunks, dim=0).unflatten(0, (batch, steps))


class _FrozenVisualDeltaTarget(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, grid: tuple[int, int]) -> None:
        super().__init__()
        input_dim = channels * grid[0] * grid[1]
        generator = torch.Generator().manual_seed(20260815)
        projection = torch.randn(hidden_dim, input_dim, generator=generator) / input_dim**0.5
        self.register_buffer("projection", projection)
        self.grid = grid

    @torch.no_grad()
    def forward(self, frames: Tensor) -> Tensor:
        batch, steps = frames.shape[:2]
        flat = frames.flatten(0, 1).float()
        pooled = F.adaptive_avg_pool2d(flat, self.grid).flatten(1)
        pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        return F.linear(pooled, self.projection).unflatten(0, (batch, steps))


class _CausalMixer(nn.Module):
    def __init__(self, hidden_dim: int, use_mamba: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.kind = "gru"
        if use_mamba:
            try:
                from mamba_ssm import Mamba  # type: ignore[import-not-found]
                self.mixer = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
                self.kind = "mamba"
            except ImportError:
                self.mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: Tensor, initial_state: Tensor | None = None) -> Tensor:
        y = self.norm(x)
        if self.kind == "mamba":
            if initial_state is not None:
                # Mamba has no stable hidden-state API across supported
                # versions; preserve the handoff as a causal token offset.
                y = y + initial_state[:, None, :]
            y = self.mixer(y)
        else:
            hidden = None if initial_state is None else initial_state.unsqueeze(0)
            y, _ = self.mixer(y, hidden)
        return x + y


class _InteractionBlock(nn.Module):
    def __init__(self, cfg: CausalTransitionEncoderConfig) -> None:
        super().__init__()
        self.visual = _CausalMixer(cfg.hidden_dim, cfg.use_mamba)
        self.action = _CausalMixer(cfg.hidden_dim, cfg.use_mamba)
        self.interaction = _CausalMixer(cfg.hidden_dim, cfg.use_mamba)
        self.cross = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 4 * cfg.hidden_dim), nn.GELU(), nn.Linear(4 * cfg.hidden_dim, cfg.hidden_dim))

    def forward(
        self,
        visual: Tensor,
        action: Tensor,
        interaction: Tensor,
        valid: Tensor,
        interaction_initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        visual, action = self.visual(visual), self.action(action)
        interaction = self.interaction(interaction, interaction_initial_state)
        tokens = torch.stack((visual, action, interaction), dim=2)
        batch, steps, streams, dim = tokens.shape
        flat = tokens.reshape(batch * steps, streams, dim)
        attended, _ = self.cross(flat, flat, flat, need_weights=False)
        flat = flat + attended + self.ffn(flat)
        visual, action, interaction = flat.reshape(batch, steps, streams, dim).unbind(dim=2)
        mask = valid.unsqueeze(-1).to(visual.dtype)
        return visual * mask, action * mask, interaction * mask


class CausalTransitionEncoder(nn.Module):
    """Causal phase/effect encoder; action stream is right-shifted for leakage protection."""

    def __init__(self, cfg: CausalTransitionEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CausalTransitionEncoderConfig()
        cfg = self.cfg
        self.visual_encoder = _VisionStem(cfg.image_channels, cfg.hidden_dim, cfg.vision_chunk_size, cfg.vision_gradient_checkpointing, cfg.vision_checkpoint_threshold)
        self.target_visual_encoder = deepcopy(self.visual_encoder).requires_grad_(False)
        transition_dim = cfg.transition_steps * cfg.action_dim
        self.action_encoder = nn.Sequential(nn.LayerNorm(transition_dim), nn.Linear(transition_dim, cfg.hidden_dim))
        self.bos_action = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.interaction_state_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        nn.init.normal_(self.bos_action, std=0.02)
        nn.init.normal_(self.interaction_state_token, std=0.02)
        self.blocks = nn.ModuleList([_InteractionBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.retrieval_head = nn.Linear(cfg.hidden_dim, cfg.retrieval_dim)
        self.phase_head = nn.Linear(cfg.hidden_dim, cfg.phase_dim)
        self.action_head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, transition_dim))
        self.visual_head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        effect_window_dim = cfg.effect_window_transitions * transition_dim
        self.effect_action_encoder = nn.Sequential(nn.LayerNorm(effect_window_dim), nn.Linear(effect_window_dim, cfg.hidden_dim))
        self.frozen_effect_target = _FrozenVisualDeltaTarget(cfg.image_channels, cfg.hidden_dim, cfg.effect_target_grid)
        self.effect_pre_head = nn.Sequential(nn.LayerNorm(2 * cfg.hidden_dim), nn.Linear(2 * cfg.hidden_dim, cfg.effect_dim))
        self.effect_post_head = nn.Sequential(nn.LayerNorm(2 * cfg.hidden_dim), nn.Linear(2 * cfg.hidden_dim, cfg.effect_dim))
        self.effect_action_head = nn.Sequential(nn.LayerNorm(cfg.effect_dim), nn.Linear(cfg.effect_dim, 4 * cfg.action_dim))
        self.effect_outcome_head = nn.Sequential(nn.LayerNorm(cfg.effect_dim), nn.Linear(cfg.effect_dim, cfg.hidden_dim))
        # The predictor is causal and is the only effect branch that can be
        # used before executing an action. The observed branch is computed
        # after a transition and is what gets written to BIT/PIM.
        self.transition_effect_predictor = nn.Sequential(
            nn.LayerNorm(2 * cfg.hidden_dim),
            nn.Linear(2 * cfg.hidden_dim, cfg.effect_dim),
        )
        # The longer effect-window heads above remain available for the
        # auxiliary loss.
        self.transition_effect_head = nn.Sequential(
            nn.LayerNorm(3 * cfg.hidden_dim),
            nn.Linear(3 * cfg.hidden_dim, cfg.effect_dim),
        )
        self.transition_effect_outcome = nn.Sequential(
            nn.LayerNorm(cfg.effect_dim),
            nn.Linear(cfg.effect_dim, cfg.hidden_dim),
        )

    @torch.no_grad()
    def update_ema_target(self) -> None:
        for target, online in zip(self.target_visual_encoder.parameters(), self.visual_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - self.cfg.ema_decay)

    @torch.no_grad()
    def encode_target_vision(self, frames: Tensor) -> Tensor:
        return self.target_visual_encoder(frames)

    def forward(
        self,
        frames: Tensor,
        transition_actions: Tensor,
        valid_mask: Tensor | None = None,
        transition_valid: Tensor | None = None,
        initial_state: Tensor | None = None,
        initial_state_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.cfg
        if frames.ndim != 5 or transition_actions.ndim != 4:
            raise ValueError("Expected frames [B,T,3,H,W] and transition_actions [B,T-1,4,A].")
        if transition_actions.shape[0] != frames.shape[0] or transition_actions.shape[1] != frames.shape[1] - 1 or transition_actions.shape[2:] != (cfg.transition_steps, cfg.action_dim):
            raise ValueError("Frame/transition time axes or action shape do not match CTE config.")
        if valid_mask is None:
            valid_mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=frames.device)
        if transition_valid is None:
            transition_valid = torch.ones(transition_actions.shape[:2], dtype=torch.bool, device=frames.device)
        if valid_mask.shape != frames.shape[:2] or transition_valid.shape != transition_actions.shape[:2]:
            raise ValueError("valid masks have incompatible shapes")
        valid_mask = valid_mask.to(device=frames.device, dtype=torch.bool)
        transition_valid = transition_valid.to(device=frames.device, dtype=torch.bool)
        if initial_state is not None and initial_state.shape != (frames.shape[0], cfg.hidden_dim):
            raise ValueError(f"initial_state must be [B,{cfg.hidden_dim}]")
        if initial_state_mask is not None:
            if initial_state is None or initial_state_mask.shape != (frames.shape[0],):
                raise ValueError("initial_state_mask requires initial_state and must be [B]")
            initial_state_mask = initial_state_mask.to(device=frames.device, dtype=torch.bool)
        # An invalid/padded transition breaks the causal segment.  Do not let
        # later frames update the recurrent state and then accidentally serve
        # as evidence for a future valid transition in the same window.
        # A frame-validity gap is a causal boundary just like an invalid
        # action group.  Cumulative masking prevents a later frame from
        # re-entering the same recurrent segment after padding/corruption.
        causal_valid = torch.cumprod(
            valid_mask.bool().to(dtype=torch.int64), dim=1
        ).bool()
        if transition_valid.shape[1] > 0:
            causal_valid[:, 1:] &= torch.cumprod(
                transition_valid.bool().to(dtype=torch.int64), dim=1
            ).bool()
        visual = self.visual_encoder(frames)
        transition_embed = self.action_encoder(transition_actions.flatten(-2))
        action = self.bos_action.expand(frames.shape[0], frames.shape[1], -1).clone()
        action[:, 1:] = transition_embed
        action[:, 1:] *= transition_valid.unsqueeze(-1).to(action.dtype)
        interaction = self.interaction_state_token.expand_as(visual)
        if initial_state is not None:
            interaction = interaction.clone()
            initial = initial_state.to(device=interaction.device, dtype=interaction.dtype)
            if initial_state_mask is not None:
                initial = torch.where(initial_state_mask[:, None], initial, interaction[:, 0])
                handoff_mask = initial_state_mask.to(
                    device=interaction.device, dtype=interaction.dtype
                )
            else:
                handoff_mask = torch.ones(
                    (frames.shape[0],), device=interaction.device, dtype=interaction.dtype
                )
            interaction[:, 0] = initial
            # The incoming state is the recurrent handoff from the preceding
            # chunk.  It is supplied as the GRU hidden state below; retaining
            # the boundary token itself keeps the state/phase heads aligned.
            recurrent_handoff = initial * handoff_mask[:, None]
        else:
            recurrent_handoff = None
        for block in self.blocks:
            visual, action, interaction = block(
                visual,
                action,
                interaction,
                causal_valid,
                interaction_initial_state=recurrent_handoff,
            )
        z = self.final_norm(interaction) * causal_valid.unsqueeze(-1).to(interaction.dtype)
        with torch.no_grad():
            target_visual = self.target_visual_encoder(frames)
        transition_complete = causal_valid[:, :-1] & causal_valid[:, 1:] & transition_valid
        # Keep the observed effect target independent of the trainable/EMA
        # visual stem.  This is the same frozen RGB projection used by the
        # longer effect-window objective and makes cache features stable after
        # Stage 1 training.
        frozen_effect_visual = self.frozen_effect_target(frames)
        transition_delta = frozen_effect_visual[:, 1:] - frozen_effect_visual[:, :-1]
        transition_effect_prediction_raw = self.transition_effect_predictor(
            torch.cat((z[:, :-1], transition_embed), dim=-1)
        )
        transition_effect_raw = self.transition_effect_head(
            torch.cat((z[:, :-1], transition_embed, transition_delta), dim=-1)
        )
        transition_effect = F.normalize(transition_effect_raw, dim=-1)
        transition_effect_prediction = F.normalize(transition_effect_prediction_raw, dim=-1)
        windows = (frames.shape[1] - 1) // cfg.effect_window_transitions
        if windows:
            starts = torch.arange(windows, device=frames.device) * cfg.effect_window_transitions
            ends = starts + cfg.effect_window_transitions
            effect_actions = transition_actions[:, : windows * cfg.effect_window_transitions].reshape(frames.shape[0], windows, cfg.effect_window_transitions, cfg.transition_steps, cfg.action_dim)
            effect_valid = transition_valid[:, : windows * cfg.effect_window_transitions].reshape(frames.shape[0], windows, cfg.effect_window_transitions)
            effect_complete = causal_valid[:, starts] & causal_valid[:, ends] & effect_valid.all(dim=-1)
            effect_embed = self.effect_action_encoder(effect_actions.flatten(-3))
            frozen_visual = frozen_effect_visual
            effect_delta = frozen_visual[:, ends] - frozen_visual[:, starts]
            effect_pre_raw = self.effect_pre_head(torch.cat((z[:, starts], effect_embed), dim=-1))
            effect_post_raw = self.effect_post_head(torch.cat((frozen_visual[:, starts], effect_delta), dim=-1))
        else:
            shape = (frames.shape[0], 0)
            effect_actions = transition_actions.new_zeros((frames.shape[0], 0, cfg.effect_window_transitions, cfg.transition_steps, cfg.action_dim))
            effect_complete = torch.zeros(shape, dtype=torch.bool, device=frames.device)
            effect_delta = z[:, :0]
            effect_pre_raw = z[:, :0]
            effect_post_raw = z[:, :0]
        if effect_pre_raw.shape[1] == 0:
            effect_pre = effect_post = effect_pre_raw.new_zeros((frames.shape[0], 0, cfg.effect_dim))
            effect_outcome_pre = effect_outcome_post = effect_pre_raw.new_zeros((frames.shape[0], 0, cfg.hidden_dim))
            effect_action = effect_pre_raw.new_zeros((frames.shape[0], 0, 4 * cfg.action_dim))
        else:
            effect_pre, effect_post = F.normalize(effect_pre_raw, dim=-1), F.normalize(effect_post_raw, dim=-1)
            effect_outcome_pre, effect_outcome_post = self.effect_outcome_head(effect_pre_raw), self.effect_outcome_head(effect_post_raw)
            effect_action = self.effect_action_head(effect_pre_raw)
        return {
            "causal_interaction_state": z,
            "retrieval": F.normalize(self.retrieval_head(z), dim=-1),
            "phase": F.normalize(self.phase_head(z), dim=-1),
            "next_action": self.action_head(z[:, :-1]).view(frames.shape[0], frames.shape[1] - 1, cfg.transition_steps, cfg.action_dim),
            "next_vision": self.visual_head(z),
            "visual_key": F.normalize(target_visual[..., : cfg.phase_dim], dim=-1),
            "effect_pre": effect_pre,
            "effect_post": effect_post,
            "effect_pre_raw": effect_pre_raw,
            "effect_post_raw": effect_post_raw,
            "effect_outcome_pre": effect_outcome_pre,
            "effect_outcome_post": effect_outcome_post,
            "effect_action": effect_action,
            "effect_actions": effect_actions,
            "effect_delta_target": effect_delta,
            "effect_complete": effect_complete,
            "target_visual": target_visual,
            "state_valid": causal_valid,
            "transition_complete": transition_complete,
            "transition_effect": transition_effect,
            "transition_effect_raw": transition_effect_raw,
            "transition_effect_prediction": transition_effect_prediction,
            "transition_effect_prediction_raw": transition_effect_prediction_raw,
            "transition_effect_outcome": self.transition_effect_outcome(transition_effect_prediction_raw),
            "transition_effect_observed_outcome": self.transition_effect_outcome(transition_effect_raw),
            "transition_delta_target": transition_delta,
        }

    @torch.no_grad()
    def initialize(self, task: str, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Initialize online state from one observation; no action/future frame is consumed."""
        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        if observation.ndim != 4:
            raise ValueError("observation must be [B,3,H,W] or [3,H,W]")
        frames = observation.unsqueeze(1)
        actions = torch.zeros((frames.shape[0], 0, self.cfg.transition_steps, self.cfg.action_dim), device=frames.device, dtype=frames.dtype)
        # The recurrent state is represented by the causal interaction token after one frame.
        outputs = self(frames, actions, transition_valid=torch.zeros((frames.shape[0], 0), dtype=torch.bool, device=frames.device))
        return outputs["causal_interaction_state"][:, 0], outputs["phase"][:, 0]

    @torch.no_grad()
    def update(self, state: Tensor, executed_action: Tensor, observation: Tensor, next_observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Update after a completed transition and return next state/phase/effect."""
        if executed_action.ndim == 2:
            executed_action = executed_action.unsqueeze(0)
        if executed_action.shape[1:] != (self.cfg.transition_steps, self.cfg.action_dim):
            raise ValueError(f"executed_action must be [B,{self.cfg.transition_steps},{self.cfg.action_dim}]")
        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        if next_observation.ndim == 3:
            next_observation = next_observation.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape != (observation.shape[0], self.cfg.hidden_dim):
            raise ValueError(f"state must be [B,{self.cfg.hidden_dim}]")
        frames = torch.stack((observation, next_observation), dim=1)
        visual = self.visual_encoder(frames)
        transition_embed = self.action_encoder(executed_action.flatten(-2)).unsqueeze(1)
        action = self.bos_action.expand(frames.shape[0], 2, -1).clone()
        action[:, 1:] = transition_embed
        # Match ``forward(..., initial_state=state)`` exactly: the incoming
        # recurrent state initializes each interaction GRU, while the
        # after-frame token starts from the learned interaction BOS token.
        interaction = self.interaction_state_token.expand(frames.shape[0], 2, -1).clone()
        interaction[:, 0] = state
        valid = torch.ones((frames.shape[0], 2), dtype=torch.bool, device=frames.device)
        for block in self.blocks:
            visual, action, interaction = block(
                visual,
                action,
                interaction,
                valid,
                interaction_initial_state=state.to(
                    device=interaction.device, dtype=interaction.dtype
                ),
            )
        z = self.final_norm(interaction)
        with torch.no_grad():
            target_visual = self.target_visual_encoder(frames)
        frozen_effect_visual = self.frozen_effect_target(frames)
        delta = frozen_effect_visual[:, 1] - frozen_effect_visual[:, 0]
        effect_raw = self.transition_effect_head(torch.cat((z[:, 0], transition_embed[:, 0], delta), dim=-1))
        effect = F.normalize(effect_raw, dim=-1)
        return z[:, 1], F.normalize(self.phase_head(z[:, 1]), dim=-1), effect
