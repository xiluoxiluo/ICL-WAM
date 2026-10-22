"""Zeva-Ego causal encoder adapted to FastWAM's executed joint14 chunks.

Reference: air-embodied-brain/Zeva, feature/zeva_ego,
bae85ad753d3afe4f2e56762373d48ac0267e101, cte_eap.py/tri_stream.py/vision.py.
The old window/observed-effect encoder remains in causal_transition_encoder.py.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EGO_CTE_SCHEMA = "iclwam_zeva_ego_cte_v1"


@dataclass(frozen=True)
class ZevaEgoCTEConfig:
    action_dim: int = 14
    executed_horizon: int = 24
    hidden_dim: int = 256
    retrieval_dim: int = 128
    phase_dim: int = 256
    effect_dim: int = 256
    aux_phase_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_views: int = 3
    image_size: int = 224
    dropout: float = 0.1
    ema_decay: float = 0.99
    temporal_backend: str = "mamba"
    vision_backbone: str = "resnet18"
    vision_pretrained: bool = True

    def __post_init__(self) -> None:
        dimensions = (self.action_dim, self.executed_horizon, self.hidden_dim,
                      self.retrieval_dim, self.phase_dim, self.effect_dim,
                      self.aux_phase_dim, self.num_layers, self.num_heads,
                      self.num_views, self.image_size)
        if min(dimensions) < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("Positive dimensions and hidden_dim divisible by num_heads are required")
        if self.phase_dim != self.hidden_dim or self.effect_dim != self.hidden_dim:
            raise ValueError("Zeva-Ego phase/effect dimensions must equal the interaction hidden dimension")
        if not 0 <= self.dropout < 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("dropout and ema_decay must be in [0,1)")
        if self.temporal_backend not in {"mamba", "gru_test"}:
            raise ValueError("temporal_backend must be mamba or explicit non-promotable gru_test")
        if self.vision_backbone not in {"resnet18", "tiny_test"}:
            raise ValueError("vision_backbone must be resnet18 or explicit non-promotable tiny_test")
        if self.vision_backbone == "tiny_test" and self.vision_pretrained:
            raise ValueError("tiny_test has no pretrained weights; set vision_pretrained=false")

    @property
    def transition_steps(self) -> int:
        return self.executed_horizon

    @property
    def effect_window_transitions(self) -> int:
        return 1

    @property
    def image_channels(self) -> int:
        return 3

    @property
    def formal_architecture(self) -> bool:
        return self.temporal_backend == "mamba" and self.vision_backbone == "resnet18"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _MultiViewVision(nn.Module):
    def __init__(self, cfg: ZevaEgoCTEConfig) -> None:
        super().__init__()
        self.num_views = cfg.num_views
        if cfg.vision_backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            resnet = resnet18(weights=ResNet18_Weights.DEFAULT if cfg.vision_pretrained else None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            width = 512
        else:
            # Only explicit CPU contract/gradient tests use this architecture.
            width = 16
            self.backbone = nn.Sequential(
                nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width),
                nn.GELU(), nn.AdaptiveAvgPool2d(1),
            )
        self.proj = nn.Linear(width, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.view_fusion = nn.Sequential(
            nn.Linear(cfg.hidden_dim * cfg.num_views, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim), nn.SiLU(),
        ) if cfg.num_views > 1 else nn.Identity()
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[1:3] != (self.num_views, 3):
            raise ValueError("Vision inputs must be independently resized [N,V,3,H,W] RGB views")
        features = self.backbone(images.flatten(0, 1)).flatten(1)
        features = self.norm(self.proj(features)).reshape(images.shape[0], self.num_views, -1)
        return self.view_fusion(features.flatten(1))


@dataclass
class EgoCTEState:
    """Attempt-local streaming state; no images or future actions are retained."""

    backend: str
    batch_size: int
    offset: int = 0
    temporal: dict[int, Tensor] = field(default_factory=dict)
    mamba: Any = None


class _TemporalStream(nn.Module):
    def __init__(self, cfg: ZevaEgoCTEConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx, self.backend = layer_idx, cfg.temporal_backend
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        if self.backend == "mamba":
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError(
                    "Formal Zeva-Ego CTE requires mamba_ssm and its CUDA extensions; "
                    "no GRU fallback is performed. gru_test is for non-promotable CPU tests only."
                ) from exc
            self.mixer = Mamba(d_model=cfg.hidden_dim, d_state=16, d_conv=4,
                               expand=2, layer_idx=layer_idx)
        else:
            self.mixer = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)

    def forward(self, value: Tensor, state: EgoCTEState | None) -> Tensor:
        normalized = self.norm(value)
        if self.backend == "mamba":
            update = self.mixer(normalized, inference_params=None if state is None else state.mamba)
        else:
            update, hidden = self.mixer(normalized, None if state is None else state.temporal.get(self.layer_idx))
            if state is not None:
                state.temporal[self.layer_idx] = hidden
        return value + update


class _TriStreamBlock(nn.Module):
    """Reference's ordered, same-boundary cross-stream attention topology."""

    def __init__(self, cfg: ZevaEgoCTEConfig, index: int) -> None:
        super().__init__()
        self.temporal = nn.ModuleList(_TemporalStream(cfg, index * 3 + i) for i in range(3))
        self.attention_norm = nn.ModuleList(nn.LayerNorm(cfg.hidden_dim) for _ in range(3))
        self.vision_from_action = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.action_from_vision = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.interaction_from_transition = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True)

    def forward(self, vision: Tensor, action: Tensor, interaction: Tensor,
                state: EgoCTEState | None = None) -> tuple[Tensor, Tensor, Tensor]:
        vision, action, interaction = [
            stream(value, state) for stream, value in zip(self.temporal, (vision, action, interaction), strict=True)
        ]
        shape = vision.shape
        vision, action, interaction = (value.reshape(-1, 1, shape[-1]) for value in (vision, action, interaction))
        update, _ = self.vision_from_action(self.attention_norm[0](vision), action, action, need_weights=False)
        vision = vision + update
        update, _ = self.action_from_vision(self.attention_norm[1](action), vision, vision, need_weights=False)
        action = action + update
        transition = torch.cat((vision, action), dim=1)
        update, _ = self.interaction_from_transition(self.attention_norm[2](interaction), transition, transition, need_weights=False)
        interaction = interaction + update
        return tuple(value.reshape(shape) for value in (vision, action, interaction))


class ZevaEgoCTE(nn.Module):
    """Causal phase and predicted effect at every real replanning boundary.

    Inputs contain T+1 observations and T *executed* ordered action chunks.
    At boundary t, the model sees observation t and actions t-1, never actions t.
    Future visual deltas are detached EMA targets, not effect inputs.
    """

    schema = EGO_CTE_SCHEMA

    def __init__(self, cfg: ZevaEgoCTEConfig | None = None) -> None:
        super().__init__()
        self.cfg = self.config = cfg or ZevaEgoCTEConfig()
        cfg = self.cfg
        self.vision_encoder = _MultiViewVision(cfg)
        self.target_vision_encoder = deepcopy(self.vision_encoder).requires_grad_(False)
        self.action_proj = nn.Linear(cfg.action_dim * cfg.executed_horizon, cfg.hidden_dim)
        self.action_sos = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
        self.cte_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
        self.norm_input = nn.LayerNorm(cfg.hidden_dim)
        self.blocks = nn.ModuleList(_TriStreamBlock(cfg, i) for i in range(cfg.num_layers))
        self.norm_final = nn.LayerNorm(cfg.hidden_dim)
        self.action_predictor = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(),
                                              nn.Linear(cfg.hidden_dim, cfg.executed_horizon * cfg.action_dim))
        self.vision_predictor = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.task_head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, cfg.retrieval_dim))
        self.progress_head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, cfg.aux_phase_dim))
        self.effect_predictor = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.effect_dim))
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        # BN over flattened trajectory frames would otherwise mix future
        # observations into earlier representations during Stage1 training.
        for module in self.vision_encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        self.target_vision_encoder.eval()
        return self

    @torch.no_grad()
    def update_ema_target(self) -> None:
        for online, target in zip(self.vision_encoder.parameters(), self.target_vision_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - self.cfg.ema_decay)
        for online, target in zip(self.vision_encoder.buffers(), self.target_vision_encoder.buffers(), strict=True):
            target.copy_(online)

    update_ema = update_ema_target

    def _visual(self, images: Tensor, *, target: bool = False) -> Tensor:
        expected = (self.cfg.num_views, 3, self.cfg.image_size, self.cfg.image_size)
        if images.ndim != 6 or images.shape[2:] != expected:
            raise ValueError(f"CTE expects [B,T,{','.join(map(str, expected))}] independently resized RGB views")
        if not images.is_floating_point():
            raise ValueError("CTE RGB inputs must be floating point and normalized to [-1,1]")
        encoder = self.target_vision_encoder if target else self.vision_encoder
        return encoder(images.flatten(0, 1)).reshape(*images.shape[:2], self.cfg.hidden_dim)

    def _streams(self, visual: Tensor, shifted: Tensor, state: EgoCTEState | None = None):
        vision, action = self.norm_input(visual), self.norm_input(shifted)
        interaction = self.cte_token.expand_as(vision)
        for block in self.blocks:
            vision, action, interaction = block(vision, action, interaction, state)
        return vision, action, self.norm_final(interaction)

    def forward(self, images: Tensor, executed_actions: Tensor,
                valid: Tensor | None = None) -> dict[str, Tensor]:
        if executed_actions.ndim != 4 or executed_actions.shape[2:] != (self.cfg.executed_horizon, self.cfg.action_dim):
            raise ValueError("executed_actions must contain ordered [B,T,executed_horizon,action_dim] complete chunks")
        batch, length = executed_actions.shape[:2]
        if images.ndim != 6 or images.shape[2:] != (self.cfg.num_views, 3, self.cfg.image_size, self.cfg.image_size):
            raise ValueError("CTE images must be independently resized [B,T+1,V,3,image_size,image_size] RGB views")
        if images.shape[:2] != (batch, length + 1):
            raise ValueError("Exactly one more boundary observation than executed chunks is required")
        if valid is None:
            valid = torch.ones((batch, length), dtype=torch.bool, device=images.device)
        if valid.shape != (batch, length):
            raise ValueError("valid must align with executed action chunks [B,T]")
        valid = valid.to(device=images.device, dtype=torch.bool)
        if length > 1 and bool((~valid[:, :-1] & valid[:, 1:]).any()):
            raise ValueError("Episode padding must be a contiguous suffix, never holes in causal history")
        boundary_valid = torch.cat((torch.ones((batch, 1), dtype=torch.bool, device=images.device), valid), dim=1)
        clean_images = torch.where(boundary_valid[..., None, None, None, None], images, torch.zeros_like(images))
        clean_actions = torch.where(valid[..., None, None], executed_actions, torch.zeros_like(executed_actions))
        visual = self._visual(clean_images)
        with torch.no_grad():
            target = self._visual(clean_images, target=True)
        embedded = self.action_proj(clean_actions.flatten(2))
        shifted = torch.cat((self.action_sos.expand(batch, 1, -1), embedded), dim=1)
        vision, action, phase = self._streams(visual, shifted)
        predicted_effect = self.effect_predictor(phase)
        # Expose the terminal boundary too; no target is manufactured for it.
        phase = torch.where(boundary_valid[..., None], phase, torch.zeros_like(phase))
        predicted_effect = torch.where(boundary_valid[..., None], predicted_effect, torch.zeros_like(predicted_effect))
        task = self.task_head(phase)
        task = torch.where(boundary_valid[..., None], task, torch.zeros_like(task))
        progress = F.normalize(self.progress_head(phase), dim=-1)
        progress = torch.where(boundary_valid[..., None], progress, torch.zeros_like(progress))
        return {
            "phase": phase, "causal_interaction_state": phase, "global": phase,
            "predicted_effect": predicted_effect,
            # Export pools raw task-head outputs, then L2 normalizes the
            # trajectory mean. Per-boundary normalization changes the bank.
            "retrieval": task,
            "z_global": task[:, :-1], "z_local": progress[:, :-1],
            "pred_act": self.action_predictor(action[:, :-1]).reshape_as(executed_actions),
            "pred_vis": self.vision_predictor(vision[:, :-1]),
            "pred_effect": predicted_effect[:, :-1],
            "target_vis": target[:, 1:],
            "target_effect": (target[:, 1:] - target[:, :-1]).detach(),
            "logit_scale": self.logit_scale, "valid": valid,
            "boundary_valid": boundary_valid,
        }

    @torch.no_grad()
    def step(self, images: Tensor, previous_executed_actions: Tensor | None = None,
             cache: EgoCTEState | None = None) -> tuple[Tensor, Tensor, EgoCTEState]:
        if self.training:
            raise RuntimeError("Online CTE must be in eval mode")
        batch = images.shape[0]
        if cache is None:
            if previous_executed_actions is not None:
                raise ValueError("The initial boundary requires SOS, not an unobserved previous chunk")
            cache = EgoCTEState(self.cfg.temporal_backend, batch)
            if cache.backend == "mamba":
                from mamba_ssm.utils.generation import InferenceParams

                cache.mamba = InferenceParams(max_seqlen=1_000_000, max_batch_size=batch)
            action = self.action_sos.expand(batch, 1, -1)
        else:
            if cache.backend != self.cfg.temporal_backend or cache.batch_size != batch:
                raise ValueError("CTE streaming state belongs to a different backend or batch")
            expected = (batch, self.cfg.executed_horizon, self.cfg.action_dim)
            if previous_executed_actions is None or previous_executed_actions.shape != expected:
                raise ValueError("Subsequent boundaries require the actual previous complete executed chunk")
            action = self.action_proj(previous_executed_actions.flatten(1)).unsqueeze(1)
        _, _, phase = self._streams(self._visual(images.unsqueeze(1)), action, cache)
        cache.offset += 1
        if cache.mamba is not None:
            cache.mamba.seqlen_offset += 1
        return phase[:, 0], self.effect_predictor(phase[:, 0]), cache


def zeva_ego_cte_loss(outputs: dict[str, Tensor], executed_actions: Tensor,
                      valid: Tensor, task_ids: Tensor, *, effect_weight: float = 0.2) -> dict[str, Tensor]:
    """Reference's five Stage1 objectives, with FastWAM action coordinates."""
    valid = valid.to(device=executed_actions.device, dtype=torch.bool)
    if valid.shape != executed_actions.shape[:2] or not bool(valid.any()):
        raise ValueError("CTE loss needs aligned, nonempty valid executed chunks")
    if task_ids.shape != (executed_actions.shape[0],):
        raise ValueError("task_ids must contain one semantic task per episode")
    if effect_weight < 0:
        raise ValueError("effect_weight must be non-negative")
    action = (outputs["pred_act"][valid] - executed_actions[valid]).square().sum(-1).mean(-1).mean()
    vision = (outputs["pred_vis"][valid] - outputs["target_vis"][valid]).square().sum(-1).mean()
    effect = (outputs["pred_effect"][valid] - outputs["target_effect"][valid]).square().sum(-1).mean()
    task_features = torch.where(valid[..., None], outputs["z_global"], torch.zeros_like(outputs["z_global"]))
    pooled = F.normalize(task_features.sum(1) / valid.sum(1, keepdim=True).clamp_min(1), dim=-1)
    episode_valid = valid.any(1)
    pooled, labels = pooled[episode_valid], task_ids.to(valid.device)[episode_valid]
    scale = outputs["logit_scale"].exp()
    logits = pooled @ pooled.T * scale
    positives = labels[:, None].eq(labels[None, :])
    positives.fill_diagonal_(False)
    # Zeva-Ego intentionally retains the self logit in the denominator.
    log_prob = logits - logits.logsumexp(-1, keepdim=True)
    global_loss = -(log_prob * positives).sum() / positives.sum().clamp_min(1)
    local = outputs["z_local"][valid]
    phase_loss = F.cross_entropy(local @ local.T * scale, torch.arange(len(local), device=local.device))
    total = 0.1 * action + 0.2 * vision + 2.0 * global_loss + phase_loss + effect_weight * effect
    return {"total": total, "action": action, "vision": vision,
            "global": global_loss, "phase": phase_loss, "effect": effect}
