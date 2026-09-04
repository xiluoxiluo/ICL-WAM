"""Mask-aware Zeva CTE objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class CTELossConfig:
    # Match the official Zeva CTE objective defaults.  These are CTE heads,
    # not FastWAM Stage-2 video/joint losses; Stage 2 still optimizes only the
    # frozen-backbone action flow-matching path.
    action_weight: float = 1.0
    vision_weight: float = 1.0
    task_weight: float = 0.2
    phase_weight: float = 0.1
    effect_weight: float = 0.25
    effect_contrastive_weight: float = 1.0
    effect_action_weight: float = 0.05
    effect_align_weight: float = 0.1
    effect_variance_weight: float = 1.0
    effect_covariance_weight: float = 0.04
    transition_effect_weight: float = 0.25
    effect_temperature: float = 0.07
    temperature: float = 0.1


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weight = mask.unsqueeze(-1).to(x.dtype)
    return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _task_identity_clustering_loss(retrieval: Tensor, valid: Tensor, semantic_ids: Tensor, temperature: float) -> Tensor:
    sample_valid = valid.bool().any(dim=1)
    if int(sample_valid.sum()) < 2:
        return retrieval.new_zeros(())
    z = F.normalize(_masked_mean(retrieval[sample_valid], valid[sample_valid]), dim=-1)
    semantic_ids = semantic_ids[sample_valid]
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    positives = semantic_ids[:, None].eq(semantic_ids[None, :])
    positives.fill_diagonal_(False)
    usable = positives.any(dim=1)
    if not bool(usable.any()):
        return z.new_zeros(())
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1).clamp_min(1))[usable].mean()


def _phase_progression_loss(phase: Tensor, transition_valid: Tensor, margin: float = 0.01) -> Tensor:
    """Penalize non-progressing phase only inside contiguous valid runs.

    A masked transition must not be removed and then bridged by indexing, as
    that would create a synthetic phase delta across an unknown/padded gap.
    """
    losses = []
    for sequence, transitions in zip(phase, transition_valid.bool(), strict=True):
        start = None
        for index, is_valid in enumerate(torch.cat((transitions, transitions.new_zeros(1)))):
            if bool(is_valid) and start is None:
                start = index
            elif not bool(is_valid) and start is not None:
                end = index
                # A run of at least two transitions gives a stable direction.
                if end - start >= 2:
                    z = sequence[start : end + 1]
                    direction = F.normalize(z[-1] - z[0], dim=-1)
                    losses.append(F.relu(margin - (z[1:] - z[:-1]) @ direction).mean())
                start = None
    return torch.stack(losses).mean() if losses else phase.new_zeros(())


def summarize_effect_window(actions: Tensor) -> Tensor:
    if actions.ndim != 5:
        raise ValueError("Effect actions must be [B,W,K,4,A].")
    sequence = actions.flatten(-3, -2)
    return torch.cat((sequence[..., 0, :], sequence[..., -1, :] - sequence[..., 0, :], sequence.mean(dim=-2), sequence.std(dim=-2, correction=0)), dim=-1)


def _effect_nce(query: Tensor, target: Tensor, mask: Tensor, temperature: float) -> Tensor:
    query, target = query[mask], target[mask]
    if query.shape[0] < 2:
        return query.new_zeros(())
    logits = F.normalize(query, dim=-1) @ F.normalize(target, dim=-1).T / temperature
    return F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))


def _vicreg(code: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    code = code[mask]
    if code.shape[0] < 2:
        return code.new_zeros(()), code.new_zeros(())
    centered = code - code.mean(dim=0, keepdim=True)
    std = (centered.var(dim=0, correction=0) + 1e-4).sqrt()
    variance = F.relu(1.0 - std).mean()
    covariance = centered.T @ centered / max(code.shape[0] - 1, 1)
    covariance.fill_diagonal_(0)
    return variance, covariance.square().sum() / code.shape[-1]


def causal_transition_encoder_loss(outputs: dict[str, Tensor], transition_actions: Tensor, valid_mask: Tensor, semantic_ids: Tensor, cfg: CTELossConfig | None = None) -> dict[str, Tensor]:
    cfg = cfg or CTELossConfig()
    complete = outputs["transition_complete"]
    if complete.shape != transition_actions.shape[:2]:
        raise ValueError("Effect outputs and transition actions are not temporally aligned.")
    action_error = F.smooth_l1_loss(outputs["next_action"], transition_actions, reduction="none").mean(dim=(-1, -2))
    action_loss = (action_error * complete).sum() / complete.sum().clamp_min(1)
    vision_error = F.mse_loss(outputs["next_vision"][:, :-1], outputs["target_visual"][:, 1:], reduction="none").mean(dim=-1)
    vision_loss = (vision_error * complete).sum() / complete.sum().clamp_min(1)
    # Only states adjacent to at least one complete transition are valid CTE
    # evidence.  This prevents padded/invalid transitions from contributing to
    # task clustering or phase progression.
    state_valid = outputs.get("state_valid", valid_mask).bool().clone()
    if complete.shape[1] > 0:
        state_valid[:, 0] &= complete[:, 0]
        state_valid[:, -1] &= complete[:, -1]
        if state_valid.shape[1] > 2:
            state_valid[:, 1:-1] &= complete[:, :-1] | complete[:, 1:]
    task_loss = _task_identity_clustering_loss(outputs["retrieval"], state_valid, semantic_ids, cfg.temperature)
    phase_loss = _phase_progression_loss(outputs["phase"], complete)
    effect_mask = outputs["effect_complete"]
    effect_target = outputs["effect_delta_target"]
    effect_contrastive = 0.5 * (_effect_nce(outputs["effect_outcome_pre"], effect_target, effect_mask, cfg.effect_temperature) + _effect_nce(outputs["effect_outcome_post"], effect_target, effect_mask, cfg.effect_temperature))
    visual_error = F.mse_loss(outputs["effect_outcome_pre"], effect_target, reduction="none").mean(dim=-1)
    effect_visual = (visual_error * effect_mask).sum() / effect_mask.sum().clamp_min(1)
    action_target = summarize_effect_window(outputs["effect_actions"])
    effect_action = (F.smooth_l1_loss(outputs["effect_action"], action_target, reduction="none").mean(dim=-1) * effect_mask).sum() / effect_mask.sum().clamp_min(1)
    effect_align = ((1.0 - (outputs["effect_pre"] * outputs["effect_post"]).sum(dim=-1)) * effect_mask).sum() / effect_mask.sum().clamp_min(1)
    variance, covariance = _vicreg(outputs["effect_post_raw"], effect_mask)
    transition_mask = outputs["transition_complete"]
    transition_target = F.normalize(outputs["transition_delta_target"].detach(), dim=-1)
    transition_error = F.mse_loss(
        outputs["transition_effect_outcome"], transition_target, reduction="none"
    ).mean(dim=-1)
    transition_effect = (transition_error * transition_mask).sum() / transition_mask.sum().clamp_min(1)
    transition_observed_nce = _effect_nce(
        outputs["transition_effect_observed_outcome"],
        transition_target,
        transition_mask,
        cfg.effect_temperature,
    )
    effect_loss = (
        cfg.effect_contrastive_weight * effect_contrastive
        + cfg.effect_action_weight * effect_action
        + cfg.effect_align_weight * effect_align
        + cfg.effect_variance_weight * variance
        + cfg.effect_covariance_weight * covariance
        + cfg.transition_effect_weight * transition_effect
        + cfg.transition_effect_weight * transition_observed_nce
    )
    total = cfg.action_weight * action_loss + cfg.vision_weight * vision_loss + cfg.task_weight * task_loss + cfg.phase_weight * phase_loss + cfg.effect_weight * effect_loss
    return {
        "total": total,
        "action": action_loss,
        "vision": vision_loss,
        "loss_task": task_loss,
        "loss_phase": phase_loss,
        "loss_effect": effect_loss,
        "effect_contrastive": effect_contrastive,
        "effect_visual": effect_visual,
        "effect_action": effect_action,
        "effect_align": effect_align,
        "effect_variance": variance,
        "effect_covariance": covariance,
        "transition_effect": transition_effect,
        "transition_observed_nce": transition_observed_nce,
    }
