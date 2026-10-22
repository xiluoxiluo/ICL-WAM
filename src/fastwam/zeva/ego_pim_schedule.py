"""Zeva-Ego PIM curriculum, selected once per optimizer step.

Workers provide candidate histories; they never own the training step. This
keeps prefetch, gradient accumulation and distributed ranks on one mode.
"""
from __future__ import annotations

import torch
from .cross_attempt_cache import HISTORY_KINDS


def select_history_kind(step: int, *, seed: int, warmup_steps: int,
                        available_kinds: tuple[str, ...] | list[str]) -> int:
    """Index the reference fixed cycle using the global optimizer step.

    ``seed`` is retained for caller compatibility; mode selection uses no RNG.
    Unavailable real history kinds are removed before indexing the cycle.
    """
    del seed
    if step < 0 or warmup_steps < 1:
        raise ValueError("step must be nonnegative and warmup_steps positive")
    available = set(available_kinds)
    if not available or not available.issubset(HISTORY_KINDS[:3]):
        raise ValueError("at least one real nonempty history kind is required")
    active_schedule = []
    for index, (kind, count) in enumerate(zip(HISTORY_KINDS[:3], (5, 2, 1))):
        if kind in available:
            active_schedule.extend([index] * count)
    if step >= warmup_steps:
        active_schedule.extend([3] * 2)
    return active_schedule[step % len(active_schedule)]


def apply_history_kind(sample: dict, kind: int) -> dict:
    """Select one completed attempt (or none), preserving the current effect."""
    if kind not in range(4):
        raise ValueError("history kind must be in [0,3]")
    result = dict(sample)
    mask = sample["pim_mask"]
    if mask.ndim != 2:
        raise ValueError("history curriculum expects a collated batch [B,N]")
    if kind == 3:
        for name in ("pim_phases", "pim_effects", "pim_mask"):
            result[name] = torch.zeros_like(sample[name])
        if "pim_scores" in sample:
            result["pim_scores"] = torch.full_like(sample["pim_scores"], -torch.inf)
    else:
        available = sample["pim_history_available"]
        if available.shape != (mask.shape[0], 3) or not bool(available[:, kind].all()):
            raise ValueError("scheduled history kind is unavailable for a query; rebuild pairings")
        for target, source in (("pim_phases", "pim_history_phases"), ("pim_effects", "pim_history_effects"),
                               ("pim_mask", "pim_history_masks"), ("pim_scores", "pim_history_scores")):
            candidates = sample[source]
            if candidates.shape[:2] != (mask.shape[0], 3):
                raise ValueError(f"invalid candidate bundle: {source}")
            result[target] = candidates[:, kind]
        if not bool(result["pim_mask"].any(dim=-1).all()):
            raise ValueError("nonempty history mode contains an empty attempt")
    result["pim_history_kind"] = torch.full((mask.shape[0],), kind, dtype=torch.long, device=mask.device)
    result["pim_history_count"] = torch.full((mask.shape[0],), int(kind != 3), dtype=torch.long, device=mask.device)
    return result


def eap_lr_factor(step: int, warmup_steps: int) -> float:
    return float(step >= warmup_steps)
