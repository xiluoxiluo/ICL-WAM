"""Deterministic phase-conditioned offline retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalResult:
    phases: Tensor
    effects: Tensor
    mask: Tensor
    scores: Tensor
    sources: tuple[dict[str, object], ...]


class MemoryBank:
    """Offline proxy bank. Retrieval is phase-only and excludes current episode."""

    def __init__(self, phase_dim: int = 128, effect_dim: int = 128, top_k: int = 4, same_task_only: bool = True) -> None:
        self.phase_dim, self.effect_dim, self.top_k, self.same_task_only = int(phase_dim), int(effect_dim), int(top_k), bool(same_task_only)
        self._rows: list[dict[str, object]] = []

    def add(
        self,
        phase: Tensor,
        effect: Tensor,
        *,
        episode_id: str,
        task_id: int | str,
        attempt_id: int = 0,
        transition_index: int = 0,
        effect_index: int | None = None,
        episode_step: int | None = None,
        window_index: int | None = None,
        valid: bool = True,
    ) -> None:
        if valid:
            if phase.shape != (self.phase_dim,) or effect.shape != (self.effect_dim,):
                raise ValueError("memory bank feature shape mismatch")
            if not torch.isfinite(phase).all() or not torch.isfinite(effect).all():
                raise ValueError("memory bank refuses non-finite features")
            self._rows.append({"phase": F.normalize(phase.detach().float().cpu(), dim=0), "effect": F.normalize(effect.detach().float().cpu(), dim=0), "episode_id": str(episode_id), "task_id": task_id, "attempt_id": int(attempt_id), "transition_index": int(transition_index), "effect_index": None if effect_index is None else int(effect_index), "episode_step": None if episode_step is None else int(episode_step), "window_index": None if window_index is None else int(window_index)})

    def retrieve(self, query_phase: Tensor, *, episode_id: str | None = None, task_id: int | str | None = None, top_k: int | None = None) -> RetrievalResult:
        k = self.top_k if top_k is None else int(top_k)
        if k < 1:
            raise ValueError("top_k must be positive")
        if query_phase.shape != (self.phase_dim,):
            raise ValueError("query_phase shape mismatch")
        if not torch.isfinite(query_phase).all():
            raise ValueError("memory bank refuses a non-finite query phase")
        query = F.normalize(query_phase.detach().float().cpu(), dim=0)
        rows = [row for row in self._rows if (episode_id is None or row["episode_id"] != str(episode_id)) and (not self.same_task_only or task_id is None or row["task_id"] == task_id)]
        scores = torch.tensor([float(row["phase"] @ query) for row in rows], dtype=torch.float32) if rows else torch.empty(0)
        if rows:
            order = torch.argsort(scores, descending=True, stable=True)[:k]
            rows, scores = [rows[int(i)] for i in order], scores[order]
        phases = torch.zeros((k, self.phase_dim), dtype=torch.float32)
        effects = torch.zeros((k, self.effect_dim), dtype=torch.float32)
        mask = torch.zeros(k, dtype=torch.bool)
        sources: list[dict[str, object]] = []
        for i, row in enumerate(rows[:k]):
            phases[i], effects[i], mask[i] = row["phase"], row["effect"], True
            sources.append({key: row[key] for key in ("episode_id", "task_id", "attempt_id", "transition_index", "effect_index", "episode_step", "window_index")})
        padded_scores = torch.full((k,), -torch.inf, dtype=torch.float32); padded_scores[:len(scores)] = scores
        return RetrievalResult(phases, effects, mask, padded_scores, tuple(sources))
