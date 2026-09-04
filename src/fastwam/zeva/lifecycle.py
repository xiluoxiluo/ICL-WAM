"""Online episode/attempt lifecycle with causal update ordering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .memory import BriefInteractionTraceBuffer, PersistentInteractionMemory


@dataclass(frozen=True)
class LifecycleConfig:
    bit_size: int = 4
    transition_steps: int = 4
    action_dim: int = 14


class CausalMemoryLifecycle:
    def __init__(self, pim: PersistentInteractionMemory, config: LifecycleConfig | None = None) -> None:
        self.config = config or LifecycleConfig(); self.pim = pim
        self._effect_dim = int(pim.config.effect_dim)
        self.bit = BriefInteractionTraceBuffer(self._effect_dim, self.config.bit_size)
        self._attempt_id = 0; self._transition_index = 0; self._pending: list[Tensor] = []
        self.last_retrieval_sources: list[dict[str, object]] = []

    def reset_episode(self, task_cluster: str, episode_id: str | None = None) -> None:
        self.pim.reset_episode(task_cluster, episode_id=episode_id); self.bit.reset(); self._attempt_id = 0; self._transition_index = 0; self._pending.clear()

    def reset_attempt(self, attempt_id: int) -> None:
        attempt_id = int(attempt_id)
        if attempt_id not in {self._attempt_id, self._attempt_id + 1}:
            raise ValueError("attempt_id must increment by one")
        # Advance PIM first so a failed lifecycle transition (for example,
        # before reset_episode()) cannot leave the transient counters claiming
        # that a new attempt has started.
        self.pim.begin_attempt(attempt_id)
        self._attempt_id = attempt_id; self._transition_index = 0; self.bit.reset(); self._pending.clear()

    def set_effect_dim(self, effect_dim: int) -> None:
        if effect_dim != self._effect_dim:
            self._effect_dim = int(effect_dim); self.bit = BriefInteractionTraceBuffer(self._effect_dim, self.config.bit_size)

    def observe_completed_transition(self, phase: Tensor, effect: Tensor, *, metadata: dict[str, object] | None = None) -> None:
        if phase.ndim == 2 and phase.shape[0] == 1:
            phase = phase[0]
        if effect.ndim == 2 and effect.shape[0] == 1:
            effect = effect[0]
        if phase.ndim != 1 or effect.ndim != 1:
            raise ValueError("online phase/effect must be vectors")
        self.bit.append(effect, self._transition_index)
        self.pim.append_completed(
            task_cluster=self.pim.task_cluster or "",
            phase=phase,
            effect=effect,
            attempt_id=self._attempt_id,
            transition_index=self._transition_index,
            metadata={
                "source_step": self._transition_index,
                "episode_step": self._transition_index * self.config.transition_steps,
                **(metadata or {}),
            },
        )
        self._transition_index += 1

    def memory_inputs(self, phase: Tensor, task_tokens: Tensor) -> dict[str, Tensor]:
        bit = self.bit.tensors(device=phase.device)
        phase_query = phase[0] if phase.ndim == 2 and phase.shape[0] == 1 else phase
        if phase_query.ndim != 1:
            raise ValueError("online phase must be [D] or [1,D]")
        phases, effects, valid, _scores, sources = self.pim.query_tensors(
            phase_query.detach().float().cpu(),
            top_k=self.pim.config.top_k,
            exclude_attempt_id=self._attempt_id,
        )
        self.last_retrieval_sources = sources
        return {"task_tokens": task_tokens, "current_phase": phase_query.unsqueeze(0), "bit_effects": bit.effects.to(phase.device), "bit_mask": bit.valid.to(phase.device), "pim_phases": phases.unsqueeze(0).to(phase.device), "pim_effects": effects.unsqueeze(0).to(phase.device), "pim_mask": valid.unsqueeze(0).to(phase.device)}
