"""Observed-effect online state for the original Zeva main profile.

The CTE is frozen. Only observations of executed actions enter BIT/PIM; neither
the policy's unexecuted horizon nor the CTE's predicted effect is evidence.
Successful-action replay is a separate, explicitly enabled inference mechanism.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from .lifecycle import CausalCTEHistory, CausalMemoryLifecycle


class MainObservedRuntime:
    """Publish each complete observed effect with its own window-start phase.

    WAM can replan every 24 controls while CTE effects span 16 controls. This
    writer therefore tracks raw control boundaries, independently of replans.
    A fresh boundary observation is required after every four executed actions.
    """

    def __init__(self, cte, lifecycle: CausalMemoryLifecycle,
                 frame_encoder: Callable[[Tensor], Tensor] | None = None) -> None:
        if lifecycle.config.memory_semantics != "completed_effect_trace":
            raise ValueError("main_observed requires completed_effect_trace")
        if lifecycle.pim.config.cross_attempt_only:
            raise ValueError("main_observed uses immediate completed-effect PIM, not Ego private traces")
        if (cte.cfg.transition_steps, cte.cfg.effect_window_transitions, cte.cfg.action_dim) != (4, 4, 14):
            raise ValueError("main_observed requires the WAM 4x4/joint14 CTE contract")
        if cte.training or any(parameter.requires_grad for parameter in cte.parameters()):
            raise ValueError("online CTE must be eval() with all parameters frozen")
        self.history = CausalCTEHistory(cte, frame_encoder=frame_encoder)
        self.lifecycle = lifecycle
        self.transition_steps = int(cte.cfg.transition_steps)
        self.effect_window_transitions = int(cte.cfg.effect_window_transitions)
        self.action_dim = int(cte.cfg.action_dim)
        self.executed_steps = 0
        self.observed_steps = 0
        self.published_effects = 0
        self.terminal_observation_step: int | None = None
        self.finalized = False
        self._started = False
        self._actions: list[Tensor] = []
        self._encoded: dict[str, Tensor] | None = None

    def start(self, initial_frame: Tensor) -> None:
        if self._started:
            raise RuntimeError("observed runtime has already started")
        if self.lifecycle.pim.task_cluster is None:
            raise RuntimeError("reset_episode is required before observing the initial frame")
        self.history.reset(initial_frame, initial_raw_step=0)
        self._started = True

    @property
    def needs_boundary_observation(self) -> bool:
        return not self._started or len(self._actions) == self.transition_steps

    def validate_action(self, action: Tensor) -> Tensor:
        if not self._started or self.finalized:
            raise RuntimeError("actions require an active observed attempt")
        if len(self._actions) >= self.transition_steps:
            raise RuntimeError("observe the completed four-action boundary before executing another action")
        action = torch.as_tensor(action).detach().float().cpu()
        if action.shape != (self.action_dim,) or not torch.isfinite(action).all():
            raise ValueError("executed action must be a finite normalized joint14 vector")
        return action

    def record_executed_action(self, action: Tensor) -> None:
        # The caller validates before acting and records only after take_action
        # returns. Failed simulator calls never become completed evidence.
        self._actions.append(self.validate_action(action).clone())
        self.executed_steps += 1

    @torch.no_grad()
    def encoded(self) -> dict[str, Tensor]:
        if not self._started:
            raise RuntimeError("observed runtime has not started")
        if self._encoded is None:
            self._encoded = self.history.forward()
        return self._encoded

    def current_phase(self) -> Tensor:
        if self._actions:
            raise RuntimeError("a policy decision requires the current completed CTE boundary")
        return self.encoded()["phase"][:, -1]

    def _publish(self, *, terminal: bool) -> None:
        encoded = self.encoded()
        complete = encoded["effect_complete"][0]
        while self.published_effects < len(complete):
            index = self.published_effects
            if not bool(complete[index]):
                raise RuntimeError("online effect contains an incomplete or synthetic transition")
            start = index * self.effect_window_transitions
            end_step = (start + self.effect_window_transitions) * self.transition_steps
            if end_step > self.observed_steps:
                raise RuntimeError("cannot publish an effect before its observed end boundary")
            self.lifecycle.observe_completed_effect(
                encoded["phase"][0, start], encoded["effect_post"][0, index],
                metadata={
                    "source_step": start * self.transition_steps,
                    "phase_source_step": start * self.transition_steps,
                    "effect_end_step": end_step,
                    "terminal_attempt": bool(terminal),
                    "terminal_observation_used": bool(terminal and end_step == self.observed_steps),
                    "effect_source": "observed_post",
                    "zeva_profile": "main_observed",
                },
            )
            self.published_effects += 1

    def observe(self, frame: Tensor, *, raw_step: int, terminal: bool = False) -> None:
        if self.finalized:
            if terminal:
                return
            raise RuntimeError("cannot observe after finalizing an attempt")
        if not self._started or int(raw_step) != self.executed_steps:
            raise ValueError("observation raw_step must equal the number of successfully executed actions")
        if len(self._actions) == self.transition_steps:
            if self.observed_steps + self.transition_steps != self.executed_steps:
                raise RuntimeError("missing CTE boundary; refusing to pair old actions with a later frame")
            self.history.append_transition(torch.stack(self._actions), frame, raw_step=self.executed_steps)
            self._actions.clear()
            self.observed_steps = self.executed_steps
            self._encoded = None
            self._publish(terminal=terminal)
        elif not terminal:
            raise ValueError("CTE boundaries require exactly four executed actions")
        if terminal:
            # Partial tails are observed but never padded into a fictitious
            # full CTE effect. The counts below keep this boundary explicit.
            self.terminal_observation_step = self.executed_steps
            self._publish(terminal=True)
            self.finalized = True

    def snapshot(self) -> dict[str, object]:
        effect_steps = self.transition_steps * self.effect_window_transitions
        return {
            "memory_semantics": "completed_effect_trace", "effect_source": "observed_post",
            "executed_steps": self.executed_steps, "observed_steps": self.observed_steps,
            "published_effects": self.published_effects,
            "pending_transition_actions": len(self._actions),
            "partial_effect_actions": self.executed_steps % effect_steps,
            "unpublished_terminal_actions": (self.executed_steps % effect_steps if self.finalized else None),
            "terminal_observation_step": self.terminal_observation_step,
            "terminal_observed": self.terminal_observation_step is not None,
            "finalized": self.finalized,
        }


class PhaseMatchedSuccessReplay:
    """Original-main first-success trace, guarded by phase and replan index.

    A failure never provides action supervision. Only the first externally
    confirmed successful attempt can lock the commands that were actually
    executed. Later attempts still sample the policy before optional blending.
    """

    def __init__(self, *, enabled: bool = False, phase_threshold: float = 0.80,
                 blend: float = 1.0, action_dim: int = 14) -> None:
        if not math.isfinite(phase_threshold) or not -1 <= phase_threshold <= 1:
            raise ValueError("success replay phase threshold must be finite and within [-1,1]")
        if not math.isfinite(blend) or not 0 <= blend <= 1:
            raise ValueError("success replay blend must be finite and within [0,1]")
        self.enabled, self.phase_threshold, self.blend = bool(enabled), float(phase_threshold), float(blend)
        self.action_dim = int(action_dim)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.success_attempt_id: int | None = None
        self._success_trace: list[dict[str, object]] = []
        self._attempt_id = -1
        self.begin_attempt(0)

    def begin_attempt(self, attempt_id: int) -> None:
        if attempt_id not in {self._attempt_id, self._attempt_id + 1}:
            raise ValueError("success replay attempt must remain constant or increment by one")
        self._attempt_id = int(attempt_id)
        self._attempt_chunks: list[dict[str, object]] = []
        self._eligible_at_start = bool(self.enabled and self.success_attempt_id is not None
                                       and attempt_id > self.success_attempt_id)
        self.applied_count = 0
        self.action_count = 0
        self._last_phase_similarity: float | None = None
        self._last_conditioning = False
        self._finalized = False

    def apply(self, phase: Tensor, action: Tensor, *, replan_index: int,
              conditioning: bool) -> tuple[Tensor, int]:
        if self._finalized:
            raise RuntimeError("success replay requires an active attempt")
        action = torch.as_tensor(action).detach().float().cpu()
        phase = torch.as_tensor(phase).detach().float().cpu().reshape(-1)
        if (action.ndim != 2 or action.shape[1] != self.action_dim or not torch.isfinite(action).all()
                or not torch.isfinite(phase).all()):
            raise ValueError("replay requires finite phase and joint14 action tensors")
        self._last_conditioning = bool(conditioning)
        self._last_phase_similarity = None
        replayed = 0
        output = action.clone()
        if self.enabled:
            if replan_index != len(self._attempt_chunks):
                raise ValueError("successful replay requires contiguous replan indices")
            if self._eligible_at_start and conditioning and 0 <= replan_index < len(self._success_trace):
                locked = self._success_trace[replan_index]
                self._last_phase_similarity = float(F.cosine_similarity(phase[None], locked["phase"][None])[0])
                if self._last_phase_similarity >= self.phase_threshold and self.blend > 0:
                    replay = locked["action"]
                    if replay.shape[0] > output.shape[0]:
                        raise ValueError("successful replay chunk exceeds the predicted policy horizon")
                    replayed = int(replay.shape[0])
                    output[:replayed] = (1 - self.blend) * output[:replayed] + self.blend * replay
                    self.applied_count += 1
            self._attempt_chunks.append({"phase": phase.clone(), "actions": [], "replan_index": replan_index})
        return output, replayed

    def record_executed_action(self, action: Tensor, *, replayed: bool) -> None:
        if self._finalized:
            raise RuntimeError("cannot record actions in a finalized replay attempt")
        if self.enabled:
            if not self._attempt_chunks:
                raise RuntimeError("a decision must precede its executed actions")
            action = torch.as_tensor(action).detach().float().cpu()
            if action.shape != (self.action_dim,) or not torch.isfinite(action).all():
                raise ValueError("replay trace requires a finite executed joint14 action")
            self._attempt_chunks[-1]["actions"].append(action.clone())
        if replayed:
            self.action_count += 1

    def finalize_attempt(self, *, success: bool, terminal_observed: bool) -> None:
        if self._finalized:
            return
        if type(success) is not bool:
            raise ValueError("success replay requires an explicit boolean outcome")
        if not terminal_observed:
            raise ValueError("success replay cannot lock an attempt without a fresh terminal observation")
        if self.enabled and success and self.success_attempt_id is None and self._attempt_chunks:
            if any(not chunk["actions"] for chunk in self._attempt_chunks):
                raise ValueError("a successful replay trace cannot include an unexecuted chunk")
            self._success_trace = [
                {"phase": chunk["phase"].clone(), "action": torch.stack(chunk["actions"]),
                 "replan_index": chunk["replan_index"]} for chunk in self._attempt_chunks
            ]
            self.success_attempt_id = self._attempt_id
        self._finalized = True

    def snapshot(self) -> dict[str, object]:
        return {
            "pim_success_replay_enabled": self.enabled,
            "pim_success_replay_phase_threshold": self.phase_threshold,
            "pim_success_replay_blend": self.blend,
            "pim_success_replay_eligible": self._eligible_at_start,
            "pim_success_replay_effective": self.enabled and self._last_conditioning,
            "pim_success_replay_applied_count": self.applied_count,
            "pim_success_replay_action_count": self.action_count,
            "pim_success_replay_source_attempts": ([self.success_attempt_id] if self.applied_count else []),
            "pim_success_trace_attempt_id": self.success_attempt_id,
            "pim_success_trace_chunks": len(self._success_trace),
            "pim_success_replay_phase_similarity": self._last_phase_similarity,
        }
