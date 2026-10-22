"""Real observed boundaries, terminal flushes, and guarded first-success replay."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fastwam.zeva.lifecycle import CausalMemoryLifecycle, LifecycleConfig
from fastwam.zeva.main_runtime import MainObservedRuntime, PhaseMatchedSuccessReplay
from fastwam.zeva.memory import PersistentInteractionMemory, PersistentInteractionMemoryConfig


class _ObservedCTE(nn.Module):
    """Expose exact observed deltas and a deliberately unrelated prediction."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)
        self.cfg = SimpleNamespace(transition_steps=4, effect_window_transitions=4,
                                   action_dim=14, image_channels=3, phase_dim=2, effect_dim=2)
        self.eval()

    def forward(self, frames, actions, **kwargs):
        values = frames.mean(dim=(-1, -2, -3))
        phase = torch.stack((torch.ones_like(values), values), dim=-1)
        starts = torch.arange((frames.shape[1] - 1) // 4) * 4
        deltas = values[:, starts + 4] - values[:, starts]
        effects = torch.stack((deltas, torch.ones_like(deltas)), dim=-1)
        return {"phase": phase, "effect_post": effects, "effect_pre": torch.full_like(effects, 999),
                "effect_complete": torch.ones(effects.shape[:2], dtype=torch.bool)}


def _lifecycle():
    memory = PersistentInteractionMemory(PersistentInteractionMemoryConfig(
        phase_dim=2, effect_dim=2, capacity=8, top_k=2, merge_threshold=1.0,
    ))
    result = CausalMemoryLifecycle(memory, LifecycleConfig())
    result.reset_episode("task", "scene-0")
    return result


def _frame(step):
    return torch.full((3, 2, 2), float(step))


def test_observed_windows_pair_their_own_start_phase_independently_of_replans():
    lifecycle = _lifecycle()
    runtime = MainObservedRuntime(_ObservedCTE(), lifecycle)
    runtime.start(_frame(0))
    for step in range(1, 33):
        runtime.record_executed_action(torch.full((14,), float(step)))
        if step % 4 == 0:
            runtime.observe(_frame(step), raw_step=step, terminal=step == 32)
        if step < 16:
            assert len(lifecycle.pim) == 0
    entries = lifecycle.pim.entries
    assert len(entries) == 2
    assert [item.metadata["phase_source_step"] for item in entries] == [0, 16]
    assert [item.metadata["effect_end_step"] for item in entries] == [16, 32]
    torch.testing.assert_close(entries[1].phase, F.normalize(torch.tensor([1.0, 16.0]), dim=0))
    torch.testing.assert_close(entries[0].effect, F.normalize(torch.tensor([16.0, 1.0]), dim=0))
    assert entries[1].metadata["terminal_observation_used"] is True
    runtime.observe(_frame(32), raw_step=32, terminal=True)
    assert len(lifecycle.pim) == 2
    assert runtime.published_effects == 2


def test_partial_terminal_is_observed_but_never_fabricated_as_a_complete_effect():
    lifecycle = _lifecycle()
    runtime = MainObservedRuntime(_ObservedCTE(), lifecycle)
    runtime.start(_frame(0))
    for step in range(1, 18):
        runtime.record_executed_action(torch.zeros(14))
        if step % 4 == 0:
            runtime.observe(_frame(step), raw_step=step)
    runtime.observe(_frame(17), raw_step=17, terminal=True)
    assert runtime.snapshot()["unpublished_terminal_actions"] == 1
    assert runtime.snapshot()["terminal_observation_step"] == 17
    assert len(lifecycle.pim) == 1
    lifecycle.end_attempt()
    lifecycle.reset_attempt(1)
    assert not lifecycle.bit.tensors().valid.any()
    assert lifecycle.memory_inputs(torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2))["pim_mask"].any()


def test_missing_boundary_and_unfrozen_cte_fail_before_additional_actions():
    encoder = _ObservedCTE()
    encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        MainObservedRuntime(encoder, _lifecycle())
    encoder.requires_grad_(False)
    runtime = MainObservedRuntime(encoder, _lifecycle())
    runtime.start(_frame(0))
    for _ in range(4):
        runtime.record_executed_action(torch.zeros(14))
    with pytest.raises(RuntimeError, match="boundary"):
        runtime.record_executed_action(torch.zeros(14))
    with pytest.raises(ValueError, match="raw_step"):
        runtime.observe(_frame(8), raw_step=8)
    assert runtime.executed_steps == 4 and runtime.observed_steps == 0
    runtime.observe(_frame(4), raw_step=4)
    with pytest.raises(ValueError, match="finite"):
        runtime.record_executed_action(torch.full((14,), float("nan")))
    assert runtime.executed_steps == 4


def test_replay_locks_only_first_success_and_preserves_index_and_phase_guards():
    replay = PhaseMatchedSuccessReplay(enabled=True, blend=0.25)
    phase = torch.tensor([1.0, 0.0])
    fresh = torch.zeros(32, 14)
    replay.apply(phase, fresh, replan_index=0, conditioning=True)
    replay.record_executed_action(torch.full((14,), -9.0), replayed=False)
    replay.finalize_attempt(success=False, terminal_observed=True)
    assert replay.success_attempt_id is None
    replay.begin_attempt(1)
    _, used = replay.apply(phase, fresh, replan_index=0, conditioning=True)
    assert used == 0
    replay.record_executed_action(torch.full((14,), 8.0), replayed=False)
    replay.finalize_attempt(success=True, terminal_observed=True)
    assert replay.snapshot()["pim_success_replay_eligible"] is False
    replay.begin_attempt(2)
    output, used = replay.apply(phase, fresh, replan_index=0, conditioning=True)
    assert used == 1
    torch.testing.assert_close(output[0], torch.full((14,), 2.0))
    replay.record_executed_action(output[0], replayed=True)
    # No nearest-neighbor fallback to replan 0 when replan 1 is absent.
    _, used = replay.apply(phase, fresh, replan_index=1, conditioning=True)
    assert used == 0
    replay.record_executed_action(torch.ones(14), replayed=False)
    replay.finalize_attempt(success=True, terminal_observed=True)
    assert replay.success_attempt_id == 1
    replay.begin_attempt(3)
    _, used = replay.apply(-phase, fresh, replan_index=0, conditioning=True)
    assert used == 0
    replay.begin_attempt(3)
    _, used = replay.apply(phase, fresh, replan_index=0, conditioning=False)
    assert used == 0
    replay.reset_episode()
    assert replay.success_attempt_id is None


def _policy(*, replay_enabled=False):
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
    from fastwam.zeva.attempt_protocol import ModelSeedStream

    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.zeva_profile, policy._main_observed, policy._ego_cte = "main_observed", True, False
    policy.zeva_mode, policy.cross_memory_enabled = "pim_on", True
    policy.cte, policy.lifecycle = _ObservedCTE(), _lifecycle()
    policy._cte_frame_encoder = None
    policy._main_runtime, policy._cte_history, policy._main_episode_instruction = None, None, None
    policy._success_replay = PhaseMatchedSuccessReplay(enabled=replay_enabled)
    policy._pending_replay_actions, policy.pending_actions = deque(), deque()
    policy._transition_actions = []
    policy._needs_episode_reset, policy._attempt_finalized = True, False
    policy._attempt_id = policy._replan_index = policy.episode_count = policy.step_count = 0
    policy._static_task_context_session = None
    policy._last_pim_diagnostics = {}
    policy._terminal_observation_step = None
    policy._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
    policy.seed, policy.timing_enabled = 12, False
    policy._model_seed_stream = ModelSeedStream(12)
    policy.model = nn.Linear(1, 1).eval().requires_grad_(False)
    policy.model.device = "cpu"
    policy.checkpoint_identity = {"addon_sha256": "test-main"}
    policy.action_horizon, policy.replan_steps, policy.num_inference_steps = 32, 24, 10
    policy._build_robotwin_image_tensor = lambda obs: _frame(obs["step"]).unsqueeze(0)
    policy._normalize_action = lambda action, state: torch.as_tensor(action).clone()

    def infer(**kwargs):
        policy._model_seed_stream.next_seed()
        policy._replan_index += 1
        return np.full((32, 14), policy._attempt_id + 1, dtype=np.float32)

    policy._infer_action_chunk = infer
    return policy


class _Environment:
    def __init__(self):
        self.actions = []
        self.observations = 0

    def get_instruction(self):
        return "task"

    def get_obs(self):
        self.observations += 1
        return {"observation": {}, "step": len(self.actions), "joint_action": {"vector": np.zeros(14)}}

    def take_action(self, action, *, action_type):
        assert action_type == "qpos"
        self.actions.append(np.array(action, copy=True))


def test_deployment_flushes_terminal_before_retry_and_replay_keeps_sampling():
    policy = _policy(replay_enabled=True)
    weights = {key: value.clone() for key, value in policy.model.state_dict().items()}
    env = _Environment()
    for _ in range(16):
        policy.step(env, env.get_obs())
    assert len(policy.lifecycle.pim) == 0  # last post-action observation is still pending
    policy.finalize_attempt(env, success=False)
    observed = env.observations
    policy.finalize_attempt(env, success=False)
    assert env.observations == observed
    assert len(policy.lifecycle.pim) == 1
    policy.begin_attempt(1)
    assert len(policy.lifecycle.pim) == 1
    assert not policy.lifecycle.bit.tensors().valid.any()
    env = _Environment()
    for _ in range(2):
        policy.step(env, env.get_obs())
    policy.finalize_attempt(env, success=True)
    assert policy.evaluation_metadata()["pim_success_replay_applied_count"] == 0
    policy.begin_attempt(2)
    env = _Environment()
    for _ in range(3):
        policy.step(env, env.get_obs())
    # First-success commands from attempt 1 affect only their executed prefix.
    assert [float(action[0]) for action in env.actions] == [2.0, 2.0, 3.0]
    assert policy._model_seed_stream.draws == 3
    metadata = policy.evaluation_metadata()
    assert metadata["pim_success_replay_eligible"] is True
    assert metadata["pim_success_replay_applied_count"] == 1
    assert metadata["pim_success_replay_action_count"] == 2
    assert metadata["inference_parameters_frozen"] is True
    for key, value in policy.model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    policy.reset()
    assert not policy.evaluation_metadata()["pim_success_replay_eligible"]
    assert policy.evaluation_metadata()["pim_entries"] == 0


def test_deployment_rejects_missing_boundaries_and_failed_normalization_before_acting():
    policy, env = _policy(), _Environment()
    for _ in range(4):
        policy.step(env, env.get_obs())
    assert policy.should_request_observation()
    with pytest.raises(ValueError, match="fresh observation"):
        policy.step(env, None)
    assert len(env.actions) == 4
    with pytest.raises(RuntimeError, match="terminal observation"):
        policy.begin_attempt(1)
    policy._normalize_action = lambda action, state: torch.full((14,), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        policy.step(env, env.get_obs())
    assert len(env.actions) == 4
