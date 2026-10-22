from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.robotwin.multiattempt import (
    ATTEMPT_SCHEMA, MANIFEST_SCHEMA, audit_attempt_videos, load_manifest, paired_comparison,
    read_attempts, summarize_attempts, validate_attempts,
)
from fastwam.zeva.attempt_protocol import ModelSeedStream


def _manifest(tmp_path):
    data = {
        "schema": MANIFEST_SCHEMA, "max_attempts": 4,
        "action_dim": 14, "action_horizon": 32, "replan_steps": 24,
        "task_config": "demo_randomized", "instruction_type": "seen",
        "episodes": [
            {"episode_id": f"task:{seed}", "task": "task", "seed": seed, "instruction": "do task"}
            for seed in (7, 8, 9)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return (*load_manifest(path), path)


def _rows(manifest, sha, *, successes=(1, 3, None), enabled=True):
    result = []
    for episode, success_at in zip(manifest["episodes"], successes, strict=True):
        for attempt in range(1, (success_at or manifest["max_attempts"]) + 1):
            success = attempt == success_at
            result.append({
                "schema": ATTEMPT_SCHEMA, "manifest_sha256": sha,
                "run_id": "test", "condition": "on" if enabled else "off",
                "episode_id": episode["episode_id"], "task": episode["task"],
                "scene_seed": episode["seed"], "instruction": episode["instruction"],
                "task_config": manifest["task_config"], "instruction_type": manifest["instruction_type"],
                "action_horizon": 32, "replan_steps": 24, "max_attempts": 4,
                "attempt": attempt, "success": success,
                "status": "success" if success else "timeout",
                "step_count": 1 if success else 10, "step_limit": 10,
                "model_seed": 123, "num_inference_steps": 10,
                "zeva_mode": "pim_on", "cross_memory_enabled": enabled,
                "checkpoint_identity": {"base_sha256": "base", "addon_sha256": "addon", "cte_sha256": "cte", "stats_sha256": "stats"},
            })
    return result


def test_request_seed_stream_is_independent_of_environment_rng():
    first, second = ModelSeedStream(19), ModelSeedStream(19)
    a, b = [], []
    for _ in range(10):
        a.append(first.next_seed())
        random.seed(999)  # What a simulator reset may do to the global RNG.
        random.random()
        b.append(second.next_seed())
    assert a == b and len(set(a)) == len(a)
    assert first.snapshot() == second.snapshot()
    assert first.snapshot()["draws"] == 10


def test_manifest_validates_unique_scenes_and_contract(tmp_path):
    manifest, _, path = _manifest(tmp_path)
    manifest["episodes"].append(manifest["episodes"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(path)
    manifest["episodes"].pop()
    manifest["action_dim"] = 16
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="joint14"):
        load_manifest(path)


def test_video_audit_requires_distinct_nonempty_files(tmp_path):
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    # Byte presence is the promised audit; codec validation is not implied.
    first.write_bytes(b"recording-one")
    second.write_bytes(b"recording-two")
    rows = [{"video_path": str(first)}, {"video_path": str(second)}]
    audit = audit_attempt_videos(rows)
    assert audit["status"] == "passed" and audit["videos"] == 2
    assert audit["total_bytes"] == first.stat().st_size + second.stat().st_size
    with pytest.raises(ValueError, match="Duplicate"):
        audit_attempt_videos([rows[0], rows[0]])
    second.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        audit_attempt_videos(rows)
    second.unlink()
    with pytest.raises(ValueError, match="missing"):
        audit_attempt_videos(rows)
    with pytest.raises(ValueError, match="video_path"):
        audit_attempt_videos([{}])


def test_resolved_policy_config_preserves_model_and_processor_overrides(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from experiments.robotwin.eval_robotwin_single import _write_runtime_policy_config
    from experiments.robotwin.fastwam_policy import deploy_policy

    cfg = OmegaConf.create({
        "override_value": 128,
        "model": {"tokenizer_max_len": "${override_value}"},
        "data": {"train": {
            "processor": {"resize": "${override_value}"},
            "num_frames": 33, "action_video_freq_ratio": 4,
            "video_size": [384, 320],
        }},
        "EVALUATION": {"device": "cpu", "replan_steps": 24, "zeva_mode": "base"},
        "mixed_precision": "no", "eval_num_inference_steps": 10,
    })
    path = tmp_path / "runtime.yaml"
    _write_runtime_policy_config(cfg, path)
    loaded = OmegaConf.load(path)
    assert loaded.model.tokenizer_max_len == 128
    assert loaded.data.train.processor.resize == 128
    assert "${" not in path.read_text()
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    monkeypatch.setattr(deploy_policy, "WorldActionRobotWinPolicy", lambda **kwargs: kwargs)
    monkeypatch.setattr(deploy_policy, "_compose_sim_cfg", lambda **_: pytest.fail("resolved config must not be recomposed"))
    result = deploy_policy.get_model({
        "runtime_config_path": str(path), "ckpt_setting": "base.pt",
        "dataset_stats_path": str(stats), "model_seed": 23,
    })
    assert result["model_cfg"].tokenizer_max_len == 128
    assert result["processor_cfg"].resize == 128
    assert result["action_horizon"] == 32 and result["num_video_frames"] == 9
    assert result["seed"] == 23


def test_report_cli_only_marks_video_audited_runs_complete(tmp_path):
    manifest, sha, path = _manifest(tmp_path)
    rows = _rows(manifest, sha)
    log = tmp_path / "attempts.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    command = [
        sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/report_zeva_robotwin_multiattempt.py"),
        "--manifest", str(path), "--log", str(log),
    ]
    required = subprocess.run(command, capture_output=True, text=True)
    assert required.returncode != 0 and "video_path" in required.stderr
    metadata = subprocess.run(command + ["--metadata-only"], capture_output=True, text=True, check=True)
    result = json.loads(metadata.stdout)
    assert result["completion_status"] == "METADATA_ONLY"
    assert result["video_audit"]["status"] == "skipped_metadata_only"
    for index, row in enumerate(rows):
        video = tmp_path / f"{index}.mp4"
        video.write_bytes(b"test-recording-bytes")
        row["video_path"] = str(video)
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    audited = subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(audited.stdout)
    assert result["completion_status"] == "COMPLETE"
    assert result["video_audit"]["videos"] == len(rows)


def test_cumulative_success_uses_manifest_denominator_and_stops_after_success(tmp_path):
    manifest, sha, _ = _manifest(tmp_path)
    result = summarize_attempts(manifest, sha, _rows(manifest, sha))
    assert result["episodes"] == 3 and result["attempts"] == 8
    assert result["micro"] == {"SR@1": 1 / 3, "SR@2": 1 / 3, "SR@3": 2 / 3, "SR@4": 2 / 3}
    assert result["rescue_after_first_failure"] == 0.5


@pytest.mark.parametrize("problem", ["missing", "duplicate", "after_success", "order", "error", "manifest", "budget", "short_timeout", "incomplete"])
def test_report_rejects_invalid_or_incomplete_runs(tmp_path, problem):
    manifest, sha, _ = _manifest(tmp_path)
    rows = _rows(manifest, sha)
    if problem == "missing":
        rows = [row for row in rows if row["scene_seed"] != 9]
    elif problem == "duplicate":
        rows.insert(2, deepcopy(rows[1]))
    elif problem == "after_success":
        row = deepcopy(rows[0]); row["attempt"] = 2; rows.insert(1, row)
    elif problem == "order":
        rows[1]["attempt"] = 2
    elif problem == "error":
        rows[1]["status"] = "error"
    elif problem == "manifest":
        rows[0]["manifest_sha256"] = "different"
    elif problem == "budget":
        rows[2]["step_limit"] += 1
    elif problem == "short_timeout":
        rows[1]["step_count"] -= 1
    else:
        rows.pop()
    with pytest.raises(ValueError):
        validate_attempts(manifest, sha, rows)


def test_paired_report_requires_same_checkpoint_and_true_memory_off(tmp_path):
    manifest, sha, _ = _manifest(tmp_path)
    on = _rows(manifest, sha)
    off = _rows(manifest, sha, successes=(1, None, None), enabled=False)
    result = paired_comparison(manifest, sha, on, off, bootstrap_samples=100)
    assert result["paired_macro_delta_SR@4"] == pytest.approx(1 / 3)
    for row in off:
        row["checkpoint_identity"]["addon_sha256"] = "other"
    with pytest.raises(ValueError, match="checkpoint_identity"):
        paired_comparison(manifest, sha, on, off)
    off = _rows(manifest, sha, enabled=False)
    for row in off:
        row["zeva_mode"] = "pim_shadow"
    with pytest.raises(ValueError, match="pim_shadow"):
        paired_comparison(manifest, sha, on, off)


def _eval_function(env, model):
    path = Path("third_party/RoboTwin/script/eval_policy.py")
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "eval_policy")

    def step(environment, policy, observation):
        environment.take_action_cnt += 1
        policy.step_count += 1
        environment.eval_success = policy.attempt_id == 1

    def reset(policy):
        policy.reset_calls += 1

    namespace = {
        "np": np, "Path": Path, "time": time, "traceback": traceback,
        "UnStableError": type("UnStableError", (Exception,), {}),
        "eval_function_decorator": lambda policy, name: step if name == "eval" else reset,
        "generate_episode_descriptions": lambda *args: [{"seen": ["do task"]}],
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["eval_policy"]


class _Environment:
    def __init__(self, expert_valid=True):
        self.expert_valid = expert_valid
        self.seeds = []
        self.instructions = []
        self.eval_video_path = None
        self.render_freq = 0
        self.step_lim = 2

    def setup_demo(self, now_ep_num, seed, **kwargs):
        self.seeds.append(seed)
        self.take_action_cnt = 0
        self.eval_success = False

    def play_once(self):
        self.plan_success = self.expert_valid
        return {"info": {}}

    def check_success(self):
        return self.expert_valid

    def close_env(self, **kwargs):
        pass

    def set_instruction(self, instruction):
        self.instructions.append(instruction)

    def get_obs(self):
        return {"frame": self.take_action_cnt}


class _Policy:
    def __init__(self):
        self.reset_calls = 0
        self.finalized = []

    def begin_attempt(self, index):
        self.attempt_id, self.step_count = index, 0

    def finalize_attempt(self, environment, success=False):
        self.finalized.append((self.attempt_id, success))

    def evaluation_metadata(self):
        return {"model_seed": 0, "action_horizon": 32, "replan_steps": 24, "num_inference_steps": 10}


def _environment_args():
    return {"task_name": "task", "policy_name": "policy", "clear_cache_freq": 1, "render_freq": 0, "task_config": "demo_randomized", "ckpt_setting": "test"}


def test_expert_false_never_substitutes_a_fixed_scene_seed():
    env, model = _Environment(expert_valid=False), _Policy()
    with pytest.raises(RuntimeError, match="refusing to substitute"):
        _eval_function(env, model)("task", env, _environment_args(), model, 0, test_num=1, fixed_seed=7)
    assert env.seeds == [7]


def test_manifest_retry_preserves_scene_instruction_and_resets_episode_once(tmp_path):
    env, model = _Environment(), _Policy()
    path = tmp_path / "attempts.jsonl"
    _eval_function(env, model)(
        "task", env, _environment_args(), model, 0, test_num=1, max_attempts=4,
        manifest_episodes=[{"episode_id": "task:7", "task": "task", "seed": 7, "instruction": "exact instruction"}],
        attempt_log_path=path, record_context={"manifest_sha256": "manifest"},
    )
    assert env.seeds == [7, 7, 7]
    assert env.instructions == ["exact instruction", "exact instruction"]
    assert model.reset_calls == 1
    assert model.finalized == [(0, False), (1, True)]
    rows = read_attempts(path)
    assert [row["attempt"] for row in rows] == [1, 2]
    assert [row["status"] for row in rows] == ["timeout", "success"]
    assert [row["reset_scope"] for row in rows] == ["episode", "attempt"]


def test_http_and_native_runtime_export_one_compatible_class():
    native = importlib.import_module("experiments.robotwin.fastwam_policy.deploy_policy")
    http = importlib.import_module("experiments.robotwin.iclwam_policy.robotwin_policy")
    assert http.WorldActionRobotWinPolicy is native.WorldActionRobotWinPolicy
    for name in ("get_model", "reset_model", "encode_obs", "eval", "_mixed_precision_to_model_dtype", "_normalize_mixed_precision"):
        assert getattr(http, name) is getattr(native, name)
    signature = inspect.signature(http.WorldActionRobotWinPolicy)
    assert signature.parameters["cross_memory_enabled"].default is True


def test_runtime_denormalizer_uses_real_merger_tensor_contract():
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
    from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign

    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    meta = {"action": [{"key": "joint", "shape": 14}], "state": [{"key": "joint", "shape": 14}]}
    merger = ConcatLeftAlign(action_target_dim=14, state_target_dim=14)
    merger.set_shape_meta(meta)
    policy.processor = SimpleNamespace(shape_meta=meta, action_state_merger=merger, normalizer=SimpleNamespace(backward=lambda batch: batch), action_state_transforms=None)
    actions = torch.arange(2 * 32 * 14).reshape(2, 32, 14).float()
    np.testing.assert_array_equal(policy._denormalize_action(actions), actions.numpy())


def test_finalization_reads_true_terminal_once_before_committing():
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy

    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.zeva_mode, policy._attempt_finalized, policy.step_count = "pim_on", False, 5
    policy.cte = SimpleNamespace(cfg=SimpleNamespace(transition_steps=4))
    policy._transition_actions, policy._cte_history = [], None
    commits, frames = [], []
    policy.lifecycle = SimpleNamespace(pim=SimpleNamespace(task_cluster="task"), end_attempt=lambda: commits.append(True))
    env = SimpleNamespace(now_obs={"stale": True}, get_obs=lambda: frames.append(5) or {"observation": {}})
    policy.finalize_attempt(env)
    policy.finalize_attempt(env)
    assert frames == [5] and commits == [True]
    assert policy._terminal_observation_step == 5


def test_same_noise_probe_recomputes_prior_and_compares_executed_time_axis():
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy

    calls = []
    class Adapter:
        def prior_for_prompt(self, task, phase, bit, mask, prompt, pim_mask, *, enable_pim=True):
            calls.append(enable_pim)
            return torch.full((2, 32, 14), float(enable_pim)), torch.ones(2, 32, 14)

        def action_prior_residual(self, mean, *, training):
            return mean

        def behavior_prefix_slot(self, task, prompt, mask, *, enable_pim=True):
            return torch.full((2, 1, 8), float(enable_pim))

    captured = []
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.replan_steps = 24
    policy.model = SimpleNamespace(
        device="cpu", zeva_injection_mode="exact_zeva", zeva_behavior_prefix_adapter=Adapter(),
        infer_action_zeva=lambda **kwargs: captured.append(kwargs) or {"action": torch.zeros(2, 32, 14)},
    )
    action = torch.zeros(2, 32, 14); action[:, 24:] = 10
    policy._measure_pim_effect(
        {"seed": 123, "zeva_action_residual": torch.ones(2, 32, 14)}, {"action": action},
        torch.zeros(2, 4), torch.zeros(2, 3), {"bit_effects": torch.zeros(2, 4, 2), "bit_mask": torch.ones(2, 4, dtype=torch.bool)},
        torch.ones(2, 6), torch.ones(2, 8, dtype=torch.bool),
    )
    assert calls == [False, True]
    assert not captured[0]["zeva_action_residual"].any()
    assert captured[0]["seed"] == 123
    assert policy._last_pim_diagnostics["action_exec_delta_rms"] == 0.0
    assert policy._last_pim_diagnostics["executed_action_steps"] == 24
    assert policy._last_pim_diagnostics["prior_mean_delta_rms"] == 1.0


def test_runtime_memory_off_keeps_parent_injections_and_recomputes_prior():
    from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy

    calls, inference = [], []
    class PromptEncoder:
        config = SimpleNamespace(global_dim=4, hidden_dim=6)

        def __call__(self, **kwargs):
            raise AssertionError("Disabled cross memory should not run its encoder")

    class Adapter:
        config = SimpleNamespace(pim_variant="cross_attempt")

        def prior_for_prompt(self, task, phase, bit, bit_mask, prompt, mask, *, enable_pim=True):
            calls.append((task.clone(), prompt.clone(), mask.clone(), enable_pim))
            return torch.zeros(1, 32, 14), torch.ones(1, 32, 14)

        def action_prior_residual(self, mean, *, training):
            return mean + 0.25  # A nonzero parent prior must remain active.

    memory = {
        "bit_effects": torch.ones(1, 4, 2), "bit_mask": torch.ones(1, 4, dtype=torch.bool),
        "pim_mask": torch.ones(1, 192, dtype=torch.bool),
    }
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.model = SimpleNamespace(
        device="cpu", torch_dtype=torch.float32, zeva_injection_mode="exact_zeva",
        zeva_prompt_encoder=PromptEncoder(), zeva_behavior_prefix_adapter=Adapter(),
        encode_prompt=lambda prompt: (torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)),
        infer_action_zeva=lambda **kwargs: inference.append(kwargs) or {"action": torch.zeros(1, 32, 14)},
    )
    from fastwam.zeva.prompt_cache import PolicyPromptCache
    policy._policy_prompt_cache = PolicyPromptCache(policy.model.encode_prompt)
    policy.zeva_mode, policy.cross_memory_enabled = "pim_on", False
    policy._static_task_context_session, policy.task_context_bank = None, None
    policy._cte_history = SimpleNamespace(forward=lambda: {"phase": torch.ones(1, 2, 3)})
    policy._needs_episode_reset = False
    policy.lifecycle = SimpleNamespace(memory_inputs=lambda phase, task: memory)
    policy._build_robotwin_image_tensor = lambda obs: torch.zeros(1, 3, 8, 8)
    policy._normalize_state = lambda state: torch.zeros(1, 14)
    policy._denormalize_action = lambda action: action.numpy()
    policy._model_seed_stream = ModelSeedStream(12)
    policy._attempt_id, policy._replan_index = 1, 0
    policy.action_horizon, policy.replan_steps, policy.num_inference_steps = 32, 24, 10
    policy.negative_prompt, policy.text_cfg_scale, policy.sigma_shift = "", 1.0, None
    policy.rand_device, policy.tiled, policy.timing_enabled = "cpu", False, False
    policy.pim_diagnostic_interval = 0
    result = policy._infer_action_chunk({"joint_action": {"vector": np.zeros(14)}}, "task")
    assert result.shape == (32, 14)
    assert len(calls) == 1 and calls[0][-1] is False
    assert not calls[0][1].any() and not calls[0][2].any()
    assert memory["pim_mask"].all()  # Histories themselves remain available.
    assert inference[0]["zeva_mode"] == "pim_on"
    assert inference[0]["enable_prefix_injection"] is True
    assert inference[0]["enable_action_prior"] is True
    assert torch.all(inference[0]["zeva_action_residual"] == 0.25)
    assert policy._model_seed_stream.draws == 1
