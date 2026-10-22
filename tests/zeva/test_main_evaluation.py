"""CPU mock rollouts exercise the actual delegated RoboTwin attempt loop."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import pytest

from fastwam.zeva_main_evaluation import (
    ATTEMPT_SCHEMA, MANIFEST_SCHEMA, NOISE_PROTOCOL, load_manifest, make_manifest,
    paired_comparison, read_attempts, summarize_attempts, validate_attempts,
)


ROOT = Path(__file__).resolve().parents[2]


def _hash(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


def _manifest(tmp_path, protocol="fixed_attempts", attempts=4, replay=False, seeds=(7,)):
    source = {"schema": "source-scenes", "task_config": "demo_randomized", "instruction_type": "seen",
              "episodes": [{"episode_id": f"task:{seed}", "task": "task", "seed": seed, "instruction": "do the task"} for seed in seeds]}
    manifest = make_manifest(source, source_sha256=_hash("source"), protocol=protocol,
                             max_attempts=attempts, model_seed=123, success_replay=replay)
    path = tmp_path / f"{protocol}_manifest.json"
    path.write_text(json.dumps(manifest))
    return (*load_manifest(path), path)


class _Environment:
    def __init__(self):
        self.eval_video_path = None
        self.render_freq = 0
        self.step_lim = 2
        self.seeds, self.instructions = [], []

    def setup_demo(self, now_ep_num, seed, **kwargs):
        self.seed = seed
        self.seeds.append(seed)
        self.take_action_cnt = 0
        self.eval_success = False
        self.plan_success = True

    def play_once(self):
        return {"info": {}}

    def check_success(self):
        return True

    def close_env(self, **kwargs):
        pass

    def set_instruction(self, instruction):
        self.instructions.append(instruction)

    def get_obs(self):
        return {"seed": self.seed}


class _Policy:
    def __init__(self, outcomes, enabled=True, replay=False):
        self.outcomes, self.enabled, self.replay = outcomes, enabled, replay
        self.draws = 0
        self.reset_calls = 0
        self.attempt_calls = []
        self.finalized = []
        self.entries = 0
        self.first_success = None
        self.eligible = False
        self.step_count = self.replans = self.replay_chunks = 0

    def reset(self):
        self.reset_calls += 1
        self.entries = 0
        self.first_success = None

    def begin_attempt(self, attempt_id):
        self.attempt_id = attempt_id
        self.attempt_calls.append(attempt_id)
        self.step_count = self.replans = self.replay_chunks = 0
        self.eligible = self.first_success is not None

    def step(self, env, obs):
        assert obs["seed"] == env.seed
        self.draws += 1
        self.replans += 1
        self.step_count += 1
        self.entries += 1
        if self.eligible and self.replay and self.enabled:
            self.replay_chunks += 1
        env.take_action_cnt += 1
        env.eval_success = self.outcomes[env.seed][self.attempt_id]

    def finalize_attempt(self, env, success):
        self.finalized.append((env.seed, self.attempt_id, success))
        if success and self.first_success is None:
            self.first_success = self.attempt_id

    def evaluation_metadata(self):
        return {
            "evaluation_profile": "main_observed", "memory_semantics": "completed_effect_trace",
            "inference_parameters_frozen": True, "observed_effect_source": "observed_post",
            "model_seed": 123, "num_inference_steps": 10, "action_horizon": 32, "replan_steps": 24,
            "zeva_mode": "pim_on", "cross_memory_enabled": self.enabled,
            "checkpoint_identity": {key: _hash(key) for key in ("base_sha256", "addon_sha256", "cte_sha256", "stats_sha256")},
            "seed_stream": {"protocol": NOISE_PROTOCOL, "seed": 123, "dtype": "float32", "device": "cpu", "draws": self.draws, "state_sha256": _hash(self.draws)},
            "replans": self.replans, "pim_entries": self.entries,
            "pim_success_replay_enabled": self.replay, "pim_success_replay_eligible": self.eligible,
            "pim_success_replay_applied_count": self.replay_chunks,
            "pim_success_replay_action_count": self.replay_chunks,
            "pim_success_replay_phase_threshold": .80, "pim_success_replay_blend": 1.0,
            "pim_success_replay_source_attempts": [self.first_success] if self.replay_chunks else [],
            "main_observed_runtime": {
                "memory_semantics": "completed_effect_trace", "effect_source": "observed_post",
                "executed_steps": self.step_count, "observed_steps": self.step_count - self.step_count % 4,
                "terminal_observation_step": self.step_count, "terminal_observed": True, "finalized": True,
            },
        }


def _actual_loop():
    path = ROOT / "third_party/RoboTwin/script/eval_policy.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "eval_policy")
    namespace = {
        "Path": Path, "time": time, "traceback": traceback,
        "UnStableError": type("UnStableError", (Exception,), {}),
        "eval_function_decorator": lambda _, method: (lambda env, policy, obs: policy.step(env, obs)) if method == "eval" else (lambda policy: policy.reset()),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["eval_policy"]


def _run(tmp_path, manifest, sha, outcomes, enabled=True):
    policy = _Policy(outcomes, enabled=enabled, replay=manifest["pim_success_replay_enabled"])
    env = _Environment()
    log = tmp_path / ("on.jsonl" if enabled else "off.jsonl")
    args = {"task_name": "task", "policy_name": "mock", "task_config": "demo_randomized", "ckpt_setting": "mock", "clear_cache_freq": 1, "render_freq": 0}
    _, successes = _actual_loop()(
        "task", env, args, policy, 0, test_num=len(manifest["episodes"]),
        instruction_type="seen", max_attempts=manifest["max_attempts"], manifest_episodes=manifest["episodes"],
        attempt_log_path=log, attempt_protocol=manifest["attempt_protocol"], evaluation_profile="main_observed",
        record_context={"manifest_sha256": sha, "run_id": "mock_run", "condition": "memory_on" if enabled else "memory_off",
                        "task_config": "demo_randomized", "instruction_type": "seen"},
    )
    return policy, env, read_attempts(log), successes, log


def test_fixed_attempts_really_continues_after_success_and_counts_later_failure(tmp_path):
    manifest, sha, _ = _manifest(tmp_path, replay=True)
    policy, env, rows, successes, _ = _run(tmp_path, manifest, sha, {7: [False, True, False, True]})
    assert [row["success"] for row in rows] == [False, True, False, True]
    assert policy.reset_calls == 1 and policy.attempt_calls == [0, 1, 2, 3]
    assert len(policy.finalized) == 4 and successes == 1
    assert set(env.seeds) == {7} and set(env.instructions) == {"do the task"}
    assert [row["post_success_attempt"] for row in rows] == [False, False, True, True]
    assert [row["pim_success_replay_applied_count"] for row in rows][:2] == [0, 0]
    summary = summarize_attempts(manifest, sha, rows)
    assert summary["primary_metric"] == "post_first_success_stability"
    assert summary["micro"]["SR@1"] == 0
    assert summary["micro"]["CSR"] == {"CSR@1": 0, "CSR@2": 1, "CSR@3": 1, "CSR@4": 1}
    assert summary["micro"]["attempt_success"]["3"]["rate"] == 0
    stability = summary["micro"]["post_first_success_stability"]
    assert stability["attempt_rate"] == .5 and stability["all_remaining_success_rate"] == 0
    assert stability["attempts_with_replay"] == 2
    assert '"SR@4"' not in json.dumps(summary)


def test_stop_on_success_has_no_post_success_observations(tmp_path):
    manifest, sha, _ = _manifest(tmp_path, protocol="stop_on_success")
    policy, _, rows, _, _ = _run(tmp_path, manifest, sha, {7: [False, True, True, True]})
    assert len(rows) == 2 and policy.attempt_calls == [0, 1]
    assert rows[1]["pim_entries_before"] == rows[0]["pim_entries_after"] > 0
    summary = summarize_attempts(manifest, sha, rows)
    assert summary["micro"]["CSR"]["CSR@4"] == 1
    assert summary["micro"]["post_first_success_stability"]["attempt_rate"] is None
    assert summary["micro"]["first_failure_recovery"]["rate"] == 1


def test_single_attempt_resets_memory_between_randomized_scenes(tmp_path):
    manifest, sha, _ = _manifest(tmp_path, protocol="single_attempt", attempts=1, seeds=(7, 8))
    policy, _, rows, successes, _ = _run(tmp_path, manifest, sha, {7: [True], 8: [False]})
    assert policy.reset_calls == 2 and policy.attempt_calls == [0, 0]
    assert [row["pim_entries_before"] for row in rows] == [0, 0]
    assert rows[1]["seed_stream_before"] == rows[0]["seed_stream"]
    summary = summarize_attempts(manifest, sha, rows)
    assert successes == 1 and summary["primary_metric"] == "SR@1"
    assert summary["micro"]["SR@1"] == .5


@pytest.mark.parametrize("problem", ["replay_before_success", "reset_on_retry", "rng_reseed", "partial", "wrong_weights", "predictive_memory", "unknown_profile", "short_timeout", "invalid_schema", "wrong_first_success", "train_mode", "missing_terminal", "wrong_replay_source"])
def test_report_rejects_invalid_execution_history(tmp_path, problem):
    manifest, sha, _ = _manifest(tmp_path, replay=True)
    _, _, rows, _, _ = _run(tmp_path, manifest, sha, {7: [False, True, False, True]})
    if problem == "replay_before_success":
        rows[0].update(pim_success_replay_eligible=True, pim_success_replay_applied_count=1, pim_success_replay_action_count=1)
    elif problem == "reset_on_retry":
        rows[1].update(reset_scope="episode", memory_reset=True)
    elif problem == "rng_reseed":
        rows[1]["seed_stream_before"] = deepcopy(rows[0]["seed_stream_before"])
    elif problem == "partial":
        rows.pop()
    elif problem == "wrong_weights":
        rows[1]["checkpoint_identity"]["addon_sha256"] = _hash("different")
    elif problem == "predictive_memory":
        rows[0]["memory_semantics"] = "predicted_effect_trace"
    elif problem == "unknown_profile":
        rows[0]["evaluation_profile"] = "zeva_ego"
    elif problem == "short_timeout":
        rows[0]["step_count"] = 1
    elif problem == "invalid_schema":
        rows[0]["schema"] = "iclwam-robotwin-attempt-v1"
    elif problem == "train_mode":
        rows[0]["inference_parameters_frozen"] = False
    elif problem == "missing_terminal":
        rows[0]["main_observed_runtime"]["terminal_observed"] = False
    elif problem == "wrong_replay_source":
        rows[2]["pim_success_replay_source_attempts"] = [0]
    else:
        rows[2]["first_success_attempt_before"] = None
    with pytest.raises(ValueError):
        validate_attempts(manifest, sha, rows)


def test_fixed_report_requires_all_k_even_after_success(tmp_path):
    manifest, sha, _ = _manifest(tmp_path)
    _, _, rows, _, _ = _run(tmp_path, manifest, sha, {7: [True, True, True, True]})
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_attempts(manifest, sha, rows[:1])


def test_paired_comparison_matches_weights_and_labels_replay(tmp_path):
    manifest, sha, _ = _manifest(tmp_path, replay=True)
    _, _, on, _, _ = _run(tmp_path, manifest, sha, {7: [False, True, True, True]}, enabled=True)
    _, _, off, _, _ = _run(tmp_path, manifest, sha, {7: [False, False, False, False]}, enabled=False)
    result = paired_comparison(manifest, sha, on, off, bootstrap_samples=30)
    assert result["paired_delta_SR@1"] == 0 and result["paired_delta_CSR@4"] == 1
    assert result["pure_failed_attempt_memory_effect_identified"] is False
    assert result["success_replay_contaminates_pre_first_success"] is False
    assert not any(row["pim_success_replay_applied_count"] for row in off)
    for row in off:
        row["checkpoint_identity"]["addon_sha256"] = _hash("other")
    with pytest.raises(ValueError, match="checkpoint_identity"):
        paired_comparison(manifest, sha, on, off)


def _dry_command(output):
    command = [sys.executable, "-S", str(ROOT / "scripts/eval_zeva_main_robotwin.py"), "--output-dir", str(output), "--dry-run"]
    for name in ("checkpoint", "cte-checkpoint", "addon-checkpoint", "dataset-stats", "success-manifest", "task-context-bank", "task-context-retrieval-checkpoint"):
        command.extend([f"--{name}", f"/not-present/{name}.pt"])
    return command


@pytest.mark.parametrize("protocol,attempts", [("single_attempt", 1), ("stop_on_success", 4), ("fixed_attempts", 6)])
def test_stdlib_only_dry_run_is_wired_to_real_simulator_entrypoint(tmp_path, protocol, attempts):
    output = tmp_path / "dry"
    command = _dry_command(output) + ["--protocol", protocol, "--max-attempts", str(attempts), "--tasks", "beat_block_hammer"]
    if protocol == "fixed_attempts":
        command.append("--success-replay")
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["artifact_validation"] == "not_checked_dry_run"
    assert plan["attempt_protocol"] == protocol and plan["episodes_per_condition"] == 20
    assert len(plan["commands"]) == 2
    assert not (output / "COMPLETE").exists()
    manifest, _ = load_manifest(output / "manifest.json")
    assert manifest["schema"] == MANIFEST_SCHEMA and manifest["max_attempts"] == attempts
    assert manifest["reference_benchmark_reproduction"] is False
    for item in plan["commands"]:
        assert item["command"][1].endswith("experiments/robotwin/eval_robotwin_single.py")
        settings = {key.lstrip("+"): json.loads(value) for entry in item["command"][4:] for key, value in [entry.split("=", 1)]}
        assert settings["EVALUATION.evaluation_profile"] == "main_observed"
        assert settings["EVALUATION.attempt_protocol"] == protocol
        assert settings["EVALUATION.max_attempts"] == attempts
        assert settings["EVALUATION.cross_memory_enabled"] is (item["condition"] == "memory_on")
        assert settings["model.zeva.task_context.mode"] == "static"
        assert settings["EVALUATION.skip_get_obs_within_replan"] is False
    repeated = subprocess.run(command, text=True, capture_output=True)
    assert repeated.returncode != 0 and "fresh" in repeated.stderr


@pytest.mark.parametrize("foundation", ["frozen", "joint"])
def test_main_dry_commands_compose_the_actual_main_model_contract(tmp_path, foundation):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.zeva_numerical_contract import config_numerical_contract

    command = _dry_command(tmp_path / foundation) + ["--tasks", "beat_block_hammer", "--foundation-mode", foundation,
                                                    "--protocol", "fixed_attempts", "--max-attempts", "6", "--success-replay"]
    plan = json.loads(subprocess.run(command, text=True, capture_output=True, check=True).stdout)
    for item in plan["commands"]:
        delegated = item["command"]
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
            cfg = compose(config_name=delegated[3], overrides=delegated[4:])
        zeva = OmegaConf.to_container(cfg.model.zeva, resolve=True)
        evaluation = OmegaConf.to_container(cfg.EVALUATION, resolve=True)
        assert delegated[3] == "sim_robotwin_zeva_main"
        assert zeva["profile"] == "main_observed" and config_numerical_contract(zeva) == "zeva_main_v1"
        assert zeva["foundation_mode"] == foundation and zeva["freeze_fastwam"] is (foundation == "frozen")
        assert zeva["task_context"]["mode"] == "static"
        assert zeva["memory"]["memory_semantics"] == "completed_effect_trace"
        assert zeva["memory"]["cross_attempt_only"] is False
        assert zeva["memory"]["pim_top_k"] == evaluation["pim_top_k"] == 4
        assert evaluation["attempt_protocol"] == "fixed_attempts" and evaluation["max_attempts"] == 6
        assert evaluation["pim_success_replay_enabled"] is True
        assert evaluation["cross_memory_enabled"] is (item["condition"] == "memory_on")


@pytest.mark.parametrize("extra", [["--protocol", "single_attempt", "--max-attempts", "4"], ["--success-replay"], ["--task-config", "robotwin_zeva_ego_pim_3cam_384"], ["--override", "model.zeva.adapter.effect_source=current_prediction"]])
def test_dry_run_rejects_mixed_main_ego_or_metric_contracts(tmp_path, extra):
    result = subprocess.run(_dry_command(tmp_path / "bad") + extra, text=True, capture_output=True)
    assert result.returncode != 0


def test_report_cli_requires_video_audit_for_complete(tmp_path):
    manifest, sha, path = _manifest(tmp_path, protocol="single_attempt", attempts=1)
    _, _, _, _, log = _run(tmp_path, manifest, sha, {7: [True]})
    command = [sys.executable, "-S", str(ROOT / "scripts/report_zeva_main_robotwin.py"), "--manifest", str(path), "--log", str(log)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and "video_path" in result.stderr
    result = subprocess.run(command + ["--metadata-only"], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["completion_status"] == "METADATA_ONLY"
