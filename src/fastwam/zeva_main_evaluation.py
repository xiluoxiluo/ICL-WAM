"""Strict, CPU-only reporting for Zeva-main observed-effect RoboTwin runs.

This is a FastWAM/RoboTwin adaptation of Zeva main, not the Zeva-Ego 10x20
benchmark contract. Scene manifests can reuse published randomized scenes,
but the metrics and runtime profiles are deliberately separate. No Torch,
Hydra, checkpoint, or simulator import is needed to validate a plan or report.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Any


PROFILE = "main_observed"
MANIFEST_SCHEMA = "iclwam-zeva-main-robotwin-manifest-v1"
ATTEMPT_SCHEMA = "iclwam-zeva-main-robotwin-attempt-v1"
REPORT_SCHEMA = "iclwam-zeva-main-robotwin-report-v1"
PROTOCOLS = ("single_attempt", "stop_on_success", "fixed_attempts")
NOISE_PROTOCOL = "torch_device_continuous_slot_v1"
REFERENCE_REVISION = "17ca9fa"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("evaluation_profile") != PROFILE:
        raise ValueError("Expected an independent Zeva-main observed-effect manifest, not an Ego evaluation manifest")
    protocol = manifest.get("attempt_protocol")
    if protocol not in PROTOCOLS:
        raise ValueError(f"attempt_protocol must be one of {PROTOCOLS}")
    attempts = _integer(manifest.get("max_attempts"), "max_attempts", 1)
    if protocol == "single_attempt" and attempts != 1:
        raise ValueError("single_attempt requires exactly one attempt per independently reset scene")
    if manifest.get("memory_semantics") != "completed_effect_trace":
        raise ValueError("Zeva-main evaluation requires completed observed-effect memory")
    if (manifest.get("action_dim"), manifest.get("action_horizon"), manifest.get("replan_steps")) != (14, 32, 24):
        raise ValueError("This FastWAM adaptation requires joint14, horizon32, execute24")
    for name in ("task_config", "instruction_type"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"Missing manifest {name}")
    if manifest.get("model_rng_protocol") != NOISE_PROTOCOL:
        raise ValueError("Main evaluator requires a continuous private policy-device model RNG")
    if _integer(manifest.get("model_seed"), "model_seed") >= 2**63:
        raise ValueError("model_seed must be < 2**63")
    _integer(manifest.get("num_inference_steps"), "num_inference_steps", 1)
    replay = _boolean(manifest.get("pim_success_replay_enabled"), "pim_success_replay_enabled")
    if replay and protocol != "fixed_attempts":
        raise ValueError("Success replay is only evaluated explicitly with fixed_attempts")
    for name, low in (("pim_success_replay_phase_threshold", -1.0), ("pim_success_replay_blend", 0.0)):
        value = manifest.get(name)
        if type(value) not in (int, float) or not low <= value <= 1.0:
            raise ValueError(f"{name} must be in [{low}, 1]")
    scenes = manifest.get("episodes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Manifest must contain a nonempty episodes list")
    ids, task_seeds = set(), set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("Every manifest episode must be an object")
        for name in ("episode_id", "task", "instruction"):
            if not isinstance(scene.get(name), str) or not scene[name].strip():
                raise ValueError(f"Episode is missing {name}")
        seed = _integer(scene.get("seed"), "scene seed")
        if scene["episode_id"] in ids or (scene["task"], seed) in task_seeds:
            raise ValueError("Duplicate episode ID or task/scene seed")
        ids.add(scene["episode_id"])
        task_seeds.add((scene["task"], seed))


def make_manifest(source: dict, *, source_sha256: str, protocol: str,
                  max_attempts: int, model_seed: int = 20260907,
                  num_inference_steps: int = 10, success_replay: bool = False,
                  replay_phase_threshold: float = 0.80, replay_blend: float = 1.0,
                  tasks: list[str] | None = None) -> dict:
    """Copy scenes only; never inherit Ego model, memory, or metric contracts."""
    _digest(source_sha256, "source manifest SHA256")
    if not isinstance(source, dict):
        raise ValueError("Source manifest must be a JSON object")
    episodes = source.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Source manifest must contain explicit seeded episodes and instructions")
    if tasks is not None:
        available = {row.get("task") for row in episodes}
        if not tasks or len(tasks) != len(set(tasks)) or not set(tasks) <= available:
            raise ValueError("Selected tasks must be unique tasks present in the source manifest")
        episodes = [row for row in episodes if row.get("task") in tasks]
    # Only scene selection crosses the contracts. Model and RNG settings below
    # belong to this new run even when the source is the public Ego 10x20 list.
    manifest = {
        "schema": MANIFEST_SCHEMA, "evaluation_profile": PROFILE,
        "attempt_protocol": protocol, "max_attempts": max_attempts,
        "memory_semantics": "completed_effect_trace",
        "action_dim": 14, "action_horizon": 32, "replan_steps": 24,
        "task_config": source.get("task_config", "demo_randomized"),
        "instruction_type": source.get("instruction_type", "seen"),
        "episodes": [{key: deepcopy(row.get(key)) for key in ("episode_id", "task", "seed", "instruction")} for row in episodes],
        "source_manifest_sha256": source_sha256,
        "source_manifest_schema": source.get("schema"),
        "scene_scope": "development_subset" if tasks else "frozen_source_scenes",
        "reference_repository": "air-embodied-brain/Zeva", "reference_revision": REFERENCE_REVISION,
        "reference_benchmark_reproduction": False,
        "model_rng_protocol": NOISE_PROTOCOL, "model_seed": model_seed,
        "num_inference_steps": num_inference_steps,
        "pim_success_replay_enabled": success_replay,
        "pim_success_replay_phase_threshold": replay_phase_threshold,
        "pim_success_replay_blend": replay_blend,
    }
    validate_manifest(manifest)
    return manifest


def load_manifest(path: str | Path, *, expected_sha256: str | None = None) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("Main manifest SHA256 differs from the frozen command plan")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("Main manifest must be a JSON object")
    validate_manifest(manifest)
    return manifest, sha


def read_attempts(path: str | Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Attempt log line {line_no} is not an object")
            rows.append(row)
    return rows


def append_attempt(path: str | Path, row: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _noise_snapshot(snapshot: Any, manifest: dict) -> dict:
    if not isinstance(snapshot, dict):
        raise ValueError("Each attempt must log its actual model RNG snapshots")
    for key, expected in (("protocol", manifest["model_rng_protocol"]), ("seed", manifest["model_seed"]), ("dtype", "float32")):
        if snapshot.get(key) != expected:
            raise ValueError(f"Model RNG {key} differs from manifest")
    _integer(snapshot.get("draws"), "model RNG draw count")
    _digest(snapshot.get("state_sha256"), "model RNG state")
    if not isinstance(snapshot.get("device"), str) or not snapshot["device"]:
        raise ValueError("Model RNG must record its device")
    return snapshot


def validate_attempts(manifest: dict, sha: str, rows: list[dict]) -> dict[str, list[dict]]:
    """Fail closed on partial runs, resets, mixed weights, replay leakage, or Ego logs."""
    validate_manifest(manifest)
    _digest(sha, "manifest SHA256")
    if not rows:
        raise ValueError("Empty attempt log is not a completed benchmark")
    scenes = {scene["episode_id"]: scene for scene in manifest["episodes"]}
    grouped: dict[str, list[dict]] = defaultdict(list)
    identity, enabled, condition, run_id = (rows[0].get(key) for key in ("checkpoint_identity", "cross_memory_enabled", "condition", "run_id"))
    if not isinstance(identity, dict):
        raise ValueError("Missing actual checkpoint_identity")
    for key in ("base_sha256", "addon_sha256", "cte_sha256", "stats_sha256"):
        _digest(identity.get(key), f"checkpoint_identity.{key}")
    _boolean(enabled, "cross_memory_enabled")
    if not isinstance(condition, str) or not condition or not isinstance(run_id, str) or not run_id:
        raise ValueError("Each log needs one condition and run_id")
    previous_noise = None
    closed_scenes: set[str] = set()
    previous_scene = None
    for row in rows:
        scene_id = row.get("episode_id")
        if scene_id not in scenes:
            raise ValueError("Attempt refers to an unselected scene")
        if scene_id != previous_scene:
            if scene_id in closed_scenes:
                raise ValueError("Scene attempts must be contiguous in the log")
            if previous_scene is not None:
                closed_scenes.add(previous_scene)
            previous_scene = scene_id
        scene = scenes[scene_id]
        expected = {
            "schema": ATTEMPT_SCHEMA, "manifest_sha256": sha,
            "evaluation_profile": PROFILE, "memory_semantics": "completed_effect_trace",
            "attempt_protocol": manifest["attempt_protocol"], "max_attempts": manifest["max_attempts"],
            "task": scene["task"], "scene_seed": scene["seed"], "instruction": scene["instruction"],
            "task_config": manifest["task_config"], "instruction_type": manifest["instruction_type"],
            "action_horizon": 32, "replan_steps": 24, "num_inference_steps": manifest["num_inference_steps"],
            "model_seed": manifest["model_seed"], "checkpoint_identity": identity,
            "cross_memory_enabled": enabled, "condition": condition, "run_id": run_id,
            "zeva_mode": "pim_on", "pim_success_replay_enabled": manifest["pim_success_replay_enabled"],
            "pim_success_replay_phase_threshold": manifest["pim_success_replay_phase_threshold"],
            "pim_success_replay_blend": manifest["pim_success_replay_blend"],
            "inference_parameters_frozen": True, "observed_effect_source": "observed_post",
            "scene_reset": True, "model_rng_reset": False,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"Attempt {scene_id} {key} differs from the main evaluation contract")
        success = _boolean(row.get("success"), "success")
        steps = _integer(row.get("step_count"), "step_count", 1)
        limit = _integer(row.get("step_limit"), "step_limit", 1)
        if steps > limit or row.get("status") != ("success" if success else "timeout") or row.get("error"):
            raise ValueError("Runtime error or inconsistent outcome is not a valid completed attempt")
        if not success and steps != limit:
            raise ValueError("Timeout must exhaust the actual environment step budget")
        observed = row.get("main_observed_runtime")
        if not isinstance(observed, dict) or any(observed.get(key) != value for key, value in {
            "memory_semantics": "completed_effect_trace", "effect_source": "observed_post",
            "executed_steps": steps, "observed_steps": steps - steps % 4,
            "terminal_observation_step": steps, "terminal_observed": True, "finalized": True,
        }.items()):
            raise ValueError("Completed attempts must include fresh terminal feedback and all observed four-action boundaries")
        before = _noise_snapshot(row.get("seed_stream_before"), manifest)
        after = _noise_snapshot(row.get("seed_stream"), manifest)
        if after["device"] != before["device"] or after["draws"] - before["draws"] != _integer(row.get("replans"), "replans", 1):
            raise ValueError("Model RNG must advance exactly once per policy replan, including replay")
        if previous_noise is not None and before != previous_noise:
            raise ValueError("Model RNG was reset or changed between attempts/scenes/tasks")
        previous_noise = after
        grouped[scene_id].append(row)
    if set(grouped) != set(scenes):
        raise ValueError("Partial evaluation: every selected scene must finish")
    for scene_id, attempts in grouped.items():
        first_success = None
        step_limit = attempts[0]["step_limit"]
        for index, row in enumerate(attempts, 1):
            if row.get("attempt") != index:
                raise ValueError("Attempt numbers must be consecutive and one-based")
            if index > manifest["max_attempts"] or row["step_limit"] != step_limit:
                raise ValueError("Attempt count or scene step budget changed")
            if row.get("reset_scope") != ("episode" if index == 1 else "attempt") or row.get("memory_reset") is not (index == 1):
                raise ValueError("Clear episode memory once, then preserve it on same-scene attempt resets")
            _integer(row.get("pim_entries_before"), "pim_entries_before")
            _integer(row.get("pim_entries_after"), "pim_entries_after")
            if index == 1 and row["pim_entries_before"] != 0:
                raise ValueError("Independent scenes must start with empty PIM")
            if index > 1 and row["pim_entries_before"] != attempts[index - 2]["pim_entries_after"]:
                raise ValueError("An attempt reset changed the completed-effect store")
            if row.get("first_success_attempt_before") != first_success or row.get("post_success_attempt") is not (first_success is not None):
                raise ValueError("First-success history is inconsistent")
            replayed = _integer(row.get("pim_success_replay_applied_count"), "replay chunk count")
            replayed_actions = _integer(row.get("pim_success_replay_action_count"), "replay action count")
            eligible = _boolean(row.get("pim_success_replay_eligible"), "replay eligibility")
            if (replayed == 0) != (replayed_actions == 0) or replayed_actions > row["step_count"]:
                raise ValueError("Replay action/chunk counts are inconsistent")
            if first_success is None and (eligible or replayed):
                raise ValueError("Success replay cannot be eligible or used before the first actual success")
            if replayed and (not eligible or not enabled or not manifest["pim_success_replay_enabled"]):
                raise ValueError("Replay used while unavailable or disabled")
            if row.get("pim_success_replay_source_attempts") != ([first_success - 1] if replayed else []):
                raise ValueError("Replay must use only the first confirmed successful attempt (zero-based runtime index)")
            if first_success is not None and manifest["attempt_protocol"] != "fixed_attempts":
                raise ValueError("Only fixed_attempts may continue after success")
            if first_success is None and row["success"]:
                first_success = index
            if row.get("first_success_attempt") != first_success:
                raise ValueError("Incorrect first_success_attempt marker")
        required = (first_success or manifest["max_attempts"]) if manifest["attempt_protocol"] == "stop_on_success" else manifest["max_attempts"]
        if len(attempts) != required:
            raise ValueError(f"Incomplete {manifest['attempt_protocol']} sequence for {scene_id}")
    return dict(grouped)


def _rate(successes: int, denominator: int) -> float | None:
    return successes / denominator if denominator else None


def summarize_attempts(manifest: dict, sha: str, rows: list[dict]) -> dict:
    grouped = validate_attempts(manifest, sha, rows)
    total = len(grouped)
    k = manifest["max_attempts"]
    first = {key: next((row["attempt"] for row in values if row["success"]), None) for key, values in grouped.items()}
    by_task: dict[str, list[str]] = defaultdict(list)
    for scene in manifest["episodes"]:
        by_task[scene["task"]].append(scene["episode_id"])
    per_task = {}
    for task, ids in by_task.items():
        per_task[task] = {
            "episodes": len(ids), "SR@1": sum(first[i] == 1 for i in ids) / len(ids),
            "CSR": {f"CSR@{n}": sum(first[i] is not None and first[i] <= n for i in ids) / len(ids) for n in range(1, k + 1)},
        }
    cumulative = {f"CSR@{n}": sum(v is not None and v <= n for v in first.values()) / total for n in range(1, k + 1)}
    post_success = [row for row in rows if row["post_success_attempt"]]
    stabilized_scenes = [values for values in grouped.values() if any(row["post_success_attempt"] for row in values)]
    first_failures = [key for key, values in grouped.items() if not values[0]["success"]]
    replay_post = [row for row in post_success if row["pim_success_replay_applied_count"]]
    metrics = {
        "SR@1": sum(v == 1 for v in first.values()) / total,
        "CSR": cumulative,
        "attempt_success": {
            str(n): {"successes": sum(row["success"] for row in rows if row["attempt"] == n),
                     "attempted_scenes": sum(row["attempt"] == n for row in rows),
                     "rate": _rate(sum(row["success"] for row in rows if row["attempt"] == n), sum(row["attempt"] == n for row in rows))}
            for n in range(1, k + 1)
        },
        "first_failure_recovery": {
            "first_failure_scenes": len(first_failures),
            "recovered_scenes": sum(first[key] is not None for key in first_failures),
            "rate": _rate(sum(first[key] is not None for key in first_failures), len(first_failures)),
            "meaning": "Descriptive recovery among this condition's first failures; not a shared failure-cohort causal contrast.",
        },
        "post_first_success_stability": {
            "status": "measured" if manifest["attempt_protocol"] == "fixed_attempts" else "not_measured_stop_after_success",
            "post_success_attempts": len(post_success),
            "post_success_successes": sum(row["success"] for row in post_success),
            "attempt_rate": _rate(sum(row["success"] for row in post_success), len(post_success)),
            "eligible_scenes": len(stabilized_scenes),
            "all_remaining_successful_scenes": sum(all(row["success"] for row in values if row["post_success_attempt"]) for values in stabilized_scenes),
            "all_remaining_success_rate": _rate(sum(all(row["success"] for row in values if row["post_success_attempt"]) for values in stabilized_scenes), len(stabilized_scenes)),
            "attempts_with_replay": len(replay_post),
            "replay_attempt_success_rate": _rate(sum(row["success"] for row in replay_post), len(replay_post)),
            "interpretation": "Conditional post-success stability, including success replay when enabled; never pure failure-learning performance.",
        },
    }
    return {
        "schema": REPORT_SCHEMA, "evaluation_profile": PROFILE, "attempt_protocol": manifest["attempt_protocol"],
        "manifest_sha256": sha, "episodes": total, "attempts": len(rows), "max_attempts": k,
        "condition": rows[0]["condition"], "cross_memory_enabled": rows[0]["cross_memory_enabled"],
        "checkpoint_identity": rows[0]["checkpoint_identity"],
        "pim_success_replay_enabled": manifest["pim_success_replay_enabled"],
        "inference_parameters_frozen": True, "observed_effect_source": "observed_post",
        "primary_metric": ("SR@1" if manifest["attempt_protocol"] == "single_attempt" else
                           "post_first_success_stability" if manifest["attempt_protocol"] == "fixed_attempts" else f"CSR@{k}"),
        "micro": metrics,
        "macro": {"SR@1": sum(value["SR@1"] for value in per_task.values()) / len(per_task),
                  "CSR": {f"CSR@{n}": sum(value["CSR"][f"CSR@{n}"] for value in per_task.values()) / len(per_task) for n in range(1, k + 1)}},
        "per_task": per_task,
        "reset_audit": {"episode_resets": total, "attempt_resets": len(rows) - total, "same_scene_retries": True, "memory_cleared_between_scenes": True, "model_rng_continuous": True},
        "limitations": [
            "CSR@k is the probability of any success within k tries, not independent single-attempt SR.",
            "Resampling alone can increase CSR; same-weight memory on/off is required to assess memory benefit.",
            "Main memory also updates within an attempt; on/off gain alone does not isolate failed-attempt memory.",
            "Later per-attempt rates under stop_on_success use only surviving failures; fixed_attempts keeps the original scene denominator.",
            "Identical initial model seeds do not guarantee matching later noise offsets after trajectory lengths diverge.",
            "This is an ICL-WAM RoboTwin adaptation, not reproduction of Zeva main's RoboCasa or Zeva-Ego's published scores.",
        ],
    }


def paired_comparison(manifest: dict, sha: str, on: list[dict], off: list[dict], *,
                      bootstrap_samples: int = 2000, bootstrap_seed: int = 0) -> dict:
    on_summary, off_summary = summarize_attempts(manifest, sha, on), summarize_attempts(manifest, sha, off)
    if not on[0]["cross_memory_enabled"] or off[0]["cross_memory_enabled"]:
        raise ValueError("Paired inputs must be memory-on and memory-off")
    if on[0]["checkpoint_identity"] != off[0]["checkpoint_identity"]:
        raise ValueError("Paired memory comparison requires exactly the same checkpoint_identity")
    on_group = {key: [row for row in on if row["episode_id"] == key] for key in (e["episode_id"] for e in manifest["episodes"])}
    off_group = {key: [row for row in off if row["episode_id"] == key] for key in on_group}
    for key in on_group:
        if on_group[key][0]["step_limit"] != off_group[key][0]["step_limit"]:
            raise ValueError("Paired scene step budgets differ")
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    delta_first = [int(on_group[key][0]["success"]) - int(off_group[key][0]["success"]) for key in on_group]
    delta_any = [int(any(row["success"] for row in on_group[key])) - int(any(row["success"] for row in off_group[key])) for key in on_group]
    rng = random.Random(bootstrap_seed)
    draws = sorted(sum(rng.choice(delta_any) for _ in delta_any) / len(delta_any) for _ in range(bootstrap_samples))
    k = manifest["max_attempts"]
    return {
        "schema": REPORT_SCHEMA, "evaluation_profile": PROFILE, "manifest_sha256": sha,
        "attempt_protocol": manifest["attempt_protocol"], "episodes": len(delta_any),
        "comparison": "same_weights_memory_on_vs_off",
        "paired_delta_SR@1": sum(delta_first) / len(delta_first),
        f"paired_delta_CSR@{k}": sum(delta_any) / len(delta_any),
        f"paired_delta_CSR@{k}_bootstrap_95pct": [draws[int(.025 * (bootstrap_samples - 1))], draws[int(.975 * (bootstrap_samples - 1))]],
        "bootstrap_unit": "paired scene; micro average over the frozen manifest",
        "same_initial_noise_state_scenes": sum(on_group[key][0]["seed_stream_before"] == off_group[key][0]["seed_stream_before"] for key in on_group),
        "success_replay_contaminates_pre_first_success": False,
        "pure_failed_attempt_memory_effect_identified": False,
        "interpretation": "SR/CSR contrasts include in-attempt observed-effect conditioning. Fixed-attempt stability can additionally include success replay. A shared first-failure trace and matched retry noise are needed to isolate failed-memory content.",
        "on": on_summary, "off": off_summary,
    }


def audit_attempt_videos(rows: list[dict]) -> dict:
    paths, total_bytes = set(), 0
    for row in rows:
        if not row.get("video_path"):
            raise ValueError("Completed evaluations require a video_path per attempt")
        path = Path(row["video_path"]).resolve()
        if path in paths:
            raise ValueError("Duplicate per-attempt video")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty per-attempt video: {path}")
        paths.add(path)
        total_bytes += path.stat().st_size
    return {"status": "passed", "videos": len(paths), "total_bytes": total_bytes, "validation": "file presence and nonempty bytes; codec not decoded"}
