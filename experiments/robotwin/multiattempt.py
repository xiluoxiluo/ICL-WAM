"""Strict, dependency-free RoboTwin manifests and multi-attempt reporting.

The manifest is the denominator. Missing episodes, runtime errors and malformed
retry sequences are rejected instead of being silently counted as benchmark
failures or removed from the result.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "iclwam-robotwin-multiattempt-v1"
ATTEMPT_SCHEMA = "iclwam-robotwin-attempt-v1"
EVALUATION_PROTOCOL_SCHEMA = "iclwam-zeva-ego-evaluation-v2"
CONTINUOUS_NOISE_PROTOCOL = "torch_continuous_slot_v1"
LEGACY_SEED_PROTOCOL = "request_seed_stream_v1"
FROZEN_MANIFEST_SHA256 = "588a85f45dc961ccf3b4ecba3a1651eeb126c4a8368d8ac7f7485c3aa4e145cf"
REFERENCE_REVISION = "bae85ad753d3afe4f2e56762373d48ac0267e101"
REFERENCE_MANIFEST_SHA256 = "1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _protocol(manifest: dict[str, Any]) -> dict[str, Any] | None:
    protocol = manifest.get("evaluation_protocol")
    if protocol is None:
        return None  # Legacy manifests remain readable, with an explicit audit label.
    if not isinstance(protocol, dict) or protocol.get("schema") != EVALUATION_PROTOCOL_SCHEMA:
        raise ValueError("Unknown evaluation protocol schema")
    if protocol.get("model_rng_protocol") not in {CONTINUOUS_NOISE_PROTOCOL, LEGACY_SEED_PROTOCOL}:
        raise ValueError("Unknown model RNG protocol")
    slots = _integer(protocol.get("slots"), "slots", 1)
    if _integer(protocol.get("model_seed"), "protocol model_seed") >= 2**63:
        raise ValueError("protocol model_seed must be < 2**63")
    _integer(protocol.get("num_inference_steps"), "protocol num_inference_steps", 1)
    _sha256(protocol.get("source_manifest_sha256"), "source_manifest_sha256")
    if protocol.get("manifest_scope") not in {"frozen_full", "development_subset", "custom"}:
        raise ValueError("Unknown manifest scope")
    tasks = sorted({row["task"] for row in manifest["episodes"]})
    assignment = protocol.get("task_slot_assignment")
    if not isinstance(assignment, dict) or set(assignment) != set(tasks):
        raise ValueError("Slot assignment must cover exactly the manifest tasks")
    for slot in assignment.values():
        if _integer(slot, "task slot") >= slots:
            raise ValueError("Task assigned outside the available slots")
    order = [task for slot in range(slots) for task in tasks if assignment[task] == slot]
    if protocol.get("task_order") != order:
        raise ValueError("Task order must follow slot order and sorted tasks within each slot")
    if protocol["manifest_scope"] == "frozen_full":
        if (protocol["source_manifest_sha256"] != FROZEN_MANIFEST_SHA256
                or len(tasks) != 10 or len(manifest["episodes"]) != 200
                or any(sum(row["task"] == task for row in manifest["episodes"]) != 20 for task in tasks)):
            raise ValueError("frozen_full requires the pinned 10-task x 20-episode source manifest")
    return protocol


def load_manifest(path: str | Path, *, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _sha256(expected_sha256, "expected manifest SHA256"):
        raise ValueError(f"Manifest SHA256 mismatch: expected {expected_sha256}, got {digest}")
    manifest = json.loads(raw)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Expected manifest schema {MANIFEST_SCHEMA}")
    attempts = _integer(manifest.get("max_attempts"), "max_attempts", 1)
    if attempts > 4:
        raise ValueError("This protocol supports at most four attempts")
    if manifest.get("action_dim") != 14 or manifest.get("action_horizon") != 32:
        raise ValueError("Manifest must preserve FastWAM joint14 / horizon32")
    replan = _integer(manifest.get("replan_steps"), "replan_steps", 1)
    if replan > 32 or replan % 4:
        raise ValueError("replan_steps must be a multiple of four in [4,32]")
    if manifest.get("task_config") not in {"demo_clean", "demo_randomized"}:
        raise ValueError("Unsupported RoboTwin task_config")
    if manifest.get("instruction_type") not in {"seen", "unseen"}:
        raise ValueError("instruction_type must be seen or unseen")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Manifest episodes must be a nonempty list")
    identities, scenes = set(), set()
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("Each episode must be an object")
        task, identity = episode.get("task"), episode.get("episode_id")
        if not isinstance(task, str) or not re.fullmatch(r"[a-zA-Z0-9_]+", task):
            raise ValueError("Invalid task name")
        if not isinstance(identity, str) or not identity:
            raise ValueError("episode_id must be a nonempty string")
        seed = _integer(episode.get("seed"), "scene seed")
        instruction = episode.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Each manifest episode needs an exact instruction")
        if identity in identities or (task, seed) in scenes:
            raise ValueError("Duplicate episode_id or task/scene seed in manifest")
        identities.add(identity)
        scenes.add((task, seed))
    _protocol(manifest)
    return manifest, digest


def read_attempts(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL line {line_number} in {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on line {line_number}")
            rows.append(value)
    return rows


def append_attempt(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def audit_attempt_videos(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require one distinct, existing, nonempty recording per attempt.

    This is a filesystem integrity check; it does not decode the video or
    certify codec validity. Resolved paths prevent aliases from satisfying
    several attempts with the same file.
    """
    if not rows:
        raise ValueError("Cannot audit an empty attempt log")
    seen: set[Path] = set()
    total_bytes = 0
    for row in rows:
        value = row.get("video_path")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Every completed attempt requires a video_path")
        path = Path(value).expanduser().resolve()
        if path in seen:
            raise ValueError(f"Duplicate attempt video path: {path}")
        if not path.is_file():
            raise ValueError(f"Attempt video is missing: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"Attempt video is empty: {path}")
        seen.add(path)
        total_bytes += size
    return {
        "status": "passed",
        "check": "unique existing nonempty files; codec not decoded",
        "videos": len(seen),
        "total_bytes": total_bytes,
    }


def audit_protocol_trace(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify reset scopes and continuous model-noise consumption when declared.

    Legacy JSONL remains reportable but cannot retroactively establish an audit
    it never recorded. Slot snapshots establish continuity, not a claim that
    FastWAM CPU samples reproduce Zeva's CUDA samples bit for bit.
    """
    protocol = _protocol(manifest)
    if protocol is None:
        return {"status": "not_recorded_legacy_manifest"}
    expected_episode_order = [
        episode["episode_id"] for task in protocol["task_order"]
        for episode in manifest["episodes"] if episode["task"] == task
    ]
    observed_episode_order = []
    previous: dict[Any, dict[str, Any]] = {}
    task_limits: dict[str, int] = {}
    initial_state = None
    draws = 0
    for row in rows:
        if not observed_episode_order or observed_episode_order[-1] != row["episode_id"]:
            observed_episode_order.append(row["episode_id"])
        expected_scope = "episode" if row["attempt"] == 1 else "attempt"
        if row.get("reset_scope") != expected_scope:
            raise ValueError("Attempt reset_scope differs from the frozen episode/attempt protocol")
        if row["attempt"] == 1 and row.get("pim_entries_before") != 0:
            raise ValueError("Every new episode must start with empty PIM")
        slot = protocol["task_slot_assignment"][row["task"]]
        if row.get("evaluation_slot") != {"index": slot, "count": protocol["slots"]}:
            raise ValueError("Attempt evaluation_slot differs from the frozen slot assignment")
        if row["model_seed"] != protocol["model_seed"] or row["num_inference_steps"] != protocol["num_inference_steps"]:
            raise ValueError("Attempt sampling configuration differs from the frozen protocol")
        if task_limits.setdefault(row["task"], row["step_limit"]) != row["step_limit"]:
            raise ValueError("Step budget changed within a frozen task")
        hashes = row.get("checkpoint_identity")
        if not isinstance(hashes, dict):
            raise ValueError("Audited evaluation requires checkpoint identities")
        required = ("base_sha256", "stats_sha256")
        if row.get("zeva_mode") != "base":
            required += ("cte_sha256", "addon_sha256")
        for key in required:
            _sha256(hashes.get(key), key)
        condition = row.get("condition")
        contracts = protocol.get("policy_contracts", {})
        if contracts:
            if condition not in contracts:
                raise ValueError("Condition is not present in the frozen policy contracts")
            if row.get("zeva_mode") != ("base" if condition == "base" else "pim_on"):
                raise ValueError("Policy mode differs from its frozen condition")
            if row.get("cross_memory_enabled") is not (condition != "new_off"):
                raise ValueError("Memory control differs from its frozen condition")
            if row.get("skip_get_obs_within_replan") is not contracts[condition]["skip_get_obs_within_replan"]:
                raise ValueError("Observation schedule differs from its frozen condition")
            if condition != "base":
                for key in ("prefix_layout", "effect_source", "memory_semantics", "prior_variant"):
                    if row.get(key) != contracts[condition][key]:
                        raise ValueError(f"Runtime {key} differs from the frozen policy contract")
                for key in ("training_recipe", "task_context_mode"):
                    if key in contracts[condition] and row.get(key) != contracts[condition][key]:
                        raise ValueError(f"Runtime {key} differs from the frozen policy contract")
                if contracts[condition].get("training_recipe") == "ego_joint_warmup":
                    _sha256(hashes.get("foundation_sha256"), "joint foundation SHA256")
                    if hashes.get("training_recipe") != "ego_joint_warmup":
                        raise ValueError("Joint model identity does not declare its training recipe")
                    frozen_foundation = contracts[condition].get("foundation_sha256")
                    if frozen_foundation is not None and hashes["foundation_sha256"] != frozen_foundation:
                        raise ValueError("Loaded foundation differs from the frozen evaluation checkpoint bundle")
                if contracts[condition].get("task_context_mode") == "task_language":
                    task_identity = hashes.get("task_context")
                    if not isinstance(task_identity, dict) or task_identity.get("mode") != "task_language":
                        raise ValueError("Task-language evaluation requires a matching retrieval identity")
                    for key in ("bank_sha256", "head_sha256", "language_encoder_sha256"):
                        _sha256(task_identity.get(key), key)
                    if (task_identity.get("readout_kind") != "frozen_task_input_embedding_mean_v1"
                            or task_identity.get("query_scope") != "language_predicted_task"
                            or task_identity.get("temperature") != 1.0):
                        raise ValueError("Task-language retrieval contract differs from the frozen reference semantics")
        before, after = row.get("seed_stream_before"), row.get("seed_stream")
        for snapshot in (before, after):
            if not isinstance(snapshot, dict) or snapshot.get("protocol") != protocol["model_rng_protocol"]:
                raise ValueError("Missing or mismatched model RNG snapshot")
            if snapshot.get("seed") != protocol["model_seed"]:
                raise ValueError("Model RNG seed differs from the frozen protocol")
            _integer(snapshot.get("draws"), "model RNG draws")
            _sha256(snapshot.get("state_sha256"), "model RNG state SHA256")
            if protocol["model_rng_protocol"] == CONTINUOUS_NOISE_PROTOCOL:
                if snapshot.get("device") != "cpu" or snapshot.get("dtype") != "float32":
                    raise ValueError("Continuous FastWAM noise must use CPU float32")
        # Legacy task subprocesses restart their request-seed stream; this is
        # explicit rather than silently described as a continuous slot stream.
        stream_key = slot if protocol["model_rng_protocol"] == CONTINUOUS_NOISE_PROTOCOL else row["task"]
        if stream_key in previous:
            if before != previous[stream_key]:
                raise ValueError("Model RNG stream was reseeded, skipped or reordered between attempts/tasks")
        else:
            if before["draws"] != 0:
                raise ValueError("Each fresh model RNG stream must begin with zero draws")
            if initial_state is not None and before["state_sha256"] != initial_state:
                raise ValueError("Slots with the same seed must share their initial RNG state")
            initial_state = before["state_sha256"]
        consumed = after["draws"] - before["draws"]
        expected_draws = (row["step_count"] + row["replan_steps"] - 1) // row["replan_steps"]
        if consumed != expected_draws:
            raise ValueError("Model RNG draw count must equal executed replans; diagnostic probes must reuse noise")
        previous[stream_key] = after
        draws += consumed
    if observed_episode_order != expected_episode_order:
        raise ValueError("Episode order differs from the frozen slot/task/manifest order")
    return {
        "status": "passed", "model_rng_protocol": protocol["model_rng_protocol"],
        "slots": protocol["slots"], "active_streams": len(previous),
        "model_noise_draws": draws, "reset_scopes": "episode_then_attempt",
        "task_step_limits": task_limits,
    }


def validate_attempts(
    manifest: dict[str, Any],
    manifest_sha256: str,
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    expected = {row["episode_id"]: row for row in manifest["episodes"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_identity = None
    for row in rows:
        if row.get("schema") != ATTEMPT_SCHEMA:
            raise ValueError("Unknown attempt record schema")
        if row.get("manifest_sha256") != manifest_sha256:
            raise ValueError("Attempt log and frozen manifest differ")
        episode = expected.get(row.get("episode_id"))
        if episode is None:
            raise ValueError("Attempt contains an episode outside the manifest")
        for key, expected_value in (
            ("task", episode["task"]), ("scene_seed", episode["seed"]),
            ("instruction", episode["instruction"]),
            ("task_config", manifest["task_config"]),
            ("instruction_type", manifest["instruction_type"]),
            ("action_horizon", manifest["action_horizon"]),
            ("replan_steps", manifest["replan_steps"]),
            ("max_attempts", manifest["max_attempts"]),
        ):
            if row.get(key) != expected_value:
                raise ValueError(f"Attempt {key} differs from manifest")
        if type(row.get("success")) is not bool:
            raise ValueError("success must be a JSON boolean")
        if row.get("status") not in {"success", "timeout"}:
            raise ValueError("Runtime errors cannot be included in a completed benchmark")
        if (row["status"] == "success") != row["success"]:
            raise ValueError("Attempt success/status disagree")
        _integer(row.get("model_seed"), "model_seed")
        _integer(row.get("num_inference_steps"), "num_inference_steps", 1)
        _integer(row.get("step_count"), "step_count")
        _integer(row.get("step_limit"), "step_limit", 1)
        if row["step_count"] > row["step_limit"]:
            raise ValueError("Attempt exceeded its recorded action budget")
        if row["status"] == "timeout" and row["step_count"] != row["step_limit"]:
            raise ValueError("Incomplete timeout attempt did not exhaust its action budget")
        identity = (
            row.get("run_id"), row.get("condition"), row["model_seed"],
            row.get("zeva_mode"), row.get("cross_memory_enabled"),
            row["num_inference_steps"],
            json.dumps(row.get("checkpoint_identity"), sort_keys=True),
        )
        if run_identity is None:
            run_identity = identity
        elif identity != run_identity:
            raise ValueError("One log must contain one condition, checkpoint and model seed")
        attempts = grouped[row["episode_id"]]
        if attempts and row["step_limit"] != attempts[0]["step_limit"]:
            raise ValueError("Attempt step budget changed within an episode")
        index = _integer(row.get("attempt"), "attempt", 1)
        if index != len(attempts) + 1 or index > manifest["max_attempts"]:
            raise ValueError("Duplicate, out-of-order or out-of-budget attempt")
        if attempts and attempts[-1]["success"]:
            raise ValueError("Attempt recorded after this episode already succeeded")
        attempts.append(row)
    if set(grouped) != set(expected):
        missing = sorted(set(expected) - set(grouped))
        raise ValueError(f"Incomplete benchmark: missing {len(missing)} manifest episodes")
    for episode_id, attempts in grouped.items():
        if not attempts[-1]["success"] and len(attempts) != manifest["max_attempts"]:
            raise ValueError(f"Incomplete attempts for {episode_id}")
    audit_protocol_trace(manifest, rows)
    return dict(grouped)


def summarize_attempts(
    manifest: dict[str, Any], manifest_sha256: str, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped = validate_attempts(manifest, manifest_sha256, rows)
    n = len(manifest["episodes"])
    by_task: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for episode in manifest["episodes"]:
        by_task[episode["task"]].append(grouped[episode["episode_id"]])

    def rates(attempt_lists):
        return {
            f"SR@{k}": sum(any(row["success"] for row in tries[:k]) for tries in attempt_lists) / len(attempt_lists)
            for k in range(1, manifest["max_attempts"] + 1)
        }

    task_rates = {task: rates(tries) for task, tries in sorted(by_task.items())}
    failed_first = [tries for tries in grouped.values() if not tries[0]["success"]]
    protocol = _protocol(manifest)
    return {
        "schema": "iclwam-robotwin-report-v1",
        "manifest_sha256": manifest_sha256,
        "episodes": n,
        "attempts": len(rows),
        "condition": rows[0].get("condition"),
        "model_seed": rows[0]["model_seed"],
        "manifest_scope": protocol["manifest_scope"] if protocol else "legacy_unspecified",
        "evaluation_protocol": protocol,
        "protocol_audit": audit_protocol_trace(manifest, rows),
        "cumulative_success_rate": [
            {
                "attempt": k,
                "solved": sum(any(row["success"] for row in tries[:k]) for tries in grouped.values()),
                "total": n,
                "rate": sum(any(row["success"] for row in tries[:k]) for tries in grouped.values()) / n,
            }
            for k in range(1, manifest["max_attempts"] + 1)
        ],
        "micro": rates(list(grouped.values())),
        "macro": {key: sum(task[key] for task in task_rates.values()) / len(task_rates) for key in next(iter(task_rates.values()))},
        "per_task": task_rates,
        "rescue_after_first_failure": (
            sum(any(row["success"] for row in tries[1:]) for tries in failed_first) / len(failed_first)
            if failed_first else None
        ),
        "first_failure_episodes": len(failed_first),
        "mean_attempts_used": len(rows) / n,
        "executed_steps": sum(row["step_count"] for row in rows),
        "wall_time_seconds": sum(float(row.get("wall_time_seconds", 0.0)) for row in rows),
        "checkpoint_identity": rows[0].get("checkpoint_identity"),
    }


def paired_comparison(
    manifest: dict[str, Any], manifest_sha256: str,
    on_rows: list[dict[str, Any]], off_rows: list[dict[str, Any]],
    *, bootstrap_samples: int = 2000, bootstrap_seed: int = 0,
) -> dict[str, Any]:
    on = validate_attempts(manifest, manifest_sha256, on_rows)
    off = validate_attempts(manifest, manifest_sha256, off_rows)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    first_on, first_off = on_rows[0], off_rows[0]
    if first_on.get("cross_memory_enabled") is not True or first_off.get("cross_memory_enabled") is not False:
        raise ValueError("Paired PIM comparison requires explicit memory-on and memory-off logs")
    if first_on.get("zeva_mode") != "pim_on" or first_off.get("zeva_mode") != "pim_on":
        raise ValueError("Use pim_on mode for both weights-identical memory conditions; not pim_shadow")
    identity = first_on.get("checkpoint_identity")
    if not isinstance(identity, dict) or any(not identity.get(key) for key in ("base_sha256", "addon_sha256", "cte_sha256", "stats_sha256")):
        raise ValueError("PIM attribution requires base/addon/CTE/stats checkpoint hashes")
    for key in ("checkpoint_identity", "model_seed", "num_inference_steps"):
        if first_on.get(key) != first_off.get(key):
            raise ValueError(f"Paired on/off {key} differs")
    k = manifest["max_attempts"]
    task_deltas: dict[str, list[float]] = defaultdict(list)
    for episode in manifest["episodes"]:
        key = episode["episode_id"]
        if on[key][0]["step_limit"] != off[key][0]["step_limit"]:
            raise ValueError("Paired attempts have different step budgets")
        delta = int(any(row["success"] for row in on[key])) - int(any(row["success"] for row in off[key]))
        task_deltas[episode["task"]].append(float(delta))
    macro_delta = sum(sum(values) / len(values) for values in task_deltas.values()) / len(task_deltas)
    rng = random.Random(bootstrap_seed)
    draws = []
    for _ in range(bootstrap_samples):
        draws.append(sum(
            sum(rng.choice(values) for _ in values) / len(values)
            for values in task_deltas.values()
        ) / len(task_deltas))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    protocol = _protocol(manifest)
    return {
        "on": summarize_attempts(manifest, manifest_sha256, on_rows),
        "off": summarize_attempts(manifest, manifest_sha256, off_rows),
        f"paired_macro_delta_SR@{k}": macro_delta,
        "paired_95pct_bootstrap_interval": [low, high],
        "bootstrap_unit": "episode, paired within each fixed task",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "noise_pairing": (
            "same initial seed per slot; later noise offsets may diverge with trajectory lengths; "
            "local diagnostic probes reuse the exact action noise"
            if protocol and protocol["model_rng_protocol"] == CONTINUOUS_NOISE_PROTOCOL
            else "same recorded model seed; legacy logs do not establish exact per-step noise pairing"
        ),
    }
