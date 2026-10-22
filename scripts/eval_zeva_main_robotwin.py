"""Evaluate Zeva-main observed-effect memory with explicit SR/CSR/stability protocols.

The simulator and model are loaded sequentially on one GPU. --dry-run requires
only Python's standard library and freezes the scene list and exact commands;
it does not claim that checkpoint compatibility or a rollout was validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.zeva_main_evaluation import (
    NOISE_PROTOCOL, PROFILE, PROTOCOLS, audit_attempt_videos, load_manifest,
    make_manifest, paired_comparison, read_attempts, summarize_attempts, validate_attempts,
)


DEFAULT_SOURCE = ROOT / "configs/robotwin_cross_attempt_10x20.json"
DEFAULT_SOURCE_SHA256 = "588a85f45dc961ccf3b4ecba3a1651eeb126c4a8368d8ac7f7485c3aa4e145cf"


def _setting(key: str, value) -> str:
    return f"{key}={json.dumps(value, ensure_ascii=True)}"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _verify_completed_task(*, log: Path, manifest: dict, manifest_sha: str, task: str) -> None:
    """Fail before launching another task if a child omitted or corrupted rows."""
    rows = read_attempts(log)
    if not rows or rows[-1].get("task") != task:
        raise ValueError("Completed child did not record the expected task")
    subset = {**manifest, "episodes": [scene for scene in manifest["episodes"] if scene["task"] == task]}
    validate_attempts(subset, manifest_sha, [row for row in rows if row.get("task") == task])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--source-manifest", default=str(DEFAULT_SOURCE),
                        help="Source scenes/instructions only; Ego model/protocol fields are never inherited")
    parser.add_argument("--expected-manifest-sha256", help="Required for a custom scene source")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="single_attempt")
    parser.add_argument("--max-attempts", type=int, help="Default 1 for single_attempt, 4 otherwise; fixed_attempts really runs all K")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cte-checkpoint", required=True)
    parser.add_argument("--addon-checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--success-manifest", required=True, help="Verified success-only offline training lineage, not evaluation outcomes")
    parser.add_argument("--task-context-bank", required=True)
    parser.add_argument("--task-context-retrieval-checkpoint", required=True)
    parser.add_argument("--task-context-top-k", type=int, default=5)
    parser.add_argument("--task-config", default="robotwin_zeva_main_pim_3cam_384")
    parser.add_argument("--foundation-mode", choices=("frozen", "joint"), default="frozen",
                        help="Training bundle identity; all inference weights stay fixed. joint is an explicit WAM experiment")
    parser.add_argument("--conditions", nargs="+", choices=("memory_off", "memory_on"), default=["memory_off", "memory_on"])
    parser.add_argument("--tasks", nargs="+", help="Explicit development subset, separately frozen and hashed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robotwin-root", default=str(ROOT / "third_party/RoboTwin"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=20260907)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--success-replay", action=argparse.BooleanOptionalAction, default=False,
                        help="Explicit fixed_attempts post-success stability experiment; unavailable before first success")
    parser.add_argument("--replay-phase-threshold", type=float, default=0.80)
    parser.add_argument("--replay-blend", type=float, default=1.0)
    parser.add_argument("--policy-text-encoder-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--policy-text-cache-size", type=int, default=16)
    parser.add_argument("--pim-diagnostic-interval", type=int, default=0)
    parser.add_argument("--override", action="append", default=[], help="Non-protocol model/processor overrides only")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("Conditions cannot be repeated")
    if args.gpu_id < 0 or args.task_context_top_k < 1 or args.policy_text_cache_size < 0 or args.pim_diagnostic_interval < 0:
        parser.error("Invalid GPU, retrieval top-k, cache size, or diagnostic interval")
    if not args.task_config.startswith("robotwin_zeva_main_"):
        parser.error("Main runner requires an explicit robotwin_zeva_main_* task; use the existing Ego runner for Ego")
    for override in args.override:
        key = override.split("=", 1)[0].lstrip("+")
        if "=" not in override or key in {"task", "ckpt", "gpu_id", "foundation_mode"} or key.startswith(("EVALUATION.", "model.zeva.", "foundation_mode_settings.")):
            parser.error("--override cannot change main lineage, memory, scene, foundation, or evaluation contracts")
    source_path = Path(args.manifest).expanduser().resolve()
    expected_sha = args.expected_manifest_sha256
    if source_path == DEFAULT_SOURCE.resolve():
        if expected_sha is not None and expected_sha != DEFAULT_SOURCE_SHA256:
            parser.error("Packaged source scene manifest must use its pinned SHA256")
        expected_sha = DEFAULT_SOURCE_SHA256
    elif expected_sha is None:
        parser.error("Custom scene manifests require --expected-manifest-sha256")
    raw = source_path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != expected_sha:
        parser.error("Source manifest SHA256 mismatch")
    attempts = args.max_attempts if args.max_attempts is not None else (1 if args.protocol == "single_attempt" else 4)
    try:
        manifest = make_manifest(
            json.loads(raw), source_sha256=source_sha, protocol=args.protocol, max_attempts=attempts,
            model_seed=args.model_seed, num_inference_steps=args.num_inference_steps,
            success_replay=args.success_replay, replay_phase_threshold=args.replay_phase_threshold,
            replay_blend=args.replay_blend, tasks=args.tasks,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    assets = {key: str(Path(getattr(args, key)).expanduser().resolve()) for key in (
        "checkpoint", "cte_checkpoint", "addon_checkpoint", "dataset_stats", "success_manifest",
        "task_context_bank", "task_context_retrieval_checkpoint",
    )}
    if not args.dry_run:
        for name, path in assets.items():
            if not Path(path).is_file():
                parser.error(f"Missing {name}: {path}")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        parser.error("Output directory must be fresh; partial or dry runs cannot be resumed as completed benchmarks")
    output.mkdir(parents=True, exist_ok=False)
    frozen_path = output / "manifest.json"
    _write_json(frozen_path, manifest)
    manifest, manifest_sha = load_manifest(frozen_path)
    tasks = sorted({scene["task"] for scene in manifest["episodes"]})
    commands = []
    for condition in args.conditions:
        for task in tasks:
            settings = {
                "task": args.task_config, "ckpt": assets["checkpoint"], "gpu_id": args.gpu_id,
                "++foundation_mode": args.foundation_mode,
                "EVALUATION.task_name": task, "EVALUATION.robotwin_root": str(Path(args.robotwin_root).expanduser().resolve()),
                "EVALUATION.task_config": manifest["task_config"], "EVALUATION.instruction_type": manifest["instruction_type"],
                "EVALUATION.dataset_stats_path": assets["dataset_stats"],
                "EVALUATION.cte_checkpoint": assets["cte_checkpoint"], "EVALUATION.addon_checkpoint": assets["addon_checkpoint"],
                "EVALUATION.zeva_mode": "pim_on", "EVALUATION.cross_memory_enabled": condition == "memory_on",
                "++EVALUATION.evaluation_profile": PROFILE, "++EVALUATION.attempt_protocol": args.protocol,
                "++EVALUATION.pim_success_replay_enabled": args.success_replay,
                "++EVALUATION.pim_success_replay_phase_threshold": args.replay_phase_threshold,
                "++EVALUATION.pim_success_replay_blend": args.replay_blend,
                "EVALUATION.max_attempts": attempts, "EVALUATION.action_horizon": 32, "EVALUATION.replan_steps": 24,
                "EVALUATION.pim_top_k": 4, "EVALUATION.skip_get_obs_within_replan": False,
                "EVALUATION.num_inference_steps": args.num_inference_steps,
                "EVALUATION.model_seed": args.model_seed, "EVALUATION.model_rng_protocol": NOISE_PROTOCOL,
                "EVALUATION.evaluation_slot_index": 0, "EVALUATION.evaluation_slot_count": 1,
                "EVALUATION.model_rng_state_path": None, "EVALUATION.model_rng_task_name": None,
                "EVALUATION.model_rng_run_id": None,
                "EVALUATION.episode_manifest": str(frozen_path), "EVALUATION.expected_manifest_sha256": manifest_sha,
                "EVALUATION.attempt_log_path": str(output / f"{condition}.jsonl"),
                "EVALUATION.condition": condition, "EVALUATION.run_id": output.name,
                "EVALUATION.output_dir": str(output / condition), "EVALUATION.results_dir": str(output / condition),
                "EVALUATION.policy_text_encoder_device": args.policy_text_encoder_device,
                "EVALUATION.policy_text_cache_size": args.policy_text_cache_size,
                "EVALUATION.pim_diagnostic_interval": args.pim_diagnostic_interval,
                "model.zeva.task_context.mode": "static", "model.zeva.task_context.top_k": args.task_context_top_k,
                "model.zeva.task_context.bank_path": assets["task_context_bank"],
                "model.zeva.task_context.retrieval_checkpoint": assets["task_context_retrieval_checkpoint"],
                "++model.zeva.offline.success_manifest_path": assets["success_manifest"],
            }
            command = [sys.executable, str(ROOT / "experiments/robotwin/eval_robotwin_single.py"),
                       "--config-name", "sim_robotwin_zeva_main", *(_setting(key, value) for key, value in settings.items()), *args.override]
            commands.append({"condition": condition, "task": task, "command": command})
    plan = {
        "evaluation_profile": PROFILE, "attempt_protocol": args.protocol, "manifest_sha256": manifest_sha,
        "source_manifest_sha256": source_sha, "episodes_per_condition": len(manifest["episodes"]),
        "max_attempts": attempts, "conditions": args.conditions, "policy_artifacts": assets,
        "foundation_training_mode": args.foundation_mode, "inference_parameter_updates": False,
        "pim_success_replay_enabled": args.success_replay, "model_rng_protocol": NOISE_PROTOCOL,
        "model_seed_formula": manifest["model_seed_formula"], "discarded_warmup_requests": 1,
        "discarded_warmup_scope": "policy_instance",
        "gpu_execution": "sequential conditions/tasks; one model resident; no guarantee of 32GB fit without measurement",
        "metric_contract": {"single_attempt": "independently reset randomized scenes: SR@1",
                            "stop_on_success": "same scene retries: CSR@k with all original scenes as denominator",
                            "fixed_attempts": "exactly K same-scene tries: CSR plus per-attempt and post-first-success stability"}[args.protocol],
        "artifact_validation": "not_checked_dry_run" if args.dry_run else "paths_checked; runtime must validate main CTE/addon/manifest lineage",
        "commands": commands,
    }
    _write_json(output / "command_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    for item in commands:
        print(f"Evaluating main {args.protocol}: {item['condition']} / {item['task']}", flush=True)
        result = subprocess.run(item["command"], cwd=ROOT, check=False)
        if result.returncode:
            _write_json(output / "run_failure.json", {"condition": item["condition"], "task": item["task"], "returncode": result.returncode})
            raise SystemExit("Rollout failed; partial logs are not a completed evaluation")
        _verify_completed_task(log=output / f"{item['condition']}.jsonl", manifest=manifest,
                               manifest_sha=manifest_sha, task=item["task"])
    rows = {condition: read_attempts(output / f"{condition}.jsonl") for condition in args.conditions}
    video_audit = audit_attempt_videos([row for condition_rows in rows.values() for row in condition_rows])
    for condition, condition_rows in rows.items():
        report = summarize_attempts(manifest, manifest_sha, condition_rows)
        report.update(completion_status="COMPLETE", video_audit=audit_attempt_videos(condition_rows))
        _write_json(output / f"{condition}.summary.json", report)
    if set(args.conditions) == {"memory_on", "memory_off"}:
        report = paired_comparison(manifest, manifest_sha, rows["memory_on"], rows["memory_off"])
        report.update(completion_status="COMPLETE", video_audit=video_audit)
        _write_json(output / "paired_memory.summary.json", report)
    _write_json(output / "COMPLETE", {"manifest_sha256": manifest_sha, "attempt_protocol": args.protocol,
                                    "conditions": args.conditions, "video_audit": video_audit})
    print(f"Completed main {args.protocol} evaluation: {output}")


if __name__ == "__main__":
    main()
