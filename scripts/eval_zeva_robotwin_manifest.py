"""Run frozen-scene, four-attempt RoboTwin conditions with strict reporting.

Models load once per task and condition. Each process has a dedicated request
seed stream, independent of scene resets. --dry-run validates the manifest and
writes the exact command plan without importing torch/SAPIEN or loading assets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.robotwin.multiattempt import (
    audit_attempt_videos, load_manifest, paired_comparison, read_attempts, summarize_attempts,
)


def _policy_contract(task: str, overrides: list[str], prefix_layout: str, effect_source: str) -> dict:
    """Resolve reader capacity without importing or instantiating the policy."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="sim_robotwin_zeva", overrides=[
            f"task={task}", *overrides,
            f"model.zeva.adapter.prefix_layout={prefix_layout}",
            f"model.zeva.adapter.effect_source={effect_source}",
        ])
    zeva = cfg.model.zeva
    top_k = int(zeva.memory.pim_top_k)
    if top_k < 1 or top_k != int(zeva.prompt.persistent_length):
        raise ValueError("Evaluation memory.pim_top_k must match the configured reader persistent_length")
    variant = str(zeva.prompt.get("pim_variant", "legacy"))
    if (variant == "cross_attempt") != bool(zeva.memory.get("cross_attempt_only", False)):
        raise ValueError("Evaluation reader variant and cross_attempt_only configuration disagree")
    return {
        "prefix_layout": str(zeva.adapter.prefix_layout),
        "effect_source": str(zeva.adapter.effect_source) if prefix_layout == "zeva_ego" else None,
        "pim_variant": variant,
        "pim_top_k": top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "configs/robotwin_cross_attempt_10x20.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--cte-checkpoint")
    parser.add_argument("--old-addon-checkpoint")
    parser.add_argument("--new-addon-checkpoint")
    parser.add_argument("--task-context-bank")
    parser.add_argument("--task-context-retrieval-checkpoint")
    parser.add_argument("--task-context-mode", choices=("static", "bank", "pooling"), default="static")
    parser.add_argument("--task-context-top-k", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", nargs="+", choices=("base", "old", "new_off", "new_on"), default=["base", "new_off", "new_on"])
    parser.add_argument("--tasks", nargs="+", help="Create a separately hashed development subset")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--pim-diagnostic-interval", type=int, default=0)
    parser.add_argument("--robotwin-root", default=str(ROOT / "third_party/RoboTwin"))
    parser.add_argument("--old-task-config", default="robotwin_zeva_fastwam_pim_3cam_384")
    parser.add_argument("--new-task-config", default="robotwin_zeva_fastwam_cross_attempt_3cam_384")
    parser.add_argument("--old-prefix-layout", choices=("legacy_merged", "zeva_ego"), default="legacy_merged")
    parser.add_argument("--new-prefix-layout", choices=("legacy_merged", "zeva_ego"), default="zeva_ego")
    parser.add_argument("--effect-source", choices=("latest_completed",), default="latest_completed")
    parser.add_argument("--override", action="append", default=[], help="Extra non-protocol Hydra overrides, e.g. model.tokenizer_max_len=128; pretrained asset root uses DIFFSYNTH_MODEL_BASE_PATH")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.model_seed < 0 or args.num_inference_steps < 1 or args.pim_diagnostic_interval < 0:
        parser.error("Invalid model seed, inference steps or diagnostic interval")
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("Conditions cannot be repeated")
    for override in args.override:
        key = override.split("=", 1)[0].lstrip("+")
        if key.startswith("EVALUATION.") or key in {"task", "ckpt", "gpu_id"}:
            parser.error("--override cannot change the frozen evaluation protocol")
        if key in {"model.zeva.adapter.prefix_layout", "model.zeva.adapter.effect_source"}:
            parser.error("Use --old-prefix-layout/--new-prefix-layout and --effect-source for prefix contracts")
    required = ["checkpoint", "dataset_stats"]
    if any(condition != "base" for condition in args.conditions):
        required.append("cte_checkpoint")
        if args.task_context_mode in {"bank", "static"}:
            required.append("task_context_bank")
        if args.task_context_mode == "static":
            required.append("task_context_retrieval_checkpoint")
    if "old" in args.conditions:
        required.append("old_addon_checkpoint")
    if {"new_off", "new_on"}.intersection(args.conditions):
        required.append("new_addon_checkpoint")
    for name in required:
        value = getattr(args, name)
        if not value:
            parser.error(f"--{name.replace('_', '-')} is required for the selected conditions")
        path = Path(value).expanduser().resolve()
        if not args.dry_run and not path.is_file():
            parser.error(f"Missing artifact: {path}")
        setattr(args, name, str(path))
    source, source_sha = load_manifest(args.manifest)
    manifest = dict(source)
    tasks = sorted({row["task"] for row in source["episodes"]})
    if args.tasks:
        if len(args.tasks) != len(set(args.tasks)) or not set(args.tasks).issubset(tasks):
            parser.error("--tasks must be unique tasks contained in the manifest")
        manifest["episodes"] = [row for row in source["episodes"] if row["task"] in args.tasks]
        manifest["parent_manifest_sha256"] = source_sha
        tasks = sorted(args.tasks)
    contracts = {}
    for condition in args.conditions:
        new = condition in {"new_off", "new_on"}
        try:
            contracts[condition] = _policy_contract(
                args.new_task_config if new else args.old_task_config, args.override,
                args.new_prefix_layout if new else args.old_prefix_layout, args.effect_source,
            )
        except ValueError as exc:
            parser.error(f"Invalid {condition} policy configuration: {exc}")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE").exists():
        parser.error("Output directory already contains a completed run; use a new directory")
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    frozen_path = output / "manifest.json"
    if frozen_path.exists() and frozen_path.read_bytes() != manifest_bytes:
        parser.error("Output directory already contains a different frozen manifest")
    frozen_path.write_bytes(manifest_bytes)
    manifest, manifest_sha = load_manifest(frozen_path)
    commands = []
    for condition in args.conditions:
        log = output / f"{condition}.jsonl"
        if log.exists() and log.stat().st_size:
            parser.error(f"Refusing to append duplicate episodes to {log}; use a new output directory")
        new = condition in {"new_off", "new_on"}
        mode = "base" if condition == "base" else "pim_on"
        addon = args.new_addon_checkpoint if new else args.old_addon_checkpoint
        contract = contracts[condition]
        for task in tasks:
            def setting(key, value):
                return f"{key}={json.dumps(value, ensure_ascii=True)}"
            command = [
                sys.executable, str(ROOT / "experiments/robotwin/eval_robotwin_single.py"),
                "--config-name", "sim_robotwin_zeva",
                setting("task", args.new_task_config if new else args.old_task_config),
                setting("ckpt", args.checkpoint), setting("gpu_id", args.gpu_id),
                setting("EVALUATION.task_name", task),
                setting("EVALUATION.robotwin_root", str(Path(args.robotwin_root).expanduser().resolve())),
                setting("EVALUATION.dataset_stats_path", args.dataset_stats),
                setting("EVALUATION.task_config", manifest["task_config"]),
                setting("EVALUATION.instruction_type", manifest["instruction_type"]),
                setting("EVALUATION.episode_manifest", str(frozen_path)),
                setting("EVALUATION.attempt_log_path", str(log)),
                setting("EVALUATION.condition", condition),
                setting("EVALUATION.run_id", output.name),
                setting("EVALUATION.output_dir", str(output / condition)),
                setting("EVALUATION.results_dir", str(output / condition)),
                setting("EVALUATION.model_seed", args.model_seed),
                setting("EVALUATION.zeva_mode", mode),
                setting("EVALUATION.cross_memory_enabled", condition != "new_off"),
                setting("EVALUATION.pim_top_k", contract["pim_top_k"]),
                setting("EVALUATION.max_attempts", manifest["max_attempts"]),
                setting("EVALUATION.action_horizon", manifest["action_horizon"]),
                setting("EVALUATION.replan_steps", manifest["replan_steps"]),
                setting("EVALUATION.num_inference_steps", args.num_inference_steps),
                setting("EVALUATION.pim_diagnostic_interval", args.pim_diagnostic_interval),
                "EVALUATION.skip_get_obs_within_replan=false",
            ]
            if condition != "base":
                command.extend([
                    setting("EVALUATION.cte_checkpoint", args.cte_checkpoint),
                    setting("EVALUATION.addon_checkpoint", addon),
                    setting("model.zeva.task_context.mode", args.task_context_mode),
                    setting("model.zeva.task_context.top_k", args.task_context_top_k),
                ])
                if args.task_context_bank:
                    command.append(setting("model.zeva.task_context.bank_path", args.task_context_bank))
                if args.task_context_retrieval_checkpoint:
                    command.append(setting("model.zeva.task_context.retrieval_checkpoint", args.task_context_retrieval_checkpoint))
            command.extend(args.override)
            command.extend([
                setting("model.zeva.adapter.prefix_layout", contract["prefix_layout"]),
                setting("model.zeva.adapter.effect_source", args.effect_source),
            ])
            commands.append({"condition": condition, "task": task, "command": command})
    plan = {
        "manifest_sha256": manifest_sha,
        "episodes": len(manifest["episodes"]),
        "conditions": args.conditions,
        "max_total_attempts": len(manifest["episodes"]) * len(args.conditions) * manifest["max_attempts"],
        "model_rng_protocol": "request_seed_stream_v1; one independent stream per task/condition process",
        "policy_contracts": contracts,
        "artifact_validation": "skipped_dry_run" if args.dry_run else "paths_exist; runtime verifies checkpoint/cache identities",
        "commands": commands,
    }
    (output / "command_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for item in commands:
        print(f"Evaluating {item['condition']} / {item['task']}", flush=True)
        result = subprocess.run(item["command"], cwd=ROOT, check=False)
        if result.returncode:
            failure = {"condition": item["condition"], "task": item["task"], "returncode": result.returncode}
            (output / "run_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise SystemExit(f"Evaluation failed: {failure}; partial logs are not a completed benchmark")
    all_rows = {condition: read_attempts(output / f"{condition}.jsonl") for condition in args.conditions}
    # Audit across conditions too: an on/off pair must not reuse a recording.
    video_audit = audit_attempt_videos([row for rows in all_rows.values() for row in rows])
    for condition, rows in all_rows.items():
        report = summarize_attempts(manifest, manifest_sha, rows)
        report["video_audit"] = audit_attempt_videos(rows)
        report["completion_status"] = "COMPLETE"
        (output / f"{condition}.summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if {"new_on", "new_off"}.issubset(args.conditions):
        report = paired_comparison(manifest, manifest_sha, all_rows["new_on"], all_rows["new_off"])
        report["video_audit"] = audit_attempt_videos(all_rows["new_on"] + all_rows["new_off"])
        report["completion_status"] = "COMPLETE"
        (output / "paired_pim.summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "COMPLETE").write_text(json.dumps({
        "manifest_sha256": manifest_sha, "conditions": args.conditions,
        "episodes_per_condition": len(manifest["episodes"]), "video_audit": video_audit,
    }, indent=2) + "\n")
    print(f"Completed and validated {len(manifest['episodes'])} manifest episodes per condition: {output}")


if __name__ == "__main__":
    main()
