"""Run frozen-scene, four-attempt RoboTwin conditions with strict reporting.

Models load once per task and condition. A private CPU Torch generator is handed
between tasks assigned to the same slot; scene/attempt resets never reseed it.
--dry-run validates the manifest and writes the exact command plan without
importing torch/SAPIEN or loading assets. Every invocation needs a fresh directory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.robotwin.multiattempt import (
    CONTINUOUS_NOISE_PROTOCOL, EVALUATION_PROTOCOL_SCHEMA, FROZEN_MANIFEST_SHA256,
    LEGACY_SEED_PROTOCOL, REFERENCE_MANIFEST_SHA256, REFERENCE_REVISION,
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
    semantics = str(zeva.memory.get("memory_semantics", "completed_effect_trace"))
    prior = str(zeva.adapter.get("prior_variant", "legacy_bit"))
    if effect_source == "current_prediction":
        if (prefix_layout != "zeva_ego" or prior != "zeva_ego" or semantics != "predicted_effect_trace"
                or str(zeva.cte.get("variant", "legacy")) != "zeva_ego"
                or str(zeva.cte.input_type) != "rgb_multiview" or int(zeva.cte.get("executed_horizon", -1)) != 24):
            raise ValueError("current_prediction requires the full Zeva-Ego CTE/prior/memory contract")
    elif semantics == "predicted_effect_trace" or prior == "zeva_ego":
        raise ValueError("Zeva-Ego predictive memory/prior requires current_prediction")
    processor = cfg.data.train.processor
    boundary_observations = (
        effect_source == "current_prediction" and not processor.get("action_state_transforms")
        and not bool(processor.get("use_stepwise_action_norm", False))
    )
    return {
        "prefix_layout": str(zeva.adapter.prefix_layout),
        "effect_source": str(zeva.adapter.effect_source) if prefix_layout == "zeva_ego" else None,
        "pim_variant": variant,
        "pim_top_k": top_k,
        "memory_semantics": semantics,
        "prior_variant": prior,
        "skip_get_obs_within_replan": boundary_observations,
    }


def _verify_rng_handoff(
    path: str | Path, *, seed: int, slot: int, slots: int, run_id: str,
    completed_tasks: list[str], log_path: str | Path,
) -> None:
    """Check that a successful subprocess committed the exact slot/log tail."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("Task finished without committing its continuous model RNG state")
    payload = json.loads(path.read_text())
    expected = {
        "protocol": CONTINUOUS_NOISE_PROTOCOL, "seed": seed,
        "slot_index": slot, "slot_count": slots, "run_id": run_id,
        "device": "cpu", "dtype": "float32", "noise_shape": [1, 32, 14],
        "completed_tasks": completed_tasks,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Continuous model RNG task handoff identity/order differs from the plan")
    try:
        raw = base64.b64decode(payload["generator_state"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid model RNG state encoding") from exc
    if hashlib.sha256(raw).hexdigest() != payload.get("state_sha256"):
        raise ValueError("Continuous model RNG handoff checksum mismatch")
    rows = read_attempts(log_path)
    if not rows or rows[-1].get("task") != completed_tasks[-1]:
        raise ValueError("Continuous model RNG handoff has no matching task log tail")
    identity = rows[-1].get("checkpoint_identity")
    if not isinstance(identity, dict) or payload.get("checkpoint_identity") != identity:
        raise ValueError("Continuous model RNG handoff checkpoint bundle differs from the task log")
    snapshot = rows[-1].get("seed_stream")
    if not isinstance(snapshot, dict) or any(payload.get(key) != snapshot.get(key) for key in (
        "protocol", "seed", "draws", "state_sha256", "device", "dtype",
    )):
        raise ValueError("Continuous model RNG handoff differs from the final attempt snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "configs/robotwin_cross_attempt_10x20.json"))
    parser.add_argument("--expected-manifest-sha256", help="Required for custom source manifests; the packaged 10x20 manifest is pinned automatically")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--cte-checkpoint")
    parser.add_argument("--old-cte-checkpoint", help="Override shared --cte-checkpoint for the legacy condition")
    parser.add_argument("--new-cte-checkpoint", help="Override shared --cte-checkpoint for new_on/new_off")
    parser.add_argument("--old-addon-checkpoint")
    parser.add_argument("--new-addon-checkpoint")
    parser.add_argument("--task-context-bank")
    parser.add_argument("--task-context-retrieval-checkpoint")
    parser.add_argument("--old-task-context-bank")
    parser.add_argument("--new-task-context-bank")
    parser.add_argument("--old-task-context-retrieval-checkpoint")
    parser.add_argument("--new-task-context-retrieval-checkpoint")
    parser.add_argument("--task-context-mode", choices=("static", "bank", "pooling"), default="static")
    parser.add_argument("--task-context-top-k", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", nargs="+", choices=("base", "old", "new_off", "new_on"), default=["base", "new_off", "new_on"])
    parser.add_argument("--tasks", nargs="+", help="Create a separately hashed development subset")
    parser.add_argument("--model-seed", "--model-rng-seed", type=int, default=20260907)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--model-rng-protocol", choices=(CONTINUOUS_NOISE_PROTOCOL, LEGACY_SEED_PROTOCOL), default=CONTINUOUS_NOISE_PROTOCOL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--pim-diagnostic-interval", type=int, default=0)
    parser.add_argument("--robotwin-root", default=str(ROOT / "third_party/RoboTwin"))
    parser.add_argument("--old-task-config", default="robotwin_zeva_fastwam_cross_attempt_3cam_384")
    parser.add_argument("--new-task-config", default="robotwin_zeva_ego_pim_3cam_384")
    parser.add_argument("--old-prefix-layout", choices=("legacy_merged", "zeva_ego"), default="zeva_ego")
    parser.add_argument("--new-prefix-layout", choices=("legacy_merged", "zeva_ego"), default="zeva_ego")
    parser.add_argument("--old-effect-source", choices=("latest_completed", "current_prediction"), default="latest_completed")
    parser.add_argument("--new-effect-source", "--effect-source", choices=("latest_completed", "current_prediction"), default="current_prediction")
    parser.add_argument("--override", action="append", default=[], help="Extra non-protocol Hydra overrides, e.g. model.tokenizer_max_len=128; pretrained asset root uses DIFFSYNTH_MODEL_BASE_PATH")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.model_seed < 2**63 or args.slots < 1 or args.num_inference_steps < 1 or args.pim_diagnostic_interval < 0:
        parser.error("Invalid model seed, slot count, inference steps or diagnostic interval")
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("Conditions cannot be repeated")
    for override in args.override:
        key = override.split("=", 1)[0].lstrip("+")
        if key.startswith("EVALUATION.") or key in {"task", "ckpt", "gpu_id"}:
            parser.error("--override cannot change the frozen evaluation protocol")
        if key in {"model.zeva.adapter.prefix_layout", "model.zeva.adapter.effect_source"}:
            parser.error("Use --old/--new-prefix-layout and --old/--new-effect-source for prefix contracts")
    required = ["checkpoint", "dataset_stats"]
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
    condition_assets = {}
    for condition in args.conditions:
        if condition == "base":
            continue
        family = "new" if condition in {"new_on", "new_off"} else "old"
        names = ["cte_checkpoint"]
        if args.task_context_mode in {"bank", "static"}:
            names.append("task_context_bank")
        if args.task_context_mode == "static":
            names.append("task_context_retrieval_checkpoint")
        assets = {}
        for name in names:
            value = getattr(args, f"{family}_{name}") or getattr(args, name)
            if not value:
                parser.error(f"{condition} requires --{family}-{name.replace('_', '-')} or shared --{name.replace('_', '-')}")
            path = Path(value).expanduser().resolve()
            if not args.dry_run and not path.is_file():
                parser.error(f"Missing {condition} artifact: {path}")
            assets[name] = str(path)
        condition_assets[condition] = assets
    expected_sha = args.expected_manifest_sha256
    if Path(args.manifest).expanduser().resolve() == (ROOT / "configs/robotwin_cross_attempt_10x20.json").resolve():
        if expected_sha is not None and expected_sha != FROZEN_MANIFEST_SHA256:
            parser.error("The packaged 10x20 manifest must use its pinned SHA256")
        expected_sha = FROZEN_MANIFEST_SHA256
    elif expected_sha is None:
        parser.error("Custom manifests require --expected-manifest-sha256")
    source, source_sha = load_manifest(args.manifest, expected_sha256=expected_sha)
    manifest = dict(source)
    tasks = sorted({row["task"] for row in source["episodes"]})
    source_assignment = {task: index % args.slots for index, task in enumerate(tasks)}
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
                args.new_prefix_layout if new else args.old_prefix_layout,
                args.new_effect_source if new else args.old_effect_source,
            )
            if condition == "base":
                contracts[condition]["skip_get_obs_within_replan"] = True
        except ValueError as exc:
            parser.error(f"Invalid {condition} policy configuration: {exc}")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        parser.error("Output directory must be fresh and must not exist; partial runs cannot be resumed")
    assignment = {task: source_assignment[task] for task in tasks}
    task_order = [task for slot in range(args.slots) for task in tasks if assignment[task] == slot]
    protocol = {
        "schema": EVALUATION_PROTOCOL_SCHEMA,
        "source_manifest_sha256": source_sha,
        "manifest_scope": "development_subset" if args.tasks else ("frozen_full" if source_sha == FROZEN_MANIFEST_SHA256 else "custom"),
        "reference_repository": "air-embodied-brain/Zeva",
        "reference_revision": REFERENCE_REVISION,
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "model_rng_protocol": args.model_rng_protocol,
        "model_seed": args.model_seed, "slots": args.slots,
        "task_slot_assignment": assignment, "task_order": task_order,
        "slot_assignment_basis": "sorted source-manifest task index modulo slots",
        "num_inference_steps": args.num_inference_steps,
        "policy_contracts": contracts,
        "noise_implementation": "CPU float32; FastWAM joint14/horizon32, not Zeva CUDA samples",
    }
    manifest["evaluation_protocol"] = protocol
    output.mkdir(parents=True, exist_ok=False)
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    frozen_path = output / "manifest.json"
    frozen_path.write_bytes(manifest_bytes)
    manifest, manifest_sha = load_manifest(frozen_path)
    commands = []
    for condition in args.conditions:
        log = output / f"{condition}.jsonl"
        new = condition in {"new_off", "new_on"}
        mode = "base" if condition == "base" else "pim_on"
        addon = args.new_addon_checkpoint if new else args.old_addon_checkpoint
        contract = contracts[condition]
        for task in task_order:
            slot = assignment[task]
            state_path = output / "rng_state" / condition / f"slot_{slot}.json"
            rng_run_id = f"{output.name}:{condition}"
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
                setting("EVALUATION.expected_manifest_sha256", manifest_sha),
                setting("EVALUATION.attempt_log_path", str(log)),
                setting("EVALUATION.condition", condition),
                setting("EVALUATION.run_id", output.name),
                setting("EVALUATION.output_dir", str(output / condition)),
                setting("EVALUATION.results_dir", str(output / condition)),
                setting("EVALUATION.model_seed", args.model_seed),
                setting("EVALUATION.model_rng_protocol", args.model_rng_protocol),
                setting("EVALUATION.evaluation_slot_index", slot),
                setting("EVALUATION.evaluation_slot_count", args.slots),
                setting("EVALUATION.zeva_mode", mode),
                setting("EVALUATION.cross_memory_enabled", condition != "new_off"),
                setting("EVALUATION.pim_top_k", contract["pim_top_k"]),
                setting("EVALUATION.max_attempts", manifest["max_attempts"]),
                setting("EVALUATION.action_horizon", manifest["action_horizon"]),
                setting("EVALUATION.replan_steps", manifest["replan_steps"]),
                setting("EVALUATION.num_inference_steps", args.num_inference_steps),
                setting("EVALUATION.pim_diagnostic_interval", args.pim_diagnostic_interval),
                setting("EVALUATION.skip_get_obs_within_replan", contract["skip_get_obs_within_replan"]),
            ]
            if args.model_rng_protocol == CONTINUOUS_NOISE_PROTOCOL:
                command.extend([
                    setting("EVALUATION.model_rng_state_path", str(state_path)),
                    setting("EVALUATION.model_rng_task_name", task),
                    setting("EVALUATION.model_rng_run_id", rng_run_id),
                ])
            if condition != "base":
                assets = condition_assets[condition]
                command.extend([
                    setting("EVALUATION.cte_checkpoint", assets["cte_checkpoint"]),
                    setting("EVALUATION.addon_checkpoint", addon),
                    setting("model.zeva.task_context.mode", args.task_context_mode),
                    setting("model.zeva.task_context.top_k", args.task_context_top_k),
                ])
                if assets.get("task_context_bank"):
                    command.append(setting("model.zeva.task_context.bank_path", assets["task_context_bank"]))
                if assets.get("task_context_retrieval_checkpoint"):
                    command.append(setting("model.zeva.task_context.retrieval_checkpoint", assets["task_context_retrieval_checkpoint"]))
            command.extend(args.override)
            command.extend([
                setting("model.zeva.adapter.prefix_layout", contract["prefix_layout"]),
                setting("model.zeva.adapter.effect_source", args.new_effect_source if new else args.old_effect_source),
            ])
            commands.append({
                "condition": condition, "task": task, "slot": slot,
                "rng_state_path": str(state_path) if args.model_rng_protocol == CONTINUOUS_NOISE_PROTOCOL else None,
                "rng_run_id": rng_run_id, "command": command,
            })
    plan = {
        "manifest_sha256": manifest_sha,
        "episodes": len(manifest["episodes"]),
        "conditions": args.conditions,
        "max_total_attempts": len(manifest["episodes"]) * len(args.conditions) * manifest["max_attempts"],
        "source_manifest_sha256": source_sha,
        "evaluation_protocol": protocol,
        "model_rng_protocol": args.model_rng_protocol,
        "policy_contracts": contracts,
        "policy_artifacts": condition_assets,
        "artifact_validation": "skipped_dry_run" if args.dry_run else "paths_exist; runtime verifies checkpoint/cache identities",
        "commands": commands,
    }
    (output / "command_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    completed: dict[tuple[str, int], list[str]] = {}
    for item in commands:
        print(f"Evaluating {item['condition']} / {item['task']}", flush=True)
        result = subprocess.run(item["command"], cwd=ROOT, check=False)
        if result.returncode:
            failure = {"condition": item["condition"], "task": item["task"], "returncode": result.returncode}
            (output / "run_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise SystemExit(f"Evaluation failed: {failure}; partial logs are not a completed benchmark")
        if args.model_rng_protocol == CONTINUOUS_NOISE_PROTOCOL:
            completed_tasks = completed.setdefault((item["condition"], item["slot"]), [])
            completed_tasks.append(item["task"])
            _verify_rng_handoff(
                item["rng_state_path"], seed=args.model_seed, slot=item["slot"], slots=args.slots,
                run_id=item["rng_run_id"], completed_tasks=completed_tasks,
                log_path=output / f"{item['condition']}.jsonl",
            )
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
        "manifest_scope": protocol["manifest_scope"], "protocol_audit": "passed",
    }, indent=2) + "\n")
    print(f"Completed and validated {len(manifest['episodes'])} manifest episodes per condition: {output}")


if __name__ == "__main__":
    main()
