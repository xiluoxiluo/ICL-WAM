"""Fixed-seed repeated-attempt evaluation entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cte-checkpoint")
    parser.add_argument("--addon-checkpoint")
    parser.add_argument("--task-context-bank")
    parser.add_argument("--task-context-retrieval-checkpoint")
    parser.add_argument("--task-language-encoder")
    parser.add_argument("--task-context-mode", choices=("task_language", "static", "bank", "pooling"))
    parser.add_argument("--task-context-top-k", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-seed", "--model-rng-seed", type=int, default=20260907)
    parser.add_argument("--model-rng-protocol", choices=("torch_device_continuous_slot_v1", "torch_continuous_slot_v1", "request_seed_stream_v1"), default="torch_device_continuous_slot_v1")
    parser.add_argument("--instruction", help="Exact instruction retained for every attempt")
    parser.add_argument("--sim-task", default="robotwin_zeva_ego_pim_3cam_384")
    parser.add_argument(
        "--prefix-layout", choices=("legacy_merged", "zeva_ego"),
        help="Override the task's prefix layout; old merged checkpoints require legacy_merged",
    )
    parser.add_argument("--effect-source", choices=("latest_completed", "current_prediction"))
    parser.add_argument("--dataset-stats")
    parser.add_argument("--cross-memory-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pim-diagnostic-interval", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("base", "zeva_stage2", "pim_shadow", "pim_on"),
        default="pim_on",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.model_seed < 2**63 or not 1 <= args.max_attempts <= 4:
        parser.error("model seed must be in [0,2**63) and max attempts in [1,4]")
    if args.mode != "base" and not args.cte_checkpoint:
        parser.error("--cte-checkpoint is required for non-base Zeva modes")
    if args.mode in {"zeva_stage2", "pim_on"} and not args.addon_checkpoint:
        parser.error("--addon-checkpoint is required for zeva_stage2 and pim_on")
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "experiments/robotwin/eval_robotwin_single.py"),
           "--config-name", "sim_robotwin_zeva.yaml", f"ckpt={args.ckpt}",
           f"EVALUATION.task_name={args.task}", f"EVALUATION.fixed_seed={args.seed}",
           f"EVALUATION.max_attempts={args.max_attempts}", f"EVALUATION.zeva_mode={args.mode}",
           f"EVALUATION.cte_checkpoint={args.cte_checkpoint}"]
    cmd.extend([
        f"task={args.sim_task}", f"EVALUATION.model_seed={args.model_seed}",
        f"EVALUATION.model_rng_protocol={args.model_rng_protocol}",
        f"EVALUATION.cross_memory_enabled={str(args.cross_memory_enabled).lower()}",
        f"EVALUATION.pim_diagnostic_interval={args.pim_diagnostic_interval}",
    ])
    if args.prefix_layout is not None:
        cmd.append(f"model.zeva.adapter.prefix_layout={args.prefix_layout}")
    if args.effect_source is not None:
        cmd.append(f"model.zeva.adapter.effect_source={args.effect_source}")
    # Resolve reader capacity from Hydra, including future task variants.
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name="sim_robotwin_zeva", overrides=[f"task={args.sim_task}"])
    cmd.append(f"EVALUATION.pim_top_k={int(cfg.model.zeva.memory.pim_top_k)}")
    if args.dataset_stats:
        cmd.append(f"EVALUATION.dataset_stats_path={args.dataset_stats}")
    if args.instruction:
        import json
        cmd.append(f"EVALUATION.fixed_instruction={json.dumps(args.instruction)}")
    if args.mode in {"zeva_stage2", "pim_on"}:
        cmd.append(f"EVALUATION.addon_checkpoint={args.addon_checkpoint}")
    context_mode = args.task_context_mode or str(cfg.model.zeva.task_context.mode)
    if args.task_context_mode is None and context_mode != "task_language" and args.task_context_bank:
        context_mode = "static" if args.task_context_retrieval_checkpoint else "bank"
    if context_mode == "task_language" and args.mode in {"zeva_stage2", "pim_on"}:
        if not all((args.task_context_bank, args.task_context_retrieval_checkpoint, args.task_language_encoder)):
            parser.error("task_language requires --task-context-bank, --task-context-retrieval-checkpoint and --task-language-encoder")
    cmd.append(f"model.zeva.task_context.mode={context_mode}")
    if args.task_context_bank:
        cmd.extend([
            f"model.zeva.task_context.bank_path={args.task_context_bank}",
        ])
    if args.task_context_retrieval_checkpoint:
        if not args.task_context_bank:
            parser.error("--task-context-retrieval-checkpoint requires --task-context-bank")
        cmd.extend([
            f"model.zeva.task_context.retrieval_checkpoint={args.task_context_retrieval_checkpoint}",
        ])
    if args.task_language_encoder:
        cmd.append(f"model.zeva.task_context.language_encoder_path={args.task_language_encoder}")
    task_context_top_k = args.task_context_top_k
    if task_context_top_k is None and args.task_context_retrieval_checkpoint:
        task_context_top_k = 5
    if task_context_top_k is not None:
        if task_context_top_k < 1:
            parser.error("--task-context-top-k must be positive")
        cmd.append(f"model.zeva.task_context.top_k={task_context_top_k}")
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
