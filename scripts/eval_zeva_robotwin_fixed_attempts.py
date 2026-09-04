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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("base", "pim_shadow", "pim_on"), default="pim_on")
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    if args.mode != "base" and not args.cte_checkpoint:
        parser.error("--cte-checkpoint is required for pim_shadow/pim_on")
    if args.mode == "pim_on" and not args.addon_checkpoint:
        parser.error("--addon-checkpoint is required for pim_on")
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "experiments/robotwin/eval_robotwin_single.py"),
           "--config-name", "sim_robotwin_zeva.yaml", f"ckpt={args.ckpt}",
           f"EVALUATION.task_name={args.task}", f"EVALUATION.fixed_seed={args.seed}",
           f"EVALUATION.max_attempts={args.max_attempts}", f"EVALUATION.zeva_mode={args.mode}",
           f"EVALUATION.cte_checkpoint={args.cte_checkpoint}", f"EVALUATION.addon_checkpoint={args.addon_checkpoint}"]
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
