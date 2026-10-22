#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='Run a contiguous range of the numbered offline training stages.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
from=1
to=11
DRY_RUN="${DRY_RUN:-0}"
while (($#)); do
    case "$1" in
        --from|--to)
            (($# >= 2)) || iclwam_die "Missing value for $1"
            [[ "$2" =~ ^(0?[1-9]|1[01])$ ]] || iclwam_die 'Stage range must be from 01 to 11.'
            if [[ "$1" == --from ]]; then from=$((10#$2)); else to=$((10#$2)); fi
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            printf '%s\n' 'Usage: bash scripts/train/12_run_all.sh [--from 01] [--to 11] [--dry-run]' \
                'Run preparation 01-09 once, then launch 10-11 on every node for multi-node training.' \
                'Use environment variables or ICLWAM_TRAIN_ENV for configuration; see scripts/train/README.md.'
            exit 0 ;;
        *) iclwam_die "Unknown orchestrator argument: $1" ;;
    esac
done
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || iclwam_die 'DRY_RUN must be 0 or 1.'
((from <= to)) || iclwam_die '--from must not be greater than --to.'
iclwam_positive_integer NNODES "$NNODES"
if ((NNODES > 1 && from < 10)); then
    iclwam_die 'Run 01-09 once with NNODES=1, then run --from 10 on each node with NNODES/NODE_RANK/MASTER_ADDR set.'
fi
stages=(01_check_environment.sh 02_precompute_text_embeds.sh 03_train_cte.sh
    04_build_boundary_cache.sh 05_build_task_memory.sh 06_train_task_language.sh
    07_build_query_context.sh 08_build_pim_train.sh 09_build_pim_validation.sh
    10_train_stage2a.sh 11_train_stage2b.sh)
for ((step=from; step<=to; step++)); do
    printf '\n========== %s ==========\n' "${stages[step-1]}"
    # Each child reads the same defaults and optional user configuration. A CLI
    # flag takes precedence even if that configuration assigns DRY_RUN=0.
    if [[ "$DRY_RUN" == 1 ]]; then
        bash "$TRAIN_SCRIPT_DIR/${stages[step-1]}" --dry-run
    else
        bash "$TRAIN_SCRIPT_DIR/${stages[step-1]}"
    fi
done
