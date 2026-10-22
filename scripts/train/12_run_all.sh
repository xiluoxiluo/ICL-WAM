#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
from=1; to=11; DRY_RUN="${DRY_RUN:-0}"
while (($#)); do
    case "$1" in
        --from|--to)
            (($# >= 2)) || iclwam_die "Missing value for $1"
            [[ "$2" =~ ^(0?[1-9]|1[01])$ ]] || iclwam_die 'Stage range must be 01 through 11.'
            if [[ "$1" == --from ]]; then from=$((10#$2)); else to=$((10#$2)); fi
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            printf '%s\n' 'Usage: bash scripts/train/12_run_all.sh [--from 01] [--to 11] [--dry-run]' \
                'Configure SUCCESS_MANIFEST and paths in environment or ICLWAM_TRAIN_ENV.' \
                'Run preparation 01-09 once on one node; run 10-11 on every participating training node.'
            exit 0 ;;
        *) iclwam_die "Unknown orchestrator option: $1" ;;
    esac
done
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || iclwam_die 'DRY_RUN must be 0 or 1.'
((from <= to)) || iclwam_die '--from must not exceed --to.'
iclwam_positive_integer NNODES "$NNODES"
if ((NNODES > 1 && from < 10)); then iclwam_die 'Run 01-09 once with NNODES=1, then --from 10 on each training node.'; fi
stages=(01_check_environment.sh 02_precompute_text_embeds.sh 03_precompute_cte_latents.sh
    04_train_cte.sh 05_build_phase_effect_cache.sh 06_build_behavior_bank.sh
    07_train_static_retrieval.sh 08_build_static_context.sh 09_build_pim_cache.sh
    10_train_stage2a.sh 11_train_stage2b.sh)
for ((step=from; step<=to; step++)); do
    printf '\n========== %s ==========\n' "${stages[step-1]}"
    if [[ "$DRY_RUN" == 1 ]]; then bash "$TRAIN_SCRIPT_DIR/${stages[step-1]}" --dry-run
    else bash "$TRAIN_SCRIPT_DIR/${stages[step-1]}"; fi
done
