#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='04: Train observed-effect CTE on success demonstrations; local step-based recipe, not the Ego 80-epoch trainer.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_single_process
iclwam_distributed "$CTE_GPU_IDS" "${CTE_NPROC_PER_NODE:-}" 1
iclwam_require_data; iclwam_require_directory "$LATENT_CACHE"
iclwam_positive_integer CTE_STEPS "$CTE_STEPS"; iclwam_positive_integer CTE_BATCH_SIZE "$CTE_BATCH_SIZE"
iclwam_positive_integer CTE_TASKS_PER_BATCH "$CTE_TASKS_PER_BATCH"
iclwam_positive_integer CTE_SAMPLES_PER_TASK "$CTE_SAMPLES_PER_TASK"
((CTE_BATCH_SIZE == CTE_TASKS_PER_BATCH * CTE_SAMPLES_PER_TASK)) || iclwam_die 'CTE_BATCH_SIZE must equal tasks per batch * samples per task.'
if [[ -n "$CTE_RESUME" ]]; then iclwam_require_file "$CTE_RESUME"
else iclwam_expect_new "$CTE_RUN_DIR"; fi
iclwam_data_args
cmd=("${TORCH_LAUNCH[@]}" scripts/train_zeva_cte.py task=robotwin_zeva_main_cte_3cam_384
    "${DATA_ARGS[@]}" "model.zeva.cte.latent_cache_path=$(iclwam_q "$LATENT_CACHE")"
    "output_dir=$(iclwam_q "$CTE_RUN_DIR")" "max_steps=$CTE_STEPS" "learning_rate=$CTE_LEARNING_RATE"
    "batch_size=$CTE_BATCH_SIZE" "cte_tasks_per_batch=$CTE_TASKS_PER_BATCH"
    "cte_samples_per_task=$CTE_SAMPLES_PER_TASK" "num_workers=$NUM_WORKERS" gradient_accumulation_steps=1)
if [[ -n "$CTE_RESUME" ]]; then cmd+=("resume=$(iclwam_q "$CTE_RESUME")"); fi
iclwam_run "${cmd[@]}" "${EXTRA_ARGS[@]}"
