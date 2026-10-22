#!/usr/bin/env bash
set -euo pipefail

DESCRIPTION='Train the reference CTE for 80 full-episode epochs on one GPU.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"

iclwam_single_process
iclwam_require_data
iclwam_require_file "$CTE_TASK_SUBSET" "$EPISODE_CONDITIONS"
iclwam_positive_integer CTE_BATCH_SIZE "$CTE_BATCH_SIZE"
((CTE_BATCH_SIZE % 2 == 0)) || iclwam_die 'CTE_BATCH_SIZE must be even for same-task episode pairs.'
[[ -n "$CTE_GPU_ID" && "$CTE_GPU_ID" != *,* ]] || iclwam_die 'CTE_GPU_ID must select one CUDA device.'
resume_args=()
if [[ -n "$CTE_RESUME" ]]; then
  iclwam_require_file "$CTE_RESUME"
  resume_args+=("resume=$(iclwam_q "$CTE_RESUME")")
fi

# Stage1 uses RGB/actions only. Keep unrelated policy/text-cache paths out of
# the dataset manifest. The trainer validates empty fresh outputs and resumes;
# do not create a log file inside CTE_RUN_DIR before starting it.
iclwam_run env "CUDA_VISIBLE_DEVICES=$CTE_GPU_ID" \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/train_zeva_ego_cte.py" \
  task=robotwin_zeva_fastwam_ego_cte_3cam_384 \
  "data.train.dataset_dirs=[$(iclwam_q "$DATA_ROOT")]" \
  "data.train.pretrained_norm_stats=$(iclwam_q "$STATS_PATH")" \
  "data.train.semantic_task_map_path=$(iclwam_q "$SEMANTIC_TASK_MAP")" \
  data.val=null \
  "cte_task_subset_path=$(iclwam_q "$CTE_TASK_SUBSET")" \
  "cte_episode_conditions_path=$(iclwam_q "$EPISODE_CONDITIONS")" \
  "output_dir=$(iclwam_q "$CTE_RUN_DIR")" \
  "batch_size=$CTE_BATCH_SIZE" \
  "num_workers=$NUM_WORKERS" \
  num_epochs=80 \
  max_steps=null \
  gradient_accumulation_steps=1 \
  device=cuda \
  "${resume_args[@]}" \
  "${EXTRA_ARGS[@]}"
