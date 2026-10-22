#!/usr/bin/env bash
set -euo pipefail

DESCRIPTION='Train and freeze the task-language classifier using train-split features.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"

iclwam_single_process
iclwam_require_file "$TASK_MEMORY_BANK" "$LANGUAGE_ENCODER" "$LANGUAGE_FEATURES"
iclwam_expect_new "$LANGUAGE_CLASSIFIER"
iclwam_data_args
iclwam_artifact_args

# Classification training consumes prepared features only; CTE held-out
# episodes are excluded by the Python entry point's split checks.
iclwam_run env "CUDA_VISIBLE_DEVICES=$CTE_GPU_ID" \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/train_zeva_task_language.py" \
  task=robotwin_zeva_ego_policy_3cam_384 \
  "${DATA_ARGS[@]}" \
  "${ARTIFACT_ARGS[@]}" \
  "++device=$(iclwam_q "$LANGUAGE_DEVICE")" \
  "${EXTRA_ARGS[@]}"
