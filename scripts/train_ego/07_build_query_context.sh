#!/usr/bin/env bash
set -euo pipefail

DESCRIPTION='Build per-decision task-context v2 using the frozen language classifier.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"

iclwam_single_process
iclwam_require_data
iclwam_require_directory "$BOUNDARY_CACHE"
iclwam_require_file \
  "$CTE_CKPT" \
  "$BOUNDARY_CACHE/manifest.json" \
  "$BOUNDARY_CACHE/episode_splits.json" \
  "$TASK_MEMORY_BANK" \
  "$LANGUAGE_ENCODER" \
  "$LANGUAGE_FEATURES" \
  "$LANGUAGE_CLASSIFIER"
iclwam_expect_new "$QUERY_CONTEXT_CACHE"
iclwam_data_args
iclwam_artifact_args

# The reference config selects per_decision queries and train-only memory.
# static_context names the builder phase, not an episode-static query scope.
iclwam_run env "CUDA_VISIBLE_DEVICES=$CTE_GPU_ID" \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/build_zeva_ego_task_memory.py" \
  task=robotwin_zeva_ego_policy_3cam_384 \
  "${DATA_ARGS[@]}" \
  "${ARTIFACT_ARGS[@]}" \
  model.zeva.task_context.build_stage=static_context \
  "++device=$(iclwam_q "$LANGUAGE_DEVICE")" \
  "${EXTRA_ARGS[@]}"
