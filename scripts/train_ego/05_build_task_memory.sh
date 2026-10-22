#!/usr/bin/env bash
set -euo pipefail

DESCRIPTION='Build the train-only task memory bank and frozen language features.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"

iclwam_single_process
iclwam_require_data
iclwam_require_directory "$BOUNDARY_CACHE"
iclwam_require_file "$CTE_CKPT" "$BOUNDARY_CACHE/manifest.json" "$BOUNDARY_CACHE/episode_splits.json"
iclwam_expect_new "$TASK_MEMORY_BANK" "$LANGUAGE_FEATURES"
# Reusing the frozen language encoder is supported. Only a new export needs
# the Wan tokenizer/input-embedding assets; no full FastWAM model is loaded.
if [[ ! -f "$LANGUAGE_ENCODER" ]]; then
  iclwam_require_directory "$DIFFSYNTH_MODEL_BASE_PATH"
fi
iclwam_data_args
iclwam_artifact_args

iclwam_run env "CUDA_VISIBLE_DEVICES=$CTE_GPU_ID" \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/build_zeva_ego_task_memory.py" \
  task=robotwin_zeva_ego_policy_3cam_384 \
  "${DATA_ARGS[@]}" \
  "${ARTIFACT_ARGS[@]}" \
  model.zeva.task_context.build_stage=bank \
  "++device=$(iclwam_q "$LANGUAGE_DEVICE")" \
  "${EXTRA_ARGS[@]}"
