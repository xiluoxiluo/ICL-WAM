#!/usr/bin/env bash
set -euo pipefail

DESCRIPTION='Cache policy instruction embeddings, reusing existing text-cache entries.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"

iclwam_single_process
iclwam_require_data
iclwam_require_directory "$DIFFSYNTH_MODEL_BASE_PATH"
iclwam_positive_integer PRECOMPUTE_BATCH_SIZE "$PRECOMPUTE_BATCH_SIZE"
iclwam_data_args

# The Python entry point spawns workers over the visible GPUs itself. An empty
# PRECOMPUTE_NUM_GPUS must be absent, because its parser calls int(value).
text_env=(env -u PRECOMPUTE_NUM_GPUS
  "CUDA_VISIBLE_DEVICES=$TEXT_GPU_IDS"
  "PRECOMPUTE_BATCH_SIZE=$PRECOMPUTE_BATCH_SIZE")
if [[ -n "$PRECOMPUTE_NUM_GPUS" ]]; then
  iclwam_positive_integer PRECOMPUTE_NUM_GPUS "$PRECOMPUTE_NUM_GPUS"
  text_env+=("PRECOMPUTE_NUM_GPUS=$PRECOMPUTE_NUM_GPUS")
fi

iclwam_run "${text_env[@]}" "$PYTHON_BIN" "$REPO_ROOT/scripts/precompute_text_embeds.py" \
  task=robotwin_zeva_ego_policy_3cam_384 \
  "${DATA_ARGS[@]}" \
  ++overwrite=false \
  "${EXTRA_ARGS[@]}"
