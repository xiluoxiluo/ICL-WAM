#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='02: Cache policy text embeddings; the compatible existing text cache can be reused.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_single_process; iclwam_require_data
iclwam_require_directory "$DIFFSYNTH_MODEL_BASE_PATH"
iclwam_positive_integer PRECOMPUTE_BATCH_SIZE "$PRECOMPUTE_BATCH_SIZE"
iclwam_data_args
text_env=(env -u PRECOMPUTE_NUM_GPUS "CUDA_VISIBLE_DEVICES=$TEXT_GPU_IDS" "PRECOMPUTE_BATCH_SIZE=$PRECOMPUTE_BATCH_SIZE")
if [[ -n "$PRECOMPUTE_NUM_GPUS" ]]; then
    iclwam_positive_integer PRECOMPUTE_NUM_GPUS "$PRECOMPUTE_NUM_GPUS"
    text_env+=("PRECOMPUTE_NUM_GPUS=$PRECOMPUTE_NUM_GPUS")
fi
iclwam_run "${text_env[@]}" "$PYTHON_BIN" "$REPO_ROOT/scripts/precompute_text_embeds.py" \
    task=robotwin_zeva_main_policy_3cam_384 "${DATA_ARGS[@]}" ++overwrite=false "${EXTRA_ARGS[@]}"
