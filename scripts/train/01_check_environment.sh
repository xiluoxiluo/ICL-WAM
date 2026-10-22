#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='01: Check real success-demonstration provenance, existing assets and the activated environment.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_no_extra; iclwam_single_process
iclwam_require_success
[[ "$CHECK_ENVIRONMENT" == 0 || "$CHECK_ENVIRONMENT" == 1 ]] || iclwam_die 'CHECK_ENVIRONMENT must be 0 or 1.'
cmd=(env "CUDA_VISIBLE_DEVICES=$GPU_IDS" "$PYTHON_BIN" "$TRAIN_SCRIPT_DIR/_preflight.py"
    --dataset-root "$DATA_ROOT" --stats "$STATS_PATH" --semantic-map "$SEMANTIC_TASK_MAP"
    --success-manifest "$SUCCESS_MANIFEST" --base-checkpoint "$BASE_CKPT"
    --action-dit-checkpoint "$ACTION_DIT_CKPT" --assets-root "$DIFFSYNTH_MODEL_BASE_PATH")
if [[ "$CHECK_ENVIRONMENT" == 1 ]]; then cmd+=(--check-environment); fi
iclwam_run "${cmd[@]}"
