#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='07: Train the static image/text retrieval head with the shared success-episode holdout.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_single_process; iclwam_require_data
iclwam_require_file "$BEHAVIOR_BANK" "$READOUT_CACHE"; iclwam_expect_new "$RETRIEVAL_HEAD"
iclwam_positive_integer RETRIEVAL_STEPS "$RETRIEVAL_STEPS"
iclwam_positive_integer RETRIEVAL_BATCH_SIZE "$RETRIEVAL_BATCH_SIZE"
iclwam_data_args; iclwam_artifact_args
iclwam_run env "CUDA_VISIBLE_DEVICES=$RETRIEVAL_GPU_ID" "$PYTHON_BIN" scripts/train_zeva_task_context_retrieval_optimized.py \
    task=robotwin_zeva_main_policy_3cam_384 "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" \
    "device=${RETRIEVAL_DEVICE:-cuda}" "model.zeva.task_context.training.steps=$RETRIEVAL_STEPS" \
    "model.zeva.task_context.training.batch_size=$RETRIEVAL_BATCH_SIZE" \
    "model.zeva.task_context.training.learning_rate=$RETRIEVAL_LEARNING_RATE" "${EXTRA_ARGS[@]}"
