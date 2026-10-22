#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='06: Build train-only behavior bank and frozen initial image/text readouts, with held-out query readouts.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_preparation_launcher; iclwam_require_data
iclwam_require_directory "$LATENT_CACHE" "$TEXT_CACHE" "$DIFFSYNTH_MODEL_BASE_PATH"
iclwam_require_file "$BASE_CKPT" "$ACTION_DIT_CKPT" "$CTE_CKPT"
iclwam_expect_new "$BEHAVIOR_BANK" "$READOUT_CACHE"
iclwam_data_args; iclwam_artifact_args
iclwam_run "${TORCH_LAUNCH[@]}" scripts/build_zeva_behavior_bank_optimized.py \
    task=robotwin_zeva_main_policy_3cam_384 "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" "${EXTRA_ARGS[@]}"
