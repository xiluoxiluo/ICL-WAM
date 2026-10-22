#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='08: Build static context for train and held-out episodes using only train bank candidates.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_single_process; iclwam_require_data
iclwam_require_file "$BEHAVIOR_BANK" "$READOUT_CACHE" "$RETRIEVAL_HEAD"
iclwam_expect_new "$STATIC_CONTEXT_CACHE"; iclwam_data_args; iclwam_artifact_args
iclwam_run env "CUDA_VISIBLE_DEVICES=$RETRIEVAL_GPU_ID" "$PYTHON_BIN" scripts/build_zeva_static_episode_context_cache.py \
    task=robotwin_zeva_main_policy_3cam_384 "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" "${EXTRA_ARGS[@]}"
