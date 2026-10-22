#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='09: Build one main PIM cache with disjoint train/validation query views and train-only history candidates.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_preparation_launcher; iclwam_require_data
iclwam_require_directory "$PHASE_CACHE"; iclwam_require_file "$CTE_CKPT"
iclwam_expect_new "$PIM_CACHE"; iclwam_data_args; iclwam_artifact_args
iclwam_run "${TORCH_LAUNCH[@]}" scripts/build_zeva_pim_retrieval_cache.py \
    task=robotwin_zeva_main_pim_3cam_384 "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" \
    "model.zeva.prepared_stage2.pim_cache_path=$(iclwam_q "$PIM_CACHE")" "${EXTRA_ARGS[@]}"
