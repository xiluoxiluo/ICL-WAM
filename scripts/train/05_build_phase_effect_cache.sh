#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='05: Build observed phase/effect cache from latent history, decoding real missing tails when required.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_preparation_launcher; iclwam_require_data
iclwam_require_directory "$LATENT_CACHE" "$DIFFSYNTH_MODEL_BASE_PATH"
iclwam_require_file "$CTE_CKPT"; iclwam_expect_new "$PHASE_CACHE"
iclwam_data_args; iclwam_artifact_args
iclwam_run "${TORCH_LAUNCH[@]}" scripts/build_zeva_robotwin_cache_from_latents.py \
    task=robotwin_zeva_main_cte_3cam_384 "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}" "${EXTRA_ARGS[@]}"
