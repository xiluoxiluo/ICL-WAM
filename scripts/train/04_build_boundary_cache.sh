#!/usr/bin/env bash
set -euo pipefail

DESCRIPTION='Export the ready CTE on every CTE frame into the boundary v2 cache.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"

iclwam_single_process
iclwam_require_data
iclwam_require_file "$CTE_CKPT"
iclwam_expect_new "$BOUNDARY_CACHE"
[[ -n "$BOUNDARY_GPU_ID" && "$BOUNDARY_GPU_ID" != *,* ]] || iclwam_die 'BOUNDARY_GPU_ID must select one CUDA device.'
iclwam_data_args
iclwam_artifact_args

# The exporter checks the CTE readiness gate and preserves its exact source
# scope and episode split; it does not require policy weights or text caches.
iclwam_run env "CUDA_VISIBLE_DEVICES=$BOUNDARY_GPU_ID" \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/build_zeva_ego_boundary_cache.py" \
  task=robotwin_zeva_ego_policy_3cam_384 \
  "${DATA_ARGS[@]}" \
  "${ARTIFACT_ARGS[@]}" \
  ++device=cuda \
  "${EXTRA_ARGS[@]}"
