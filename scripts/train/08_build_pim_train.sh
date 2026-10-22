#!/usr/bin/env bash
set -euo pipefail

DESCRIPTION='Build the strict Zeva-Ego PIM v4 training cache.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"

iclwam_no_extra
iclwam_single_process
iclwam_require_directory "$BOUNDARY_CACHE"
iclwam_require_file \
  "$SEMANTIC_TASK_MAP" \
  "$EPISODE_CONDITIONS" \
  "$BOUNDARY_CACHE/episode_splits.json"
iclwam_expect_new "$PIM_TRAIN"

iclwam_run "$PYTHON_BIN" "$REPO_ROOT/scripts/build_zeva_cross_attempt_cache.py" \
  --phase-cache "$BOUNDARY_CACHE" \
  --semantic-map "$SEMANTIC_TASK_MAP" \
  --episode-splits "$BOUNDARY_CACHE/episode_splits.json" \
  --conditions "$EPISODE_CONDITIONS" \
  --history-pairing reference_nearest \
  --query-scope all_cte_frames_v1 \
  --pairing-metric torch_float32_first_in_source_order \
  --uniform-indexing reference_round \
  --max-past-attempts 4 \
  --max-entries-per-attempt 64 \
  --bit-size 4 \
  --seed 20260919 \
  --split train \
  --output "$PIM_TRAIN"
