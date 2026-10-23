#!/usr/bin/env bash
set -euo pipefail
DESCRIPTION='03: Precompute main-success Wan VAE latents with unchanged episode provenance.'
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
iclwam_init "$@"; iclwam_preparation_launcher; iclwam_require_data
iclwam_require_directory "$DIFFSYNTH_MODEL_BASE_PATH"
iclwam_positive_integer LATENT_BATCH_SIZE "$LATENT_BATCH_SIZE"
iclwam_positive_integer LATENT_PREFETCH_FACTOR "$LATENT_PREFETCH_FACTOR"
iclwam_positive_integer LATENT_STREAMING_MAX_OPEN "$LATENT_STREAMING_MAX_OPEN"
[[ "$LATENT_STREAMING_PYAV" == 0 || "$LATENT_STREAMING_PYAV" == 1 ]] || iclwam_die 'LATENT_STREAMING_PYAV must be 0 or 1.'
[[ "$LATENT_STREAMING_VERIFY_EVERY" =~ ^[0-9]+$ ]] || iclwam_die 'LATENT_STREAMING_VERIFY_EVERY must be >= 0.'
streaming_pyav=false
if [[ "$LATENT_STREAMING_PYAV" == 1 ]]; then streaming_pyav=true; fi
[[ "$LATENT_FAST_COLLATE" == 0 || "$LATENT_FAST_COLLATE" == 1 ]] || iclwam_die "LATENT_FAST_COLLATE must be 0 or 1."
[[ "$LATENT_PROFILE_EVERY_BATCHES" =~ ^[0-9]+$ ]] || iclwam_die "LATENT_PROFILE_EVERY_BATCHES must be >= 0."
fast_collate=false
if [[ "$LATENT_FAST_COLLATE" == 1 ]]; then fast_collate=true; fi
[[ "$LATENT_RESUME" == 0 || "$LATENT_RESUME" == 1 ]] || iclwam_die 'LATENT_RESUME must be 0 or 1.'
if [[ "$LATENT_RESUME" == 1 ]]; then iclwam_require_directory "$LATENT_CACHE"
else iclwam_expect_new "$LATENT_CACHE"; fi
iclwam_data_args
iclwam_run "${TORCH_LAUNCH[@]}" scripts/precompute_zeva_cte_latents.py \
    task=robotwin_zeva_main_cte_3cam_384 "${DATA_ARGS[@]}" \
    "model.zeva.cte.latent_cache_path=$(iclwam_q "$LATENT_CACHE")" \
    "++cte_latent_cache.batch_size=$LATENT_BATCH_SIZE" "++cte_latent_cache.num_workers=$NUM_WORKERS" \
    "++cte_latent_cache.prefetch_factor=$LATENT_PREFETCH_FACTOR" \
    "++cte_latent_cache.fast_collate=$fast_collate" \
    "++cte_latent_cache.streaming_pyav=$streaming_pyav" \
    "++cte_latent_cache.streaming_verify_every=$LATENT_STREAMING_VERIFY_EVERY" \
    "++cte_latent_cache.streaming_max_open=$LATENT_STREAMING_MAX_OPEN" \
    "++cte_latent_cache.profile_every_batches=$LATENT_PROFILE_EVERY_BATCHES" \
    ++cte_latent_cache.presample_images=true ++cte_latent_cache.overwrite=false "${EXTRA_ARGS[@]}"
