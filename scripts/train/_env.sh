#!/usr/bin/env bash
# Shared defaults from the existing scripts/*.sh and configs/data/robotwin.yaml.
# Override with exported variables, or a shell file selected by ICLWAM_TRAIN_ENV.
# REPO_ROOT is resolved by _common.sh; no conda environment name is assumed.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${DATA:-/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0}}"
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/data/share/1919650160032350208/zjj/fastwam/checkpoints}"
BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/checkpoints/fastwam_ckpt.pt}"
ACTION_DIT_CKPT="${ACTION_DIT_CKPT:-$REPO_ROOT/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
STATS_PATH="${STATS_PATH:-$DATA_ROOT/robotw_stats.json}"
SEMANTIC_TASK_MAP="${SEMANTIC_TASK_MAP:-$DATA_ROOT/meta/semantic_task_map.json}"
TEXT_CACHE="${TEXT_CACHE:-$DATA_ROOT/text_embeds_cache_new}"

# New required inputs: these are suggested locations, NOT files found in the old
# scripts. Supply the actual ordered task subset and real episode conditions.
CTE_TASK_SUBSET="${CTE_TASK_SUBSET:-$DATA_ROOT/meta/cte_task_subset.json}"
EPISODE_CONDITIONS="${EPISODE_CONDITIONS:-$DATA_ROOT/meta/episode_conditions.json}"

# One stable experiment root for all stages. Change it for a new experiment;
# never mix the old September CTE/VAE/static-retrieval artifacts into this run.
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/zeva_ego_reference_v3}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$RUN_ROOT/artifacts}"
CTE_RUN_DIR="${CTE_RUN_DIR:-$RUN_ROOT/cte}"
CTE_CKPT="${CTE_CKPT:-$CTE_RUN_DIR/cte_epoch_080.pt}"
BOUNDARY_CACHE="${BOUNDARY_CACHE:-$ARTIFACT_ROOT/boundary_v2}"
TASK_MEMORY_BANK="${TASK_MEMORY_BANK:-$ARTIFACT_ROOT/task_memory_bank.pt}"
LANGUAGE_ENCODER="${LANGUAGE_ENCODER:-$ARTIFACT_ROOT/task_language_encoder.pt}"
LANGUAGE_FEATURES="${LANGUAGE_FEATURES:-$ARTIFACT_ROOT/task_language_features.pt}"
LANGUAGE_CLASSIFIER="${LANGUAGE_CLASSIFIER:-$ARTIFACT_ROOT/task_language_classifier.pt}"
QUERY_CONTEXT_CACHE="${QUERY_CONTEXT_CACHE:-$ARTIFACT_ROOT/query_context_v2.pt}"
PIM_TRAIN="${PIM_TRAIN:-$ARTIFACT_ROOT/pim_train_v4}"
PIM_VALIDATION="${PIM_VALIDATION:-$ARTIFACT_ROOT/pim_validation_v4}"

# The stages are independent: freezing B does not select a different A parent.
STAGE2A_FOUNDATION_MODE="${STAGE2A_FOUNDATION_MODE:-joint}"
STAGE2B_FOUNDATION_MODE="${STAGE2B_FOUNDATION_MODE:-joint}"
STAGE2A_STEPS="${STAGE2A_STEPS:-5000}"
STAGE2B_STEPS="${STAGE2B_STEPS:-2000}"
STAGE2A_RUN_DIR="${STAGE2A_RUN_DIR:-$RUN_ROOT/stage2a_$STAGE2A_FOUNDATION_MODE}"
STAGE2B_RUN_DIR="${STAGE2B_RUN_DIR:-$RUN_ROOT/stage2b_$STAGE2B_FOUNDATION_MODE}"
# STAGE2A_CKPT is derived from STAGE2A_STEPS in _common.sh unless overridden.
CTE_RESUME="${CTE_RESUME:-}"
STAGE2A_RESUME="${STAGE2A_RESUME:-}"
STAGE2B_RESUME="${STAGE2B_RESUME:-}"

# GPU IDs are machine-local. Set GPU_IDS explicitly on an 8-GPU node, or inherit
# an existing CUDA_VISIBLE_DEVICES allocation. CTE/boundary remain single rank.
GPU_IDS="${GPU_IDS-${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}"
TEXT_GPU_IDS="${TEXT_GPU_IDS:-$GPU_IDS}"
CTE_GPU_ID="${CTE_GPU_ID:-${GPU_IDS%%,*}}"
BOUNDARY_GPU_ID="${BOUNDARY_GPU_ID:-$CTE_GPU_ID}"
LANGUAGE_DEVICE="${LANGUAGE_DEVICE:-cpu}"
PRECOMPUTE_NUM_GPUS="${PRECOMPUTE_NUM_GPUS:-}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-8}"
CTE_BATCH_SIZE="${CTE_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# NPROC_PER_NODE, GRAD_ACCUM_STEPS, ZEVA_OPTIMIZER_SHARDING are derived from the
# actual topology in iclwam_stage2. An explicit override is validated there.
CHECK_ENVIRONMENT="${CHECK_ENVIRONMENT:-1}"
