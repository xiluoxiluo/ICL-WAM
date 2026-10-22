#!/usr/bin/env bash
# Internal helpers. Launch a numbered script instead of sourcing this file.

TRAIN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$TRAIN_SCRIPT_DIR/../.." && pwd)}"
if [[ -n "${ICLWAM_TRAIN_ENV:-}" ]]; then
    [[ -f "$ICLWAM_TRAIN_ENV" ]] || { echo "Missing ICLWAM_TRAIN_ENV: $ICLWAM_TRAIN_ENV" >&2; exit 2; }
    # This is an explicitly selected, user-owned shell configuration file.
    source "$ICLWAM_TRAIN_ENV"
fi
REPO_ROOT="$(cd -- "$REPO_ROOT" && pwd)"
source "$TRAIN_SCRIPT_DIR/_env.sh"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"
cd -- "$REPO_ROOT"

iclwam_die() { echo "ERROR: $*" >&2; exit 2; }

iclwam_positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || iclwam_die "$1 must be a positive integer (got '$2')."
}

iclwam_q() {
    # Quote Hydra's RHS as data; never evaluate a command or concatenate a shell
    # command string. In particular, commas in a path are not a Hydra sweep.
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '"%s"' "$value"
}

iclwam_init() {
    DRY_RUN="${DRY_RUN:-0}"
    EXTRA_ARGS=()
    local arg key
    while (($#)); do
        case "$1" in
            --dry-run) DRY_RUN=1; shift ;;
            -h|--help)
                printf '%s\n\n' "${DESCRIPTION:-ICL-WAM offline training}"
                printf 'Usage: bash %s [--dry-run] [Hydra overrides...]\n' "${BASH_SOURCE[1]##*/}"
                printf '%s\n' 'Configure paths, GPUs and resume through _env.sh / environment variables.' \
                    'See scripts/train/README.md. --dry-run prints commands without Python, CUDA or file writes.'
                exit 0 ;;
            --) shift; EXTRA_ARGS+=("$@"); break ;;
            --*) iclwam_die "Unknown launcher option: $1" ;;
            *) EXTRA_ARGS+=("$1"); shift ;;
        esac
    done
    [[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || iclwam_die 'DRY_RUN must be 0 or 1.'
    for arg in "${EXTRA_ARGS[@]}"; do
        [[ "$arg" == *=* ]] || iclwam_die "Expected a Hydra key=value override: $arg"
        key="${arg%%=*}"; key="${key#+}"; key="${key#+}"; key="${key#~}"
        case "$key" in
            task|ckpt|output_dir|resume|foundation_mode|foundation_mode_settings*|max_steps|num_epochs|batch_size|gradient_accumulation_steps|zeva_optimizer_sharding|device|overwrite|cte_task_subset_path|cte_episode_conditions_path|cte_smoke_steps|data|data.train|data.val|data.train.dataset_dirs|data.train.pretrained_norm_stats|data.train.semantic_task_map_path|data.train.text_embedding_cache_dir|model|model.action_dit_pretrained_path|model.zeva|model.zeva.cte|model.zeva.cte.checkpoint|model.zeva.cache|model.zeva.cache.path|model.zeva.task_context|model.zeva.task_context.*path|model.zeva.task_context.retrieval_checkpoint|model.zeva.task_context.build_stage|model.zeva.prepared_stage2|model.zeva.prepared_stage2.*cache_path|model.zeva.policy_checkpoint|model.zeva.policy_warmstart_checkpoint|model.zeva.foundation_mode|model.zeva.freeze_fastwam|model.zeva.training_stage|model.zeva.training_recipe)
                iclwam_die "Workflow-controlled override '$key': use its environment variable in _env.sh so all stages share the same paths/recipe." ;;
        esac
    done
}

iclwam_no_extra() {
    ((${#EXTRA_ARGS[@]} == 0)) || iclwam_die 'This stage has no Hydra overrides; configure it with environment variables.'
}

iclwam_run() {
    printf '\n+ '; printf '%q ' "$@"; printf '\n'
    [[ "$DRY_RUN" == 1 ]] || "$@"
}

iclwam_require_file() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    local path
    for path in "$@"; do [[ -f "$path" ]] || iclwam_die "Required file missing: $path"; done
}

iclwam_require_directory() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    local path
    for path in "$@"; do [[ -d "$path" ]] || iclwam_die "Required directory missing: $path"; done
}

iclwam_require_data() {
    iclwam_require_directory "$DATA_ROOT"
    iclwam_require_file "$STATS_PATH" "$SEMANTIC_TASK_MAP"
}

iclwam_expect_new() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    local path
    for path in "$@"; do
        [[ ! -e "$path" && ! -L "$path" ]] || iclwam_die "Output already exists: $path. Reuse it by starting at a later stage, or choose a new RUN_ROOT/output path."
    done
}

iclwam_single_process() {
    [[ "${WORLD_SIZE:-1}" == 1 ]] || iclwam_die 'This preparation stage is single-process; do not launch it under torchrun.'
    [[ "$NODE_RANK" == 0 ]] || iclwam_die 'Run shared preparation stages once on node 0, before launching Stage2 on all nodes.'
}

iclwam_data_args() {
    DATA_ARGS=(
        "data.train.dataset_dirs=[$(iclwam_q "$DATA_ROOT")]"
        "data.train.pretrained_norm_stats=$(iclwam_q "$STATS_PATH")"
        "data.train.semantic_task_map_path=$(iclwam_q "$SEMANTIC_TASK_MAP")"
        "data.train.text_embedding_cache_dir=$(iclwam_q "$TEXT_CACHE")"
        'data.val=null'
    )
}

iclwam_artifact_args() {
    ARTIFACT_ARGS=(
        "ckpt=$(iclwam_q "$BASE_CKPT")"
        "model.action_dit_pretrained_path=$(iclwam_q "$ACTION_DIT_CKPT")"
        "model.zeva.cte.checkpoint=$(iclwam_q "$CTE_CKPT")"
        "model.zeva.cache.path=$(iclwam_q "$BOUNDARY_CACHE")"
        "model.zeva.task_context.bank_path=$(iclwam_q "$TASK_MEMORY_BANK")"
        "model.zeva.task_context.language_encoder_path=$(iclwam_q "$LANGUAGE_ENCODER")"
        "model.zeva.task_context.language_feature_cache_path=$(iclwam_q "$LANGUAGE_FEATURES")"
        "model.zeva.task_context.retrieval_checkpoint=$(iclwam_q "$LANGUAGE_CLASSIFIER")"
        "model.zeva.task_context.static_episode_cache_path=$(iclwam_q "$QUERY_CONTEXT_CACHE")"
    )
}

iclwam_stage2() {
    local stage="$1" mode steps output resume task world accum sharding parent_filename gpu seen=,
    local -a visible_ids launch cmd
    [[ "$stage" == A || "$stage" == B ]] || iclwam_die 'Internal error: expected Stage2 A or B.'
    [[ "${WORLD_SIZE:-1}" == 1 ]] || iclwam_die 'This script starts torchrun itself; do not nest it inside another distributed launcher.'
    [[ -n "$GPU_IDS" && "$GPU_IDS" != -1 && "$GPU_IDS" != *, && "$GPU_IDS" != ,* && "$GPU_IDS" != *,,* && "$GPU_IDS" != *[[:space:]]* ]] || iclwam_die 'GPU_IDS must be a nonempty comma-separated CUDA device list.'
    IFS=, read -r -a visible_ids <<< "$GPU_IDS"
    for gpu in "${visible_ids[@]}"; do
        [[ "$seen" != *",$gpu,"* ]] || iclwam_die "GPU_IDS contains a duplicate device: $gpu"
        seen+="$gpu,"
    done
    NPROC_PER_NODE="${NPROC_PER_NODE:-${#visible_ids[@]}}"
    iclwam_positive_integer NPROC_PER_NODE "$NPROC_PER_NODE"
    iclwam_positive_integer NNODES "$NNODES"
    iclwam_positive_integer BATCH_SIZE "$BATCH_SIZE"
    iclwam_positive_integer GLOBAL_BATCH_SIZE "$GLOBAL_BATCH_SIZE"
    iclwam_positive_integer STAGE2A_STEPS "$STAGE2A_STEPS"
    iclwam_positive_integer STAGE2B_STEPS "$STAGE2B_STEPS"
    [[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] && ((NODE_RANK < NNODES)) || iclwam_die 'NODE_RANK must be in [0, NNODES).'
    ((NPROC_PER_NODE <= ${#visible_ids[@]})) || iclwam_die 'NPROC_PER_NODE exceeds the number of entries in GPU_IDS.'
    world=$((NNODES * NPROC_PER_NODE))
    ((GLOBAL_BATCH_SIZE % (world * BATCH_SIZE) == 0)) || iclwam_die 'GLOBAL_BATCH_SIZE must be divisible by NNODES * NPROC_PER_NODE * BATCH_SIZE.'
    accum="${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (world * BATCH_SIZE)))}"
    iclwam_positive_integer GRAD_ACCUM_STEPS "$accum"
    ((world * BATCH_SIZE * accum == GLOBAL_BATCH_SIZE)) || iclwam_die 'GRAD_ACCUM_STEPS gives a different effective batch; update GLOBAL_BATCH_SIZE explicitly.'
    sharding="${ZEVA_OPTIMIZER_SHARDING:-}"
    if [[ -z "$sharding" ]]; then
        if ((world > 1)); then sharding=zero1; else sharding=none; fi
    fi
    [[ "$sharding" == none || "$sharding" == zero1 ]] || iclwam_die 'ZEVA_OPTIMIZER_SHARDING must be none or zero1.'
    [[ "$sharding" != zero1 || "$world" -gt 1 ]] || iclwam_die 'zero1 requires at least two ranks.'
    [[ "$STAGE2A_FOUNDATION_MODE" == joint || "$STAGE2A_FOUNDATION_MODE" == frozen ]] || iclwam_die 'STAGE2A_FOUNDATION_MODE must be joint or frozen.'
    [[ "$STAGE2B_FOUNDATION_MODE" == joint || "$STAGE2B_FOUNDATION_MODE" == frozen ]] || iclwam_die 'STAGE2B_FOUNDATION_MODE must be joint or frozen.'
    printf -v parent_filename 'step_%06d_addon.pt' "$STAGE2A_STEPS"
    STAGE2A_CKPT="${STAGE2A_CKPT:-$STAGE2A_RUN_DIR/checkpoints/weights/$parent_filename}"
    if [[ "$stage" == A ]]; then
        mode="$STAGE2A_FOUNDATION_MODE"; steps="$STAGE2A_STEPS"; output="$STAGE2A_RUN_DIR"; resume="$STAGE2A_RESUME"
        task=robotwin_zeva_ego_policy_16gpu_3cam_384
    else
        mode="$STAGE2B_FOUNDATION_MODE"; steps="$STAGE2B_STEPS"; output="$STAGE2B_RUN_DIR"; resume="$STAGE2B_RESUME"
        task=robotwin_zeva_ego_pim_16gpu_3cam_384
    fi
    iclwam_require_data
    iclwam_require_directory "$DIFFSYNTH_MODEL_BASE_PATH" "$TEXT_CACHE" "$BOUNDARY_CACHE" "$PIM_TRAIN" "$PIM_VALIDATION"
    iclwam_require_file "$BASE_CKPT" "$ACTION_DIT_CKPT" "$CTE_CKPT" "$TASK_MEMORY_BANK" "$LANGUAGE_ENCODER" "$LANGUAGE_FEATURES" "$LANGUAGE_CLASSIFIER" "$QUERY_CONTEXT_CACHE"
    if [[ -n "$resume" ]]; then
        # A complete trainer state directory restores the parent and optimizer;
        # a deployment addon .pt alone cannot resume training.
        iclwam_require_directory "$resume"
    else
        iclwam_expect_new "$output"
        if [[ "$stage" == B ]]; then iclwam_require_file "$STAGE2A_CKPT"; fi
    fi
    launch=("$PYTHON_BIN" -m torch.distributed.run "--nproc-per-node=$NPROC_PER_NODE")
    if ((NNODES == 1)); then
        launch+=(--standalone)
    else
        [[ -n "$MASTER_ADDR" ]] || iclwam_die 'Set MASTER_ADDR to the reachable node-0 address for multi-node training.'
        iclwam_positive_integer MASTER_PORT "$MASTER_PORT"
        ((MASTER_PORT <= 65535)) || iclwam_die 'MASTER_PORT must be at most 65535.'
        launch+=("--nnodes=$NNODES" "--node-rank=$NODE_RANK" "--master-addr=$MASTER_ADDR" "--master-port=$MASTER_PORT")
    fi
    iclwam_data_args
    iclwam_artifact_args
    cmd=(env "CUDA_VISIBLE_DEVICES=$GPU_IDS" "${launch[@]}" scripts/train_zeva_fastwam_prepared.py
        "task=$task" "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}"
        "model.zeva.prepared_stage2.pim_cache_path=$(iclwam_q "$PIM_TRAIN")"
        "model.zeva.prepared_stage2.validation_pim_cache_path=$(iclwam_q "$PIM_VALIDATION")"
        "output_dir=$(iclwam_q "$output")" "foundation_mode=$mode" "max_steps=$steps"
        "batch_size=$BATCH_SIZE" "gradient_accumulation_steps=$accum" "zeva_optimizer_sharding=$sharding"
        "num_workers=$NUM_WORKERS")
    if [[ -n "$resume" ]]; then
        cmd+=("resume=$(iclwam_q "$resume")")
    elif [[ "$stage" == B ]]; then
        cmd+=("model.zeva.policy_checkpoint=$(iclwam_q "$STAGE2A_CKPT")")
    fi
    cmd+=("${EXTRA_ARGS[@]}")
    printf '\nStage2%s: mode=%s, world=%s, per-rank batch=%s, accumulation=%s, effective batch=%s, optimizer=%s\n' \
        "$stage" "$mode" "$world" "$BATCH_SIZE" "$accum" "$GLOBAL_BATCH_SIZE" "$sharding"
    iclwam_run "${cmd[@]}"
}
