#!/usr/bin/env bash
# Internal helpers for the independent Zeva-main observed-effect workflow.
TRAIN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$TRAIN_SCRIPT_DIR/../.." && pwd)}"
if [[ -n "${ICLWAM_TRAIN_ENV:-}" ]]; then
    [[ -f "$ICLWAM_TRAIN_ENV" ]] || { echo "Missing ICLWAM_TRAIN_ENV: $ICLWAM_TRAIN_ENV" >&2; exit 2; }
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
iclwam_positive_integer() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || iclwam_die "$1 must be a positive integer (got '$2')."; }
iclwam_q() {
    local value="$1"
    value="${value//\\/\\\\}"; value="${value//\"/\\\"}"
    printf '"%s"' "$value"
}
iclwam_init() {
    DRY_RUN="${DRY_RUN:-0}"; EXTRA_ARGS=()
    local arg key
    while (($#)); do
        case "$1" in
            --dry-run) DRY_RUN=1; shift ;;
            -h|--help)
                printf '%s\n\n' "${DESCRIPTION:-Zeva-main offline training}"
                printf 'Usage: bash %s [--dry-run] [Hydra overrides...]\n' "${BASH_SOURCE[1]##*/}"
                printf '%s\n' 'See scripts/train/README.md; old Ego workflow is scripts/train_ego/.' \
                    '--dry-run prints shell-quoted commands without launching Python, checking GPU/data, or writing artifacts.'
                exit 0 ;;
            --) shift; EXTRA_ARGS+=("$@"); break ;;
            --*) iclwam_die "Unknown launcher option: $1" ;;
            *) EXTRA_ARGS+=("$1"); shift ;;
        esac
    done
    [[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || iclwam_die 'DRY_RUN must be 0 or 1.'
    for arg in "${EXTRA_ARGS[@]}"; do
        [[ "$arg" == *=* ]] || iclwam_die "Expected Hydra key=value: $arg"
        key="${arg%%=*}"; key="${key#+}"; key="${key#+}"; key="${key#~}"
        case "$key" in
            task|ckpt|output_dir|resume|foundation_mode|foundation_mode_settings*|max_steps|num_epochs|learning_rate|zeva_foundation_learning_rate|batch_size|gradient_accumulation_steps|zeva_optimizer_sharding|mixed_precision|device|overwrite|seed|data|data.train|data.val|data.train.dataset_dirs|data.train.pretrained_norm_stats|data.train.semantic_task_map_path|data.train.text_embedding_cache_dir|data.train.val_set_proportion|model|model.action_dit_pretrained_path|model.zeva|model.zeva.*|cte_latent_cache|cte_latent_cache.*)
                iclwam_die "Workflow-controlled override '$key': set its variable in _env.sh so every stage uses one main lineage." ;;
        esac
    done
}
iclwam_no_extra() { ((${#EXTRA_ARGS[@]} == 0)) || iclwam_die 'This stage has no Hydra overrides; use its environment variables.'; }
iclwam_run() { printf '\n+ '; printf '%q ' "$@"; printf '\n'; [[ "$DRY_RUN" == 1 ]] || "$@"; }
iclwam_require_file() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    local path; for path in "$@"; do [[ -f "$path" ]] || iclwam_die "Required file missing: $path"; done
}
iclwam_require_directory() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    local path; for path in "$@"; do [[ -d "$path" ]] || iclwam_die "Required directory missing: $path"; done
}
iclwam_require_success() {
    [[ "$DRY_RUN" == 1 || -f "$SUCCESS_MANIFEST" ]] && return 0
    printf 'Missing SUCCESS_MANIFEST: %s\n' "$SUCCESS_MANIFEST" >&2
    printf '%s\n' 'Record real collection outcomes first (the command below reads embedded outcomes; it never assumes success):' >&2
    printf '  %q ' "$PYTHON_BIN" >&2
    printf '%q ' "$REPO_ROOT/scripts/build_zeva_success_manifest.py" --dataset-root "$DATA_ROOT" --output "$SUCCESS_MANIFEST" >&2
    printf '\n%s\n' 'For an external outcome table add --outcomes-json /absolute/real_outcomes.json. See README for an explicitly documented success-only collection declaration.' >&2
    iclwam_die 'Unknown collection outcomes cannot enter the main success-demonstration curriculum.'
}
iclwam_require_data() {
    iclwam_require_success
    iclwam_require_directory "$DATA_ROOT"
    iclwam_require_file "$STATS_PATH" "$SEMANTIC_TASK_MAP"
}
iclwam_expect_new() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    local path; for path in "$@"; do
        [[ ! -e "$path" && ! -L "$path" ]] || iclwam_die "Output already exists: $path. Start at a later stage or choose a new RUN_ROOT; resume training only with an explicit resume input."
    done
}
iclwam_single_process() {
    [[ "${WORLD_SIZE:-1}" == 1 ]] || iclwam_die 'The launcher starts its own processes; do not nest it inside torchrun.'
    [[ "$NODE_RANK" == 0 ]] || iclwam_die 'Run shared preparation once on node 0.'
    [[ "$NNODES" == 1 ]] || iclwam_die 'Preparation uses one node; set NNODES=1 for stages 01-09.'
}
iclwam_data_args() {
    DATA_ARGS=(
        "data.train.dataset_dirs=[$(iclwam_q "$DATA_ROOT")]"
        "data.train.pretrained_norm_stats=$(iclwam_q "$STATS_PATH")"
        "data.train.semantic_task_map_path=$(iclwam_q "$SEMANTIC_TASK_MAP")"
        "data.train.text_embedding_cache_dir=$(iclwam_q "$TEXT_CACHE")"
        'data.train.val_set_proportion=0.0' 'data.val=null'
        "model.zeva.offline.success_manifest_path=$(iclwam_q "$SUCCESS_MANIFEST")"
        "++model.zeva.offline.validation_fraction=$VALIDATION_FRACTION"
        "++model.zeva.offline.split_seed=$SPLIT_SEED" "seed=$TRAIN_SEED"
    )
}
iclwam_artifact_args() {
    ARTIFACT_ARGS=(
        "ckpt=$(iclwam_q "$BASE_CKPT")"
        "model.action_dit_pretrained_path=$(iclwam_q "$ACTION_DIT_CKPT")"
        "model.zeva.cte.checkpoint=$(iclwam_q "$CTE_CKPT")"
        "model.zeva.cte.latent_cache_path=$(iclwam_q "$LATENT_CACHE")"
        "model.zeva.cache.path=$(iclwam_q "$PHASE_CACHE")"
        "model.zeva.task_context.bank_path=$(iclwam_q "$BEHAVIOR_BANK")"
        "model.zeva.task_context.readout_cache_path=$(iclwam_q "$READOUT_CACHE")"
        "model.zeva.task_context.retrieval_checkpoint=$(iclwam_q "$RETRIEVAL_HEAD")"
        "model.zeva.task_context.static_episode_cache_path=$(iclwam_q "$STATIC_CONTEXT_CACHE")"
    )
}
iclwam_distributed() {
    local ids="$1" requested="$2" nodes="$3" gpu seen=,
    local -a visible_ids
    [[ "${WORLD_SIZE:-1}" == 1 ]] || iclwam_die 'Do not nest this launcher inside torchrun.'
    [[ -n "$ids" && "$ids" != -1 && "$ids" != *, && "$ids" != ,* && "$ids" != *,,* && "$ids" != *[[:space:]]* ]] || iclwam_die 'GPU_IDS must be a nonempty comma-separated device list.'
    IFS=, read -r -a visible_ids <<< "$ids"
    for gpu in "${visible_ids[@]}"; do
        [[ "$seen" != *",$gpu,"* ]] || iclwam_die "Duplicate GPU device: $gpu"
        seen+="$gpu,"
    done
    NPROC_ACTUAL="${requested:-${#visible_ids[@]}}"
    iclwam_positive_integer NPROC_PER_NODE "$NPROC_ACTUAL"
    iclwam_positive_integer NNODES "$nodes"
    ((NPROC_ACTUAL <= ${#visible_ids[@]})) || iclwam_die 'NPROC_PER_NODE exceeds visible devices.'
    [[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] && ((NODE_RANK < nodes)) || iclwam_die 'NODE_RANK must be in [0, NNODES).'
    WORLD_RANKS=$((NPROC_ACTUAL * nodes))
    TORCH_LAUNCH=(env "CUDA_VISIBLE_DEVICES=$ids" "$PYTHON_BIN" -m torch.distributed.run "--nproc-per-node=$NPROC_ACTUAL")
    if ((nodes == 1)); then TORCH_LAUNCH+=(--standalone)
    else
        [[ -n "$MASTER_ADDR" ]] || iclwam_die 'Multi-node training requires MASTER_ADDR.'
        iclwam_positive_integer MASTER_PORT "$MASTER_PORT"
        ((MASTER_PORT <= 65535)) || iclwam_die 'MASTER_PORT must be <= 65535.'
        TORCH_LAUNCH+=("--nnodes=$nodes" "--node-rank=$NODE_RANK" "--master-addr=$MASTER_ADDR" "--master-port=$MASTER_PORT")
    fi
}
iclwam_preparation_launcher() {
    iclwam_single_process
    iclwam_distributed "$PREP_GPU_IDS" "${PREP_NPROC_PER_NODE:-}" 1
}
iclwam_stage2() {
    local stage="$1" mode steps output resume task batch global accum sharding lr foundation_lr parent_filename
    local -a cmd
    [[ "$STAGE2A_FOUNDATION_MODE" == frozen || "$STAGE2A_FOUNDATION_MODE" == joint ]] || iclwam_die 'STAGE2A_FOUNDATION_MODE must be frozen or joint.'
    [[ "$STAGE2B_FOUNDATION_MODE" == frozen || "$STAGE2B_FOUNDATION_MODE" == joint ]] || iclwam_die 'STAGE2B_FOUNDATION_MODE must be frozen or joint.'
    iclwam_positive_integer STAGE2A_STEPS "$STAGE2A_STEPS"
    iclwam_positive_integer STAGE2B_STEPS "$STAGE2B_STEPS"
    printf -v parent_filename 'step_%06d_addon.pt' "$STAGE2A_STEPS"
    STAGE2A_CKPT="${STAGE2A_CKPT:-$STAGE2A_RUN_DIR/checkpoints/weights/$parent_filename}"
    if [[ "$stage" == A ]]; then
        mode="$STAGE2A_FOUNDATION_MODE"; steps="$STAGE2A_STEPS"; output="$STAGE2A_RUN_DIR"; resume="$STAGE2A_RESUME"
        batch="$STAGE2A_BATCH_SIZE"; global="$STAGE2A_GLOBAL_BATCH_SIZE"; accum="${STAGE2A_GRAD_ACCUM_STEPS:-}"
        sharding="$STAGE2A_OPTIMIZER_SHARDING"; lr="$STAGE2A_LEARNING_RATE"; foundation_lr="$STAGE2A_FOUNDATION_LEARNING_RATE"
        task=robotwin_zeva_main_policy_3cam_384
    elif [[ "$stage" == B ]]; then
        mode="$STAGE2B_FOUNDATION_MODE"; steps="$STAGE2B_STEPS"; output="$STAGE2B_RUN_DIR"; resume="$STAGE2B_RESUME"
        batch="$STAGE2B_BATCH_SIZE"; global="$STAGE2B_GLOBAL_BATCH_SIZE"; accum="${STAGE2B_GRAD_ACCUM_STEPS:-}"
        sharding="$STAGE2B_OPTIMIZER_SHARDING"; lr="$STAGE2B_LEARNING_RATE"; foundation_lr="$STAGE2B_FOUNDATION_LEARNING_RATE"
        task=robotwin_zeva_main_pim_3cam_384
    else iclwam_die 'Internal Stage2 stage must be A or B.'; fi
    iclwam_distributed "$GPU_IDS" "${NPROC_PER_NODE:-}" "$NNODES"
    iclwam_positive_integer BATCH_SIZE "$batch"; iclwam_positive_integer GLOBAL_BATCH_SIZE "$global"
    ((global % (WORLD_RANKS * batch) == 0)) || iclwam_die 'Global batch must be divisible by world size * per-rank batch.'
    accum="${accum:-$((global / (WORLD_RANKS * batch)))}"
    iclwam_positive_integer GRAD_ACCUM_STEPS "$accum"
    ((WORLD_RANKS * batch * accum == global)) || iclwam_die 'Requested accumulation changes the effective global batch.'
    [[ "$sharding" == none || "$sharding" == zero1 ]] || iclwam_die 'Optimizer sharding must be none or zero1.'
    [[ "$sharding" != zero1 || "$WORLD_RANKS" -gt 1 ]] || iclwam_die 'zero1 requires at least two ranks.'
    iclwam_require_data
    iclwam_require_directory "$DIFFSYNTH_MODEL_BASE_PATH" "$TEXT_CACHE" "$PHASE_CACHE" "$PIM_CACHE"
    iclwam_require_file "$BASE_CKPT" "$ACTION_DIT_CKPT" "$CTE_CKPT" "$BEHAVIOR_BANK" "$READOUT_CACHE" "$RETRIEVAL_HEAD" "$STATIC_CONTEXT_CACHE"
    if [[ -n "$resume" ]]; then iclwam_require_directory "$resume"
    else
        iclwam_expect_new "$output"
        if [[ "$stage" == B ]]; then iclwam_require_file "$STAGE2A_CKPT"; fi
    fi
    iclwam_data_args; iclwam_artifact_args
    cmd=("${TORCH_LAUNCH[@]}" scripts/train_zeva_fastwam_prepared.py "task=$task"
        "${DATA_ARGS[@]}" "${ARTIFACT_ARGS[@]}"
        "model.zeva.prepared_stage2.pim_cache_path=$(iclwam_q "$PIM_CACHE")"
        'model.zeva.prepared_stage2.validation_pim_cache_path=null'
        "output_dir=$(iclwam_q "$output")" "foundation_mode=$mode" "max_steps=$steps"
        "learning_rate=$lr" "zeva_foundation_learning_rate=$foundation_lr" 'mixed_precision=bf16'
        "batch_size=$batch" "gradient_accumulation_steps=$accum" "zeva_optimizer_sharding=$sharding" "num_workers=$NUM_WORKERS")
    if [[ -n "$resume" ]]; then cmd+=("resume=$(iclwam_q "$resume")")
    elif [[ "$stage" == B ]]; then cmd+=("model.zeva.policy_checkpoint=$(iclwam_q "$STAGE2A_CKPT")"); fi
    cmd+=("${EXTRA_ARGS[@]}")
    printf '\nMain Stage2%s: mode=%s, world=%s, per-rank batch=%s, accumulation=%s, effective batch=%s, optimizer=%s\n' \
        "$stage" "$mode" "$WORLD_RANKS" "$batch" "$accum" "$global" "$sharding"
    iclwam_run "${cmd[@]}"
}
