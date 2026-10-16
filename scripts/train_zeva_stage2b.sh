#!/usr/bin/env bash
set -euo pipefail

# Zeva -> FastWAM Stage-2B (PIM adapter only).
# This launcher deliberately restores the released Zeva adapter regime:
#   * frozen FastWAM / CTE / Stage2A policy-injection modules
#   * train CausalPromptEncoder + prefix_project + scalar gate only
#   * 500 optimization steps by default
#   * base LR 5e-5 (encoder/projector receive the existing 5x multiplier)
#   * PIM context/support dropout is consumed by the formal prepared dataset
# Command-line overrides remain last and therefore can replace these defaults.

ROOT="/data/share/1919650160032350208/zjj/ICL-WAM"
DATA="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0"
OLD_DATA="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0_lerobotv30/local/robotwin2.0_old"
ART="$OLD_DATA/zeva_stage2_artifacts"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/data/share/1919650160032350208/zjj/fastwam/checkpoints}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

export GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
WORLD_SIZE="${#GPU_ARRAY[@]}"

export BASE_CKPT="$ROOT/checkpoints/fastwam_ckpt.pt"
export CTE_CKPT="$ROOT/runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt"
export PHASE_CACHE="$OLD_DATA/zeva_phase_effect_cache_v4"
export BEHAVIOR_BANK="$ART/behavior_bank.pt"
export READOUT_CACHE="$ART/initial_readouts.pt"
export RETRIEVAL_HEAD="$ART/static_retrieval_head.pt"
export STATIC_CONTEXT_CACHE="$ART/static_episode_context.pt"
export PIM_CACHE="$ART/prepared_pim_v1"

# Zeva-aligned defaults.  Keep these short: on expert imitation data a long
# Stage2B run can learn the shortcut g->0 because the frozen policy already
# explains the target action without PIM.
STAGE2B_MAX_STEPS="${STAGE2B_MAX_STEPS:-500}"
STAGE2B_LR="${STAGE2B_LR:-5e-5}"

if [[ -z "${STAGE2A_CKPT:-}" ]]; then
    echo "[ERROR] STAGE2A_CKPT must point to the NEW exact-ZeVA Stage2A addon." >&2
    exit 1
fi
export STAGE2A_CKPT

case "$STAGE2A_CKPT" in
    *"/2026-09-28_19-38-51/"*|*"/robotwin_eval_export/step_"*"_addon.pt")
        echo "[ERROR] Old pre-exact-ZeVA Stage2A checkpoint is not compatible: $STAGE2A_CKPT" >&2
        exit 1
        ;;
esac

export RUN_DIR="${RUN_DIR:-$ROOT/runs/robotwin_zeva_fastwam_stage2b_zeva_aligned/$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$RUN_DIR"

check_file() {
    local name="$1" path="$2"
    [[ -f "$path" ]] || { echo "[ERROR] missing $name: $path" >&2; exit 1; }
    echo "[OK] $name: $path"
}
check_dir() {
    local name="$1" path="$2"
    [[ -d "$path" ]] || { echo "[ERROR] missing $name: $path" >&2; exit 1; }
    echo "[OK] $name: $path"
}

check_file BASE_CKPT "$BASE_CKPT"
check_file CTE_CKPT "$CTE_CKPT"
check_file STAGE2A_CKPT "$STAGE2A_CKPT"
check_dir PHASE_CACHE "$PHASE_CACHE"
check_file PHASE_CACHE_MANIFEST "$PHASE_CACHE/manifest.json"
check_file PHASE_CACHE_INDEX "$PHASE_CACHE/episode_index.json"
check_file BEHAVIOR_BANK "$BEHAVIOR_BANK"
check_file READOUT_CACHE "$READOUT_CACHE"
check_file RETRIEVAL_HEAD "$RETRIEVAL_HEAD"
check_file STATIC_CONTEXT_CACHE "$STATIC_CONTEXT_CACHE"
check_dir PIM_CACHE "$PIM_CACHE"

python -m py_compile \
    "$ROOT/scripts/train_zeva_fastwam_prepared.py" \
    "$ROOT/src/fastwam/zeva/pim_training.py" \
    "$ROOT/src/fastwam/zeva/attempt_protocol.py" \
    "$ROOT/src/fastwam/trainer.py" \
    "$ROOT/src/fastwam/models/wan22/fastwam.py" \
    "$ROOT/src/fastwam/models/wan22/mot.py" \
    "$ROOT/src/fastwam/zeva/behavior_prefix_adapter.py"

if ! grep -q "_prepend_exact_zeva_behavior_token_to_video_stream" "$ROOT/src/fastwam/models/wan22/fastwam.py"; then
    echo "[ERROR] exact-ZeVA main-stream Behavior Prefix patch is missing." >&2
    exit 1
fi
if ! grep -q "query_has_context" "$ROOT/src/fastwam/models/wan22/mot.py"; then
    echo "[ERROR] safe all-False cross-attention mask fix is missing." >&2
    exit 1
fi

cat <<EOF
============================================================
 Zeva -> FastWAM Stage-2B
------------------------------------------------------------
 GPUs                 : $GPU_IDS (world=$WORLD_SIZE)
 Stage2A checkpoint   : $STAGE2A_CKPT
 max_steps            : $STAGE2B_MAX_STEPS
 base learning rate   : $STAGE2B_LR
 PIM context dropout  : config model.zeva.prompt.pim_context_dropout
 PIM support dropout  : config model.zeva.prompt.pim_support_dropout
 run dir              : $RUN_DIR
============================================================
EOF

# Defaults are inserted before "$@" so an explicit user override still wins.
exec bash "$ROOT/scripts/train_zeva_fastwam_prepared.sh" \
    model.zeva.training_stage=pim_adapter \
    model.zeva.policy_checkpoint="$STAGE2A_CKPT" \
    resume=null \
    max_steps="$STAGE2B_MAX_STEPS" \
    learning_rate="$STAGE2B_LR" \
    "$@"
