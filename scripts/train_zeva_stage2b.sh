#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ICL-WAM
# Zeva -> FastWAM Stage-2B (PIM Adapter) Training
#
# 默认：
#   - GPU: 0,1
#   - Stage: pim_adapter
#   - 使用已验证的 prepared Stage-2 artifacts
#   - 从 Stage-2A step_005000_addon.pt 初始化
#
# 推荐启动：
#
# STAGE2A_CKPT=/data/share/1919650160032350208/zjj/ICL-WAM/runs/robotwin_zeva_fastwam_stage2_3cam_384/2026-09-28_19-38-51/checkpoints/weights/step_005000_addon.pt \
# bash scripts/train_zeva_stage2b.sh
#
# 指定 GPU：
#
# GPU_IDS=0,1,2,3 \
# STAGE2A_CKPT=/path/to/step_005000_addon.pt \
# bash scripts/train_zeva_stage2b.sh
# ============================================================


# ------------------------------------------------------------
# Fixed project paths
# ------------------------------------------------------------

ROOT="/data/share/1919650160032350208/zjj/ICL-WAM"
DATA="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0"

cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="/data/share/1919650160032350208/zjj/fastwam/checkpoints"


# ------------------------------------------------------------
# GPU
# ------------------------------------------------------------

export GPU_IDS="${GPU_IDS:-0,1}"


# ------------------------------------------------------------
# Base FastWAM / CTE
# ------------------------------------------------------------

export BASE_CKPT="$ROOT/checkpoints/fastwam_ckpt.pt"

export CTE_CKPT="$ROOT/runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt"


# ------------------------------------------------------------
# Prepared Stage-2 artifacts
# ------------------------------------------------------------

export PHASE_CACHE="$DATA/zeva_phase_effect_cache_v4"

export ART="$DATA/zeva_stage2_artifacts"

export BEHAVIOR_BANK="$ART/behavior_bank.pt"
export READOUT_CACHE="$ART/initial_readouts.pt"
export RETRIEVAL_HEAD="$ART/static_retrieval_head.pt"
export STATIC_CONTEXT_CACHE="$ART/static_episode_context.pt"
export PIM_CACHE="$ART/prepared_pim_v1"


# ------------------------------------------------------------
# Stage-2A checkpoint
# ------------------------------------------------------------
#
# 优先使用用户显式指定的 STAGE2A_CKPT。
# 若未指定，则自动寻找最近的 step_005000_addon.pt。
# ------------------------------------------------------------

if [[ -z "${STAGE2A_CKPT:-}" ]]; then
    STAGE2A_CKPT="$(
        find "$ROOT/runs/robotwin_zeva_fastwam_stage2_3cam_384" \
            -type f \
            -path "*/checkpoints/weights/step_005000_addon.pt" \
            -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
    )"
fi

export STAGE2A_CKPT


# ------------------------------------------------------------
# Stage-2B run directory
# ------------------------------------------------------------

export RUN_DIR="${RUN_DIR:-$ROOT/runs/robotwin_zeva_fastwam_stage2b_3cam_384/$(date +%Y-%m-%d_%H-%M-%S)}"


# ------------------------------------------------------------
# Fail-fast validation
# ------------------------------------------------------------

check_file() {
    local name="$1"
    local path="$2"

    if [[ ! -f "$path" ]]; then
        echo "[ERROR] $name 不存在:"
        echo "        $path"
        exit 1
    fi
}


check_dir() {
    local name="$1"
    local path="$2"

    if [[ ! -d "$path" ]]; then
        echo "[ERROR] $name 不存在:"
        echo "        $path"
        exit 1
    fi
}


check_file "BASE_CKPT" "$BASE_CKPT"
check_file "CTE_CKPT" "$CTE_CKPT"

check_dir "PHASE_CACHE" "$PHASE_CACHE"

check_file "BEHAVIOR_BANK" "$BEHAVIOR_BANK"
check_file "READOUT_CACHE" "$READOUT_CACHE"
check_file "RETRIEVAL_HEAD" "$RETRIEVAL_HEAD"
check_file "STATIC_CONTEXT_CACHE" "$STATIC_CONTEXT_CACHE"

check_dir "PIM_CACHE" "$PIM_CACHE"


if [[ -z "${STAGE2A_CKPT:-}" ]]; then
    echo "[ERROR] 没有找到 Stage-2A step_005000_addon.pt"
    echo
    echo "请手动指定："
    echo
    echo "STAGE2A_CKPT=/完整路径/step_005000_addon.pt \\"
    echo "bash scripts/train_zeva_stage2b.sh"
    exit 1
fi

check_file "STAGE2A_CKPT" "$STAGE2A_CKPT"

check_file \
    "train_zeva_fastwam_prepared.sh" \
    "$ROOT/scripts/train_zeva_fastwam_prepared.sh"

check_file \
    "train_zeva_fastwam_prepared.py" \
    "$ROOT/scripts/train_zeva_fastwam_prepared.py"


# ------------------------------------------------------------
# Print resolved config
# ------------------------------------------------------------

echo "============================================================"
echo " Zeva -> FastWAM Stage-2B Training"
echo "============================================================"
echo "ROOT                 = $ROOT"
echo "DATA                 = $DATA"
echo "GPU_IDS              = $GPU_IDS"
echo
echo "BASE_CKPT            = $BASE_CKPT"
echo "CTE_CKPT             = $CTE_CKPT"
echo
echo "PHASE_CACHE          = $PHASE_CACHE"
echo "BEHAVIOR_BANK        = $BEHAVIOR_BANK"
echo "READOUT_CACHE        = $READOUT_CACHE"
echo "RETRIEVAL_HEAD       = $RETRIEVAL_HEAD"
echo "STATIC_CONTEXT_CACHE = $STATIC_CONTEXT_CACHE"
echo "PIM_CACHE            = $PIM_CACHE"
echo
echo "STAGE2A_CKPT         = $STAGE2A_CKPT"
echo "RUN_DIR              = $RUN_DIR"
echo
echo "training_stage       = pim_adapter"
echo "resume               = null"
echo "============================================================"


mkdir -p "$RUN_DIR"


# ------------------------------------------------------------
# Lightweight preflight
# ------------------------------------------------------------

bash -n "$ROOT/scripts/train_zeva_fastwam_prepared.sh"

python -m py_compile \
    "$ROOT/scripts/train_zeva_fastwam_prepared.py" \
    "$ROOT/src/fastwam/trainer.py" \
    "$ROOT/src/fastwam/models/wan22/fastwam.py" \
    "$ROOT/src/fastwam/zeva/behavior_prefix_adapter.py"


echo "[OK] Stage-2B preflight passed."
echo "[INFO] Launching pim_adapter training..."
echo


# ------------------------------------------------------------
# Launch Stage-2B
# ------------------------------------------------------------

exec bash "$ROOT/scripts/train_zeva_fastwam_prepared.sh" \
    model.zeva.training_stage=pim_adapter \
    model.zeva.policy_checkpoint="$STAGE2A_CKPT" \
    resume=null \
    "$@"