#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ICL-WAM
# Zeva-aligned FastWAM Stage-2 Prepared Training
#
# 用途：
#   Stage2A: policy_injection
#   Stage2B: pim_adapter（由外部 override）
#
# 当前 Zeva-aligned 版本：
#   - FastWAM frozen
#   - CTE frozen
#   - Behavior Prefix 注入 FastWAM main MoT stream
#   - 不修改 CTE / BIT / PIM / prepared cache 语义
# ============================================================


# ============================================================
# 0. Basic paths
# ============================================================

ROOT="/data/share/1919650160032350208/zjj/ICL-WAM"

# 当前 RoboTwin 数据配置所在位置
DATA="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0"

# 之前生成 Zeva Stage-2 artifacts 的真实目录
OLD_DATA="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0_lerobotv30/local/robotwin2.0_old"

# 已找到的 Stage-2 artifact 根目录
ART="$OLD_DATA/zeva_stage2_artifacts"

# 评测导出目录
EVAL_EXPORT="$ROOT/robotwin_eval_export"


cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/data/share/1919650160032350208/zjj/fastwam/checkpoints}"

# 出错时显示完整 Hydra traceback
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"


# ============================================================
# 1. GPU
# ============================================================

# 默认 2 GPU
# smoke test 时：
#   GPU_IDS=0 bash scripts/train_zeva_fastwam_prepared.sh ...
#
GPU_IDS="${GPU_IDS:-0,1}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
WORLD_SIZE="${#GPU_ARRAY[@]}"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"


# ============================================================
# 2. Helper functions
# ============================================================

check_file() {
    local name="$1"
    local path="$2"

    if [[ -z "$path" ]]; then
        echo "[ERROR] $name is empty"
        exit 1
    fi

    if [[ ! -f "$path" ]]; then
        echo "[ERROR] $name not found:"
        echo "        $path"
        exit 1
    fi

    echo "[OK] $name"
    echo "     $path"
}


check_dir() {
    local name="$1"
    local path="$2"

    if [[ -z "$path" ]]; then
        echo "[ERROR] $name is empty"
        exit 1
    fi

    if [[ ! -d "$path" ]]; then
        echo "[ERROR] $name not found:"
        echo "        $path"
        exit 1
    fi

    echo "[OK] $name"
    echo "     $path"
}


# ============================================================
# 3. Frozen FastWAM checkpoint
# ============================================================

# 优先使用训练仓 canonical checkpoint
BASE_CKPT="$ROOT/checkpoints/fastwam_ckpt.pt"

# 如果 canonical checkpoint 不存在，再退回 eval export
if [[ ! -f "$BASE_CKPT" ]]; then
    if [[ -f "$EVAL_EXPORT/fastwam_ckpt.pt" ]]; then
        BASE_CKPT="$EVAL_EXPORT/fastwam_ckpt.pt"
    fi
fi

export BASE_CKPT


# ============================================================
# 4. Frozen CTE checkpoint
# ============================================================

# 优先使用原始正式 CTE checkpoint
CTE_CKPT="$ROOT/runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt"

# 如果不存在，再使用 eval export 中的副本
if [[ ! -f "$CTE_CKPT" ]]; then
    if [[ -f "$EVAL_EXPORT/cte.pt" ]]; then
        CTE_CKPT="$EVAL_EXPORT/cte.pt"
    fi
fi

export CTE_CKPT


# ============================================================
# 5. Phase / Effect cache
# ============================================================

export PHASE_CACHE="$OLD_DATA/zeva_phase_effect_cache_v4"


# ============================================================
# 6. Static task-context artifacts
# ============================================================

# ------------------------------------------------------------
# Behavior Bank
#
# 最优先：
#   与 static_episode_context.pt 同一 ART 目录下的版本
#
# 如果旧 ART 目录里没有，再使用 robotwin_eval_export 副本。
# ------------------------------------------------------------

if [[ -f "$ART/behavior_bank.pt" ]]; then
    export BEHAVIOR_BANK="$ART/behavior_bank.pt"

elif [[ -f "$EVAL_EXPORT/behavior_bank.pt" ]]; then
    export BEHAVIOR_BANK="$EVAL_EXPORT/behavior_bank.pt"

else
    echo "[ERROR] behavior_bank.pt not found"
    echo "checked:"
    echo "  $ART/behavior_bank.pt"
    echo "  $EVAL_EXPORT/behavior_bank.pt"
    exit 1
fi


# ------------------------------------------------------------
# Initial readouts
# ------------------------------------------------------------

export READOUT_CACHE="$ART/initial_readouts.pt"


# ------------------------------------------------------------
# Static retrieval head
#
# 同样优先使用旧 ART 目录里的原始版本。
# ------------------------------------------------------------

if [[ -f "$ART/static_retrieval_head.pt" ]]; then
    export RETRIEVAL_HEAD="$ART/static_retrieval_head.pt"

elif [[ -f "$EVAL_EXPORT/static_retrieval_head.pt" ]]; then
    export RETRIEVAL_HEAD="$EVAL_EXPORT/static_retrieval_head.pt"

else
    echo "[ERROR] static_retrieval_head.pt not found"
    echo "checked:"
    echo "  $ART/static_retrieval_head.pt"
    echo "  $EVAL_EXPORT/static_retrieval_head.pt"
    exit 1
fi


# ------------------------------------------------------------
# Static episode context
# ------------------------------------------------------------

export STATIC_CONTEXT_CACHE="$ART/static_episode_context.pt"


# ============================================================
# 7. Prepared PIM
# ============================================================

export PIM_CACHE="$ART/prepared_pim_v1"


# ============================================================
# 8. Output run directory
# ============================================================

RUN_DIR="${RUN_DIR:-$ROOT/runs/robotwin_zeva_fastwam_stage2_3cam_384/$(date +%Y-%m-%d_%H-%M-%S)}"

export RUN_DIR

mkdir -p "$RUN_DIR"


# ============================================================
# 9. Print resolved paths
# ============================================================

echo
echo "============================================================"
echo " Zeva-aligned FastWAM Prepared Stage-2"
echo "============================================================"
echo
echo "ROOT                 = $ROOT"
echo "DATA                 = $DATA"
echo "OLD_DATA             = $OLD_DATA"
echo "ART                  = $ART"
echo "EVAL_EXPORT          = $EVAL_EXPORT"
echo
echo "GPU_IDS              = $GPU_IDS"
echo "WORLD_SIZE           = $WORLD_SIZE"
echo
echo "BASE_CKPT            = $BASE_CKPT"
echo "CTE_CKPT             = $CTE_CKPT"
echo
echo "PHASE_CACHE          = $PHASE_CACHE"
echo
echo "BEHAVIOR_BANK        = $BEHAVIOR_BANK"
echo "READOUT_CACHE        = $READOUT_CACHE"
echo "RETRIEVAL_HEAD       = $RETRIEVAL_HEAD"
echo "STATIC_CONTEXT_CACHE = $STATIC_CONTEXT_CACHE"
echo
echo "PIM_CACHE            = $PIM_CACHE"
echo
echo "RUN_DIR              = $RUN_DIR"
echo
echo "============================================================"
echo


# ============================================================
# 10. Fail-fast checks
# ============================================================

check_file "BASE_CKPT" "$BASE_CKPT"
check_file "CTE_CKPT" "$CTE_CKPT"

check_dir "PHASE_CACHE" "$PHASE_CACHE"

check_file "BEHAVIOR_BANK" "$BEHAVIOR_BANK"
check_file "READOUT_CACHE" "$READOUT_CACHE"
check_file "RETRIEVAL_HEAD" "$RETRIEVAL_HEAD"
check_file "STATIC_CONTEXT_CACHE" "$STATIC_CONTEXT_CACHE"

check_dir "PIM_CACHE" "$PIM_CACHE"


# ------------------------------------------------------------
# Phase cache completeness
# ------------------------------------------------------------

check_file \
    "PHASE_CACHE/manifest.json" \
    "$PHASE_CACHE/manifest.json"

check_file \
    "PHASE_CACHE/episode_index.json" \
    "$PHASE_CACHE/episode_index.json"


echo
echo "============================================================"
echo " All required Stage-2 artifacts exist."
echo "============================================================"
echo


# ============================================================
# 11. Important checkpoint warning
# ============================================================

echo "[INFO] Zeva-aligned topology is active."
echo "[INFO] Do NOT initialize Stage2A from old addon checkpoints."
echo
echo "[INFO] Old checkpoints that must NOT be used for new Stage2A:"
echo "       $EVAL_EXPORT/step_003000_addon.pt"
echo "       $EVAL_EXPORT/step_004000_addon.pt"
echo "       $EVAL_EXPORT/step_005000_addon.pt"
echo
echo "[INFO] Correct training chain:"
echo
echo "       Frozen FastWAM"
echo "             |"
echo "             v"
echo "       NEW Stage2A"
echo "             |"
echo "             v"
echo "       NEW Stage2B"
echo


# ============================================================
# 12. Python syntax preflight
# ============================================================

python -m py_compile \
    "$ROOT/scripts/train_zeva_fastwam_prepared.py" \
    "$ROOT/src/fastwam/trainer.py" \
    "$ROOT/src/fastwam/runtime.py" \
    "$ROOT/src/fastwam/models/wan22/fastwam.py" \
    "$ROOT/src/fastwam/zeva/behavior_prefix_adapter.py"

echo "[OK] Python preflight passed."
echo


# ============================================================
# 13. Launch
# ============================================================

echo "============================================================"
echo " Launching torchrun"
echo "============================================================"
echo

exec torchrun \
    --standalone \
    --nproc_per_node="$WORLD_SIZE" \
    scripts/train_zeva_fastwam_prepared.py \
    task=robotwin_zeva_fastwam_stage2_3cam_384 \
    output_dir="$RUN_DIR" \
    ++ckpt="$BASE_CKPT" \
    ++model.zeva.cte.checkpoint="$CTE_CKPT" \
    ++model.zeva.cache.path="$PHASE_CACHE" \
    ++model.zeva.task_context.bank_path="$BEHAVIOR_BANK" \
    ++model.zeva.task_context.readout_cache_path="$READOUT_CACHE" \
    ++model.zeva.task_context.retrieval_checkpoint="$RETRIEVAL_HEAD" \
    ++model.zeva.task_context.static_episode_cache_path="$STATIC_CONTEXT_CACHE" \
    ++model.zeva.prepared_stage2.pim_cache_path="$PIM_CACHE" \
    "$@"