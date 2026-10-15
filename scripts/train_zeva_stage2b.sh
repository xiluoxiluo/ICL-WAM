#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ICL-WAM
# Zeva -> FastWAM Stage-2B
# PIM Adapter Training
#
# Current exact-Zeva topology:
#
#   Frozen FastWAM
#        +
#   NEW Stage2A policy-injection checkpoint
#        ↓
#   Stage2B pim_adapter
#
# IMPORTANT:
#   - Do NOT use old 2026-09-28 Stage2A checkpoints.
#   - Do NOT use robotwin_eval_export/step_*_addon.pt.
#   - STAGE2A_CKPT must be explicitly specified.
#
# Smoke example:
#
#   GPU_IDS=0 \
#   STAGE2A_CKPT=/data/share/1919650160032350208/zjj/ICL-WAM/runs/zeva_aligned_stage2a_smoke/checkpoints/weights/step_000002_addon.pt \
#   RUN_DIR=/data/share/1919650160032350208/zjj/ICL-WAM/runs/zeva_aligned_stage2b_smoke \
#   bash scripts/train_zeva_stage2b.sh \
#       max_steps=2 \
#       save_every=1 \
#       log_every=1 \
#       num_workers=0
#
# Formal example:
#
#   GPU_IDS=0,1 \
#   STAGE2A_CKPT=/path/to/NEW_STAGE2A/checkpoints/weights/step_005000_addon.pt \
#   bash scripts/train_zeva_stage2b.sh
#
# ============================================================


# ============================================================
# 0. Fixed project paths
# ============================================================

ROOT="/data/share/1919650160032350208/zjj/ICL-WAM"

# 当前训练配置的数据根目录仍然保留
DATA="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0"

# 实际已经验证过的 Zeva prepared artifacts 所在目录
OLD_DATA="/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0_lerobotv30/local/robotwin2.0_old"

ART="$OLD_DATA/zeva_stage2_artifacts"

EVAL_EXPORT="$ROOT/robotwin_eval_export"


cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/data/share/1919650160032350208/zjj/fastwam/checkpoints}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"


# ============================================================
# 1. GPU
# ============================================================

export GPU_IDS="${GPU_IDS:-0,1}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
WORLD_SIZE="${#GPU_ARRAY[@]}"


# ============================================================
# 2. Frozen FastWAM checkpoint
# ============================================================

export BASE_CKPT="$ROOT/checkpoints/fastwam_ckpt.pt"


# ============================================================
# 3. Frozen CTE checkpoint
# ============================================================

export CTE_CKPT="$ROOT/runs/robotwin_zeva_fastwam_3cam_384/2026-09-23_20-32-07/cte.pt"


# ============================================================
# 4. Prepared Stage-2 artifacts
# ============================================================

# 已验证 Phase / Effect cache
export PHASE_CACHE="$OLD_DATA/zeva_phase_effect_cache_v4"

# 全部使用同一套 Stage-2 artifacts，避免 hash/source identity 混用
export BEHAVIOR_BANK="$ART/behavior_bank.pt"
export READOUT_CACHE="$ART/initial_readouts.pt"
export RETRIEVAL_HEAD="$ART/static_retrieval_head.pt"
export STATIC_CONTEXT_CACHE="$ART/static_episode_context.pt"
export PIM_CACHE="$ART/prepared_pim_v1"


# ============================================================
# 5. Stage-2A checkpoint
# ============================================================
#
# 必须显式指定。
#
# 原因：
# 当前 exact-ZeVA patch 已改变：
#
#   Behavior Prefix:
#       text-context token
#             ↓
#       FastWAM main MoT stream token
#
# 因此旧 Stage2A addon 不兼容新的正式 Stage2B。
# ============================================================

if [[ -z "${STAGE2A_CKPT:-}" ]]; then

    echo
    echo "============================================================"
    echo "[ERROR] STAGE2A_CKPT 未指定"
    echo "============================================================"
    echo
    echo "必须显式指定 NEW exact-ZeVA Stage2A checkpoint。"
    echo
    echo "Stage2B smoke 示例："
    echo
    echo "export STAGE2A_CKPT=\"$ROOT/runs/zeva_aligned_stage2a_smoke/checkpoints/weights/step_000002_addon.pt\""
    echo
    echo "然后运行："
    echo
    echo "bash scripts/train_zeva_stage2b.sh \\"
    echo "  max_steps=2 \\"
    echo "  save_every=1 \\"
    echo "  log_every=1 \\"
    echo "  num_workers=0"
    echo

    exit 1
fi

export STAGE2A_CKPT


# ============================================================
# 6. Protect against known OLD checkpoints
# ============================================================

case "$STAGE2A_CKPT" in

    *"/2026-09-28_19-38-51/"*)
        echo
        echo "[ERROR] 检测到旧 Stage2A checkpoint："
        echo "$STAGE2A_CKPT"
        echo
        echo "该 checkpoint 属于旧 Behavior Prefix 注入拓扑，禁止用于新 Stage2B。"
        exit 1
        ;;

    *"/robotwin_eval_export/step_"*"_addon.pt")
        echo
        echo "[ERROR] 检测到 robotwin_eval_export 中的旧 addon："
        echo "$STAGE2A_CKPT"
        echo
        echo "该 checkpoint 属于旧拓扑，禁止用于新 Stage2B。"
        exit 1
        ;;

esac


# ============================================================
# 7. Stage-2B run directory
# ============================================================

export RUN_DIR="${RUN_DIR:-$ROOT/runs/robotwin_zeva_fastwam_stage2b_zeva_aligned/$(date +%Y-%m-%d_%H-%M-%S)}"


# ============================================================
# 8. Validation helpers
# ============================================================

check_file() {

    local name="$1"
    local path="$2"

    if [[ -z "$path" ]]; then
        echo "[ERROR] $name path is empty"
        exit 1
    fi

    if [[ ! -f "$path" ]]; then
        echo "[ERROR] $name 不存在："
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
        echo "[ERROR] $name path is empty"
        exit 1
    fi

    if [[ ! -d "$path" ]]; then
        echo "[ERROR] $name 不存在："
        echo "        $path"
        exit 1
    fi

    echo "[OK] $name"
    echo "     $path"
}


# ============================================================
# 9. Fail-fast artifact validation
# ============================================================

echo
echo "============================================================"
echo " Zeva -> FastWAM Stage-2B artifact check"
echo "============================================================"
echo


check_file \
    "BASE_CKPT" \
    "$BASE_CKPT"


check_file \
    "CTE_CKPT" \
    "$CTE_CKPT"


check_dir \
    "PHASE_CACHE" \
    "$PHASE_CACHE"


check_file \
    "PHASE_CACHE/manifest.json" \
    "$PHASE_CACHE/manifest.json"


check_file \
    "PHASE_CACHE/episode_index.json" \
    "$PHASE_CACHE/episode_index.json"


check_file \
    "BEHAVIOR_BANK" \
    "$BEHAVIOR_BANK"


check_file \
    "READOUT_CACHE" \
    "$READOUT_CACHE"


check_file \
    "RETRIEVAL_HEAD" \
    "$RETRIEVAL_HEAD"


check_file \
    "STATIC_CONTEXT_CACHE" \
    "$STATIC_CONTEXT_CACHE"


check_dir \
    "PIM_CACHE" \
    "$PIM_CACHE"


check_file \
    "STAGE2A_CKPT" \
    "$STAGE2A_CKPT"


check_file \
    "train_zeva_fastwam_prepared.sh" \
    "$ROOT/scripts/train_zeva_fastwam_prepared.sh"


check_file \
    "train_zeva_fastwam_prepared.py" \
    "$ROOT/scripts/train_zeva_fastwam_prepared.py"


# ============================================================
# 10. Print resolved configuration
# ============================================================

echo
echo "============================================================"
echo " Zeva -> FastWAM Stage-2B"
echo "============================================================"

echo
echo "Project:"
echo "  ROOT                 = $ROOT"
echo "  DATA                 = $DATA"
echo "  OLD_DATA             = $OLD_DATA"
echo "  ART                  = $ART"

echo
echo "GPU:"
echo "  GPU_IDS              = $GPU_IDS"
echo "  WORLD_SIZE           = $WORLD_SIZE"

echo
echo "Frozen models:"
echo "  BASE_CKPT            = $BASE_CKPT"
echo "  CTE_CKPT             = $CTE_CKPT"

echo
echo "Prepared artifacts:"
echo "  PHASE_CACHE          = $PHASE_CACHE"
echo "  BEHAVIOR_BANK        = $BEHAVIOR_BANK"
echo "  READOUT_CACHE        = $READOUT_CACHE"
echo "  RETRIEVAL_HEAD       = $RETRIEVAL_HEAD"
echo "  STATIC_CONTEXT_CACHE = $STATIC_CONTEXT_CACHE"
echo "  PIM_CACHE            = $PIM_CACHE"

echo
echo "Stage2A initialization:"
echo "  STAGE2A_CKPT         = $STAGE2A_CKPT"

echo
echo "Stage2B:"
echo "  RUN_DIR              = $RUN_DIR"
echo "  training_stage       = pim_adapter"
echo "  resume               = null"

echo
echo "============================================================"
echo


# ============================================================
# 11. Create output
# ============================================================

mkdir -p "$RUN_DIR"


# ============================================================
# 12. Lightweight preflight
# ============================================================

bash -n \
    "$ROOT/scripts/train_zeva_fastwam_prepared.sh"


python -m py_compile \
    "$ROOT/scripts/train_zeva_fastwam_prepared.py" \
    "$ROOT/src/fastwam/trainer.py" \
    "$ROOT/src/fastwam/runtime.py" \
    "$ROOT/src/fastwam/models/wan22/fastwam.py" \
    "$ROOT/src/fastwam/models/wan22/mot.py" \
    "$ROOT/src/fastwam/zeva/behavior_prefix_adapter.py"


echo "[OK] Python/shell preflight passed."


# ============================================================
# 13. Verify exact-ZeVA main-stream patch exists
# ============================================================

if ! grep -q \
    "_prepend_exact_zeva_behavior_token_to_video_stream" \
    "$ROOT/src/fastwam/models/wan22/fastwam.py"
then

    echo
    echo "[ERROR] exact-ZeVA main-stream Behavior Prefix patch 未检测到。"
    echo "禁止使用旧 text-context injection topology 训练 Stage2B."
    exit 1
fi


# ============================================================
# 14. Verify safe all-false cross-attention fix exists
# ============================================================

if ! grep -q \
    "query_has_context" \
    "$ROOT/src/fastwam/models/wan22/mot.py"
then

    echo
    echo "[ERROR] mot.py 中未检测到 safe all-False context-mask 修复。"
    echo
    echo "当前 exact-ZeVA Behavior Prefix 的 text context mask 为全 False，"
    echo "缺少该修复可能导致 BF16 CUDA forward 出现 NaN。"
    exit 1
fi


echo "[OK] exact-ZeVA main-stream topology detected."
echo "[OK] safe cross-attention mask fix detected."


# ============================================================
# 15. Optional git diff syntax check
# ============================================================

if [[ -d "$ROOT/.git" ]]; then

    if ! git diff --check; then

        echo
        echo "[ERROR] git diff --check failed."
        exit 1

    fi

    echo "[OK] git diff --check passed."

fi


# ============================================================
# 16. Launch Stage-2B
# ============================================================

echo
echo "============================================================"
echo " Launching Stage-2B pim_adapter training"
echo "============================================================"
echo
echo "Initialization:"
echo
echo "  NEW Stage2A"
echo "       ↓"
echo "  CausalPromptEncoder"
echo "       +"
echo "  prefix_project"
echo "       +"
echo "  pim_gate"
echo
echo "FastWAM / CTE / Stage2A policy branch remain frozen."
echo
echo "============================================================"
echo


exec bash "$ROOT/scripts/train_zeva_fastwam_prepared.sh" \
    model.zeva.training_stage=pim_adapter \
    model.zeva.policy_checkpoint="$STAGE2A_CKPT" \
    resume=null \
    "$@"