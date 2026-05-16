#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Main-Aux Tri / Multi-stream one-exp runner
#
# Usage:
#   bash run_tri_main_aux_one_exp_2seed_train_test.sh rgb_wavelet_dct 0
#   bash run_tri_main_aux_one_exp_2seed_train_test.sh rgb_wavelet_semantic 1
#   bash run_tri_main_aux_one_exp_2seed_train_test.sh rgb_dct_semantic 2
#   bash run_tri_main_aux_one_exp_2seed_train_test.sh wavelet_dct_semantic 3
#   bash run_tri_main_aux_one_exp_2seed_train_test.sh rgb_wavelet_dct_semantic 4
#
# Args:
#   $1 = experiment tag
#   $2 = GPU id
# =========================================

EXP="${1:?Usage: bash $0 <exp_tag> <gpu_id>}"
GPU="${2:-0}"

TRAIN_SCRIPT="train.py"
TEST_SCRIPT="test.py"

DATA_DIR="/home/oem/deepfake/hdd"
COMP="raw"

ROOT="/home/oem/deepfake/Ourmethod/Tri_stream"

SEEDS=(42 52)
VERSIONS=(v1 v2)

# ---------- train settings ----------
EPOCHS=2000
TRAIN_BS=16
LR=3e-4
PATIENCE=5
MONITOR="f1_binary"

# ---------- test settings ----------
TEST_BS=32
THRESHOLD=0.5

# ---------- Wavelet settings ----------
WAVELET="sym4"
WAVELET_LEVEL=2
WAVELET_TYPE="swt"
SUBBAND="ll_energy"

# ---------- DCT settings ----------
DCT_MODE="block"
FREQ_IN="ycbcr"
BLOCK_ENERGY="ac"

# ---------- Semantic settings ----------
CLIP_BACKBONE="openai/clip-vit-base-patch32"

# ---------- pretrained single-stream checkpoints ----------
# TODO: 실제 best checkpoint 경로로 수정
RGB_CKPT="/home/oem/deepfake/Ourmethod/Dual_fusion_step/_convnext_ckpt/convnext-tiny_best.pth"

WAVELET_CKPT="/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoint_wavelet2/resnet/sym4/sym4_level2_swt_ll_energy/best_resnet50_wavelet.pth"

DCT_CKPT="/home/oem/deepfake/Ourmethod/Tri_stream/models/convnext_tiny_dct_ycbcr_dctblock_fftglobal_bandall_r0.12-0.35_blkEac_inch3_seed101_best.pth"

# 1: freeze pretrained branches and train fusion/attention/classifier only
# 0: fine-tune all loaded branches
FREEZE_BRANCHES=1

# =========================================
# Select one experiment
# Main/Aux는 train_tri_stream.py 내부에서 자동 결정:
#   Main = rgb, wavelet
#   Aux  = dct, semantic
# =========================================
case "${EXP}" in
    rgb_wavelet_dct)
        STREAMS="rgb,wavelet,dct"
        TAG="rgb_wavelet_dct"
        ;;
    rgb_wavelet_semantic)
        STREAMS="rgb,wavelet,semantic"
        TAG="rgb_wavelet_semantic"
        ;;
    rgb_dct_semantic)
        STREAMS="rgb,dct,semantic"
        TAG="rgb_dct_semantic"
        ;;
    wavelet_dct_semantic)
        STREAMS="wavelet,dct,semantic"
        TAG="wavelet_dct_semantic"
        ;;
    rgb_wavelet_dct_semantic)
        STREAMS="rgb,wavelet,dct,semantic"
        TAG="rgb_wavelet_dct_semantic"
        ;;
    *)
        echo "[ERROR] Unknown EXP: ${EXP}"
        echo ""
        echo "Available experiments:"
        echo "  rgb_wavelet_dct"
        echo "  rgb_wavelet_semantic"
        echo "  rgb_dct_semantic"
        echo "  wavelet_dct_semantic"
        echo "  rgb_wavelet_dct_semantic"
        exit 1
        ;;
esac

CKPT_ROOT="${ROOT}/_ckpt/${TAG}"
RESULT_ROOT="${ROOT}/_result/${TAG}"
CKPT_NAME="best_tri_${TAG}.pth"

echo "========================================="
echo " Main-Aux Tri/Multi-stream one-exp runner"
echo "========================================="
echo "EXP          : ${EXP}"
echo "GPU          : ${GPU}"
echo "STREAMS      : ${STREAMS}"
echo "MAIN         : rgb/wavelet if included"
echo "AUX          : dct/semantic if included"
echo "DATA_DIR     : ${DATA_DIR}"
echo "ROOT         : ${ROOT}"
echo "SEEDS        : ${SEEDS[*]}"
echo "VERSIONS     : ${VERSIONS[*]}"
echo "WAVELET      : ${WAVELET}-L${WAVELET_LEVEL}-${WAVELET_TYPE}-${SUBBAND}"
echo "DCT          : ${DCT_MODE}, ${FREQ_IN}, ${BLOCK_ENERGY}"
echo "CLIP         : ${CLIP_BACKBONE}"
echo "RGB_CKPT     : ${RGB_CKPT}"
echo "WAVELET_CKPT : ${WAVELET_CKPT}"
echo "DCT_CKPT     : ${DCT_CKPT}"
echo "FREEZE       : ${FREEZE_BRANCHES}"
echo "========================================="

# =====================================================
# Build branch checkpoint args only for streams included
# =====================================================
BRANCH_CKPT_ARGS=()

if [[ "${STREAMS}" == *"rgb"* ]]; then
    BRANCH_CKPT_ARGS+=(--rgb-ckpt "${RGB_CKPT}")
fi

if [[ "${STREAMS}" == *"wavelet"* ]]; then
    BRANCH_CKPT_ARGS+=(--wavelet-ckpt "${WAVELET_CKPT}")
fi

if [[ "${STREAMS}" == *"dct"* ]]; then
    BRANCH_CKPT_ARGS+=(--dct-ckpt "${DCT_CKPT}")
fi

FREEZE_ARGS=()
if [[ "${FREEZE_BRANCHES}" -eq 1 ]]; then
    if [[ "${STREAMS}" == *"rgb"* ]]; then
        FREEZE_ARGS+=(--freeze-rgb)
    fi
    if [[ "${STREAMS}" == *"wavelet"* ]]; then
        FREEZE_ARGS+=(--freeze-wavelet)
    fi
    if [[ "${STREAMS}" == *"dct"* ]]; then
        FREEZE_ARGS+=(--freeze-dct)
    fi
    if [[ "${STREAMS}" == *"semantic"* ]]; then
        FREEZE_ARGS+=(--freeze-semantic)
    fi
fi

# =====================================================
# 1) TRAIN: 2 seeds
# =====================================================
# for i in "${!SEEDS[@]}"; do
#     SEED="${SEEDS[$i]}"
#     VER="${VERSIONS[$i]}"
#     CKPT_DIR="${CKPT_ROOT}/${VER}"

#     mkdir -p "${CKPT_DIR}"

#     echo ""
#     echo "-----------------------------------------"
#     echo " Training ${TAG} / ${VER} / seed=${SEED}"
#     echo " CKPT_DIR=${CKPT_DIR}"
#     echo "-----------------------------------------"

#     python "${TRAIN_SCRIPT}" \
#       --gpu "${GPU}" \
#       --data-dir "${DATA_DIR}" \
#       --compression "${COMP}" \
#       --streams "${STREAMS}" \
#       --seed "${SEED}" \
#       --epochs "${EPOCHS}" \
#       --batch-size "${TRAIN_BS}" \
#       --lr "${LR}" \
#       --patience "${PATIENCE}" \
#       --monitor "${MONITOR}" \
#       --wavelet "${WAVELET}" \
#       --wavelet-level "${WAVELET_LEVEL}" \
#       --wavelet-type "${WAVELET_TYPE}" \
#       --subband "${SUBBAND}" \
#       --dct-mode "${DCT_MODE}" \
#       --freq-in "${FREQ_IN}" \
#       --block-energy "${BLOCK_ENERGY}" \
#       --clip-backbone "${CLIP_BACKBONE}" \
#       --checkpoint "${CKPT_DIR}" \
#       "${BRANCH_CKPT_ARGS[@]}" \
#       "${FREEZE_ARGS[@]}"

#     echo "✅ Finished training ${TAG} / ${VER}"
# done

# =====================================================
# 2) TEST: 2 seeds
# =====================================================
for VER in "${VERSIONS[@]}"; do
    CKPT_PATH="${CKPT_ROOT}/${VER}/${CKPT_NAME}"
    OUT_DIR="${RESULT_ROOT}/${VER}"

    mkdir -p "${OUT_DIR}"

    echo ""
    echo "-----------------------------------------"
    echo " Testing ${TAG} / ${VER}"
    echo " CKPT_PATH=${CKPT_PATH}"
    echo " OUT_DIR=${OUT_DIR}"
    echo "-----------------------------------------"

    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "[ERROR] checkpoint not found: ${CKPT_PATH}"
        exit 1
    fi

    python "${TEST_SCRIPT}" \
      --gpu "${GPU}" \
      --streams "${STREAMS}" \
      --checkpoint "${CKPT_PATH}" \
      --batch-size "${TEST_BS}" \
      --threshold "${THRESHOLD}" \
      --wavelet "${WAVELET}" \
      --wavelet-level "${WAVELET_LEVEL}" \
      --wavelet-type "${WAVELET_TYPE}" \
      --subband "${SUBBAND}" \
      --dct-mode "${DCT_MODE}" \
      --freq-in "${FREQ_IN}" \
      --block-energy "${BLOCK_ENERGY}" \
      --clip-backbone "${CLIP_BACKBONE}" \
      --csv "${OUT_DIR}"

    echo "✅ Finished testing ${TAG} / ${VER}"
done

echo ""
echo "========================================="
echo " ✅ Finished ${EXP}"
echo "========================================="