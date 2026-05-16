#!/usr/bin/env bash
set -euo pipefail

# =========================================
# RGB Single Stream - ResNet50 5-run test runner
# =========================================

GPU="0"
MODEL="resnet50"

SCRIPT="test.py"

BASE_CKPT_DIR="/home/oem/deepfake/Ourmethod/RGBsparial_step1/checkpoints/resnet50"
BASE_RESULT_DIR="/home/oem/deepfake/Ourmethod/RGBsparial_step1/_results/resnet"

CKPT_NAME="resnet50_best.pth"

VERSIONS=(v1 v2 v3 v4 v5)

echo "========================================="
echo " RGB Single Stream ResNet50 Test"
echo "========================================="
echo "GPU             : ${GPU}"
echo "MODEL           : ${MODEL}"
echo "BASE_CKPT_DIR   : ${BASE_CKPT_DIR}"
echo "BASE_RESULT_DIR : ${BASE_RESULT_DIR}"
echo "CKPT_NAME       : ${CKPT_NAME}"
echo "VERSIONS        : ${VERSIONS[*]}"
echo "========================================="

for VER in "${VERSIONS[@]}"; do
    CKPT_PATH="${BASE_CKPT_DIR}/${VER}/${CKPT_NAME}"
    OUT_DIR="${BASE_RESULT_DIR}/${VER}"

    echo ""
    echo "-----------------------------------------"
    echo " Test ${MODEL} - ${VER}"
    echo " CKPT_PATH: ${CKPT_PATH}"
    echo " OUT_DIR  : ${OUT_DIR}"
    echo "-----------------------------------------"

    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "[ERROR] checkpoint not found: ${CKPT_PATH}"
        exit 1
    fi

    mkdir -p "${OUT_DIR}"

    CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
        --gpu "${GPU}" \
        --model "${MODEL}" \
        --checkpoint "${CKPT_PATH}" \
        --batch-size 32 \
        --threshold 0.5 \
        --csv "${OUT_DIR}"

    echo "-----------------------------------------"
    echo " Finished test ${MODEL} - ${VER}"
    echo "-----------------------------------------"
done

echo ""
echo "========================================="
echo " All ResNet50 RGB tests finished"
echo "========================================="