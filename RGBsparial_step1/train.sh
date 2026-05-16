#!/usr/bin/env bash
set -euo pipefail

# =========================================
# RGB Single Stream - ResNet50 5-run runner
# =========================================

GPU="1"
MODEL="resnet50"
DATA_DIR="/home/oem/deepfake/hdd"
BASE_CKPT_DIR="/home/oem/deepfake/Ourmethod/RGBsparial_step1/checkpoints/resnet50"

SCRIPT="train_3090.py"

VERSIONS=(v1 v2 v3 v4 v5)

echo "========================================="
echo " RGB Single Stream ResNet50 Training"
echo "========================================="
echo "GPU           : ${GPU}"
echo "MODEL         : ${MODEL}"
echo "DATA_DIR      : ${DATA_DIR}"
echo "BASE_CKPT_DIR : ${BASE_CKPT_DIR}"
echo "VERSIONS      : ${VERSIONS[*]}"
echo "========================================="

for VER in "${VERSIONS[@]}"; do
    CKPT_DIR="${BASE_CKPT_DIR}/${VER}"

    echo ""
    echo "-----------------------------------------"
    echo " Start training ${MODEL} - ${VER}"
    echo " CKPT_DIR: ${CKPT_DIR}"
    echo "-----------------------------------------"

    CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
        --gpu "${GPU}" \
        --model "${MODEL}" \
        --data-dir "${DATA_DIR}" \
        --ckpt "${CKPT_DIR}"

    echo "-----------------------------------------"
    echo " Finished ${MODEL} - ${VER}"
    echo "-----------------------------------------"
done

echo ""
echo "========================================="
echo " All ResNet50 RGB training runs finished"
echo "========================================="