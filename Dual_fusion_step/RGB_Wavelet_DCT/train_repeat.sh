#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Wavelet + DCT Dual-Stream (MLP Fusion)
# 5-seed training runner
# =========================================

GPU="1"
DATA_DIR="/home/oem/deepfake/hdd"

MAIN_CKPT="/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoint_wavelet2/resnet/sym4/sym4_level2_swt_ll_energy/best_resnet50_wavelet.pth"
BASE_CKPT_DIR="/home/oem/deepfake/Ourmethod/Dual_fusion_step/RGB_Wavelet_DCT/_ckpt/wavelet/atten"

SEEDS=(42 52 123 777 2026)
VERSIONS=(v1 v2 v3 v4 v5)

echo "========================================="
echo "GPU        : ${GPU}"
echo "DATA_DIR   : ${DATA_DIR}"
echo "MAIN_CKPT  : ${MAIN_CKPT}"
echo "SAVE_ROOT  : ${BASE_CKPT_DIR}"
echo "========================================="

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    VER="${VERSIONS[$i]}"
    CKPT_DIR="${BASE_CKPT_DIR}/${VER}"

    echo
    echo "-----------------------------------------"
    echo " Run ${VER} | seed=${SEED}"
    echo " CKPT_DIR=${CKPT_DIR}"
    echo "-----------------------------------------"

    mkdir -p "${CKPT_DIR}"

    python -m RGB_Wavelet_DCT.train \
      --gpu "${GPU}" \
      --data-dir "${DATA_DIR}" \
      --seed "${SEED}" \
      --mode wavelet_dct \
      --fusion cross_attention \
      --wavelet sym4 \
      --wavelet-level 2 \
      --wavelet-type swt \
      --subband ll_energy \
      --main-ckpt "${MAIN_CKPT}" \
      --freeze-main \
      --resnet-pretrained-dct \
      --checkpoint "${CKPT_DIR}"

    echo "Finished ${VER} (seed=${SEED})"
done

echo
echo "========================================="
echo " All 5 runs completed successfully."
echo "========================================="