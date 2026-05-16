#!/usr/bin/env bash
set -euo pipefail

# =========================================
# RGB + Wavelet Dual-Stream (MLP Fusion)
# 5-seed training runner
# =========================================

GPU="0"
DATA_DIR="/home/oem/deepfake/hdd"
BASE_CKPT_DIR="/home/oem/deepfake/Ourmethod/Dual_fusion_step/RGB_Wavelet/_ckpt/atten"

RGB_CKPT="/home/oem/deepfake/Ourmethod/Dual_fusion_step/_convnext_ckpt/convnext-tiny_best.pth"
WAVELET_CKPT="/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoint_wavelet2/resnet/sym4/sym4_level2_swt_ll_energy/best_resnet50_wavelet.pth"

SEEDS=(42 52 123 777 2026)
VERSIONS=(v1 v2 v3 v4 v5)

echo "========================================="
echo " Dual-stream training start (MLP fusion) "
echo "========================================="
echo "GPU          : ${GPU}"
echo "DATA_DIR     : ${DATA_DIR}"
echo "RGB_CKPT     : ${RGB_CKPT}"
echo "WAVELET_CKPT : ${WAVELET_CKPT}"
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

    python -m RGB_Wavelet.train \
        --gpu "${GPU}" \
        --data-dir "${DATA_DIR}" \
        --seed "${SEED}" \
        --checkpoint "${CKPT_DIR}" \
        --fusion cross_attention \
        --wavelet sym4 \
        --wavelet-level 2 \
        --wavelet-type swt \
        --subband ll_energy \
        --rgb-ckpt "${RGB_CKPT}" \
        --wavelet-ckpt "${WAVELET_CKPT}" \
        --freeze-rgb \
        --freeze-wavelet

    echo "Finished ${VER} (seed=${SEED})"
done

echo
echo "========================================="
echo " All 5 runs completed successfully."
echo "========================================="