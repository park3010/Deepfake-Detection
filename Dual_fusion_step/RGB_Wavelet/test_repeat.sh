#!/usr/bin/env bash
set -euo pipefail

# =========================================
# RGB + Wavelet Dual-Stream Cross-Attention
# repeated external test runner
# - tests v1, v2, ...
# - saves each result CSV separately
# =========================================

GPU="0"

BASE_CKPT_DIR="/home/oem/deepfake/Ourmethod/Dual_fusion_step/RGB_Wavelet/_ckpt/atten"
BASE_RESULT_DIR="/home/oem/deepfake/Ourmethod/Dual_fusion_step/RGB_Wavelet/_result/atten"

# RGB_CKPT="/home/oem/deepfake/Ourmethod/Dual_fusion_step/_convnext_ckpt/convnext-tiny_best.pth"

FUSION="cross_attention"
CKPT_NAME="best_dual_cross_attention.pth"

VERSIONS=(v1 v2)

echo "========================================="
echo " Dual-stream repeated test start"
echo "========================================="
echo "GPU             : ${GPU}"
echo "BASE_CKPT_DIR   : ${BASE_CKPT_DIR}"
echo "BASE_RESULT_DIR : ${BASE_RESULT_DIR}"
echo "FUSION          : ${FUSION}"
echo "CKPT_NAME       : ${CKPT_NAME}"
echo "VERSIONS        : ${VERSIONS[*]}"
echo "========================================="

for VER in "${VERSIONS[@]}"; do
    CKPT_PATH="${BASE_CKPT_DIR}/${VER}/${CKPT_NAME}"
    OUT_DIR="${BASE_RESULT_DIR}/${VER}"

    echo
    echo "-----------------------------------------"
    echo " Test ${VER}"
    echo " CKPT_PATH=${CKPT_PATH}"
    echo " OUT_DIR=${OUT_DIR}"
    echo "-----------------------------------------"

    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "[SKIP] checkpoint not found: ${CKPT_PATH}"
        continue
    fi

    mkdir -p "${OUT_DIR}"

    python -m RGB_Wavelet.test \
        --gpu "${GPU}" \
        --fusion "${FUSION}" \
        --checkpoint "${CKPT_PATH}" \
        --img-size 224 \
        --wavelet sym4 \
        --wavelet-level 2 \
        --wavelet-type swt \
        --subband ll_energy \
        --batch-size 32 \
        --threshold 0.5 \
        --csv "${OUT_DIR}" \

    echo "Finished ${VER}"
done

echo
echo "========================================="
echo " All tests completed."
echo "========================================="