#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Wavelet + DCT Dual-Stream Cross-Attention
# repeated external test runner
# - tests v1, v2, ...
# - saves each result CSV separately
# =========================================

GPU="2"

BASE_CKPT_DIR="/home/oem/deepfake/Ourmethod/Dual_fusion_step/RGB_Wavelet_DCT/_ckpt/wavelet/mlp"
BASE_RESULT_DIR="/home/oem/deepfake/Ourmethod/Dual_fusion_step/RGB_Wavelet_DCT/_result/wavelet/mlp"

FUSION="mlp"
MODE="wavelet_dct"
CKPT_NAME="best_wavelet_dct_mlp.pth"

# 실제로 2번 반복 학습한 경우
VERSIONS=(v1 v2)

# 만약 위 train shell처럼 5-seed 전체를 테스트하려면 아래로 변경
# VERSIONS=(v1 v2 v3 v4 v5)

echo "========================================="
echo " Wavelet + DCT repeated test start"
echo "========================================="
echo "GPU             : ${GPU}"
echo "BASE_CKPT_DIR   : ${BASE_CKPT_DIR}"
echo "BASE_RESULT_DIR : ${BASE_RESULT_DIR}"
echo "MODE            : ${MODE}"
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
        echo "[ERROR] checkpoint not found: ${CKPT_PATH}"
        exit 1
    fi

    mkdir -p "${OUT_DIR}"

    python -m RGB_Wavelet_DCT.test \
      --gpu "${GPU}" \
      --mode "${MODE}" \
      --fusion "${FUSION}" \
      --checkpoint "${CKPT_PATH}" \
      --wavelet sym4 \
      --wavelet-level 2 \
      --wavelet-type swt \
      --subband ll_energy \
      --dct-mode gray3 \
      --batch-size 32 \
      --threshold 0.5 \
      --csv "${OUT_DIR}"

    echo "Finished test ${VER}"
done

echo
echo "========================================="
echo " All repeated tests completed successfully."
echo "========================================="