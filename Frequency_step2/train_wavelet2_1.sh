#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# FF++ Wavelet stream batch runner (db4, db8 only)
# - Saves checkpoints as:
#   checkpoint/<wavelet>/<wavelet>_level<level>_<type>_<subband>/
# ----------------------------

# ====== EDIT THESE ======
DATA_DIR="/home/oem/deepfake/hdd"     # <-- FF++ root (original_sequences/, manipulated_sequences/)
COMP="raw"                            # raw / c23 / c40 등
GPU="1"

SCRIPT="train_wavelet2_1.py"
STREAM="wavelet"

BACKBONE="convnextv2_tiny"            # resnet50 or convnextv2_tiny
USE_CONVNEXT_CBAM=0                   # 1: add --convnext-cbam, 0: no
USE_WAVELET_GRAY=0                    # 1: add --wavelet-gray, 0: no

IMG=224
EPOCHS=2000
BS=16
LR=3e-4
PATIENCE=5
MIN_DELTA=0.0

CHECKPOINT_ROOT="/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoint_wavelet2/convnext"
SKIP_DONE=1                           # 1이면 best ckpt 있으면 해당 조합 스킵
# ========================

WAVELETS=(haar sym4 db4 db8)
LEVELS=(1 2)
TYPES=(dwt swt)
SUBBANDS=(ll high ll_energy)

mkdir -p "${CHECKPOINT_ROOT}"

for W in "${WAVELETS[@]}"; do
  mkdir -p "${CHECKPOINT_ROOT}/${W}"

  for L in "${LEVELS[@]}"; do
    for T in "${TYPES[@]}"; do
      for S in "${SUBBANDS[@]}"; do
        RUN_DIR="${CHECKPOINT_ROOT}/${W}/${W}_level${L}_${T}_${S}"
        mkdir -p "${RUN_DIR}"

        # 네 코드가 만드는 best 파일명 (조합 폴더마다 따로 존재)
        BEST_FILE="${RUN_DIR}/best_${BACKBONE}_${STREAM}.pth"

        if [[ "${SKIP_DONE}" == "1" && -f "${BEST_FILE}" ]]; then
          echo "[SKIP] already exists: ${BEST_FILE}"
          continue
        fi

        # 이전 실행 중 남았을 수 있는 tmp 파일 정리 (선택)
        rm -f "${RUN_DIR}"/*.tmp* 2>/dev/null || true

        echo "============================================================"
        echo "RUN: ${W}_level${L}_${T}_${S}"
        echo "CKPT_DIR: ${RUN_DIR}"
        echo "============================================================"

        # optional flags
        EXTRA_FLAGS=()
        if [[ "${BACKBONE}" == "convnextv2_tiny" && "${USE_CONVNEXT_CBAM}" == "1" ]]; then
          EXTRA_FLAGS+=("--convnext-cbam")
        fi
        if [[ "${USE_WAVELET_GRAY}" == "1" ]]; then
          EXTRA_FLAGS+=("--wavelet-gray")
        fi

        python3 "${SCRIPT}" \
          --mode train \
          --data-dir "${DATA_DIR}" --compression "${COMP}" \
          --stream "${STREAM}" \
          --backbone "${BACKBONE}" \
          --wavelet "${W}" --wavelet-level "${L}" --wavelet-type "${T}" \
          --subband "${S}" \
          --img-size "${IMG}" \
          --epochs "${EPOCHS}" --batch-size "${BS}" --lr "${LR}" \
          --patience "${PATIENCE}" --min-delta "${MIN_DELTA}" \
          --checkpoint "${RUN_DIR}" \
          --gpu "${GPU}" \
          "${EXTRA_FLAGS[@]}"

      done
    done
  done
done