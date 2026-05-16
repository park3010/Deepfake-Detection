#!/usr/bin/env bash
set -euo pipefail

# ====== COMMON SETTINGS ======
DATA_DIR="/home/oem/deepfake/hdd"
COMP="raw"
GPU="1"

SCRIPT="train_wavelet2_1.py"
STREAM="wavelet"

IMG=224
EPOCHS=2000
BS=16
LR=3e-4
PATIENCE=5
MIN_DELTA=0.0

CKPT_ROOT="/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoint_wavelet2"
SKIP_DONE=1
# =============================

# You can change these seeds if you want.
SEEDS=(52 123 777 2026)

run_repeat_train () {
  local BACKBONE="$1"
  local WAVELET="$2"
  local LEVEL="$3"
  local WTYPE="$4"
  local SUBBAND="$5"
  local MODEL_ROOT="$6"

  for IDX in "${!SEEDS[@]}"; do
    local RUN_NO=$((IDX + 1))
    local V="v${RUN_NO}"
    local SEED="${SEEDS[$IDX]}"

    RUN_DIR="${CKPT_ROOT}/${MODEL_ROOT}/${WAVELET}_level${LEVEL}_${WTYPE}_${SUBBAND}_1/${V}"
    mkdir -p "${RUN_DIR}"

    BEST_FILE="${RUN_DIR}/best_${BACKBONE}_${STREAM}.pth"

    if [[ "${SKIP_DONE}" == "1" && -f "${BEST_FILE}" ]]; then
      echo "[SKIP] already exists: ${BEST_FILE}"
      continue
    fi

    rm -f "${RUN_DIR}"/*.tmp* 2>/dev/null || true

    echo "============================================================"
    echo "RUN: ${MODEL_ROOT}/${WAVELET}_level${LEVEL}_${WTYPE}_${SUBBAND}/${V}"
    echo "BACKBONE: ${BACKBONE}"
    echo "SEED: ${SEED}"
    echo "CKPT_DIR: ${RUN_DIR}"
    echo "============================================================"

    EXTRA_FLAGS=()

    python3 "${SCRIPT}" \
      --mode train \
      --data-dir "${DATA_DIR}" \
      --compression "${COMP}" \
      --stream "${STREAM}" \
      --backbone "${BACKBONE}" \
      --wavelet "${WAVELET}" \
      --wavelet-level "${LEVEL}" \
      --wavelet-type "${WTYPE}" \
      --subband "${SUBBAND}" \
      --img-size "${IMG}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BS}" \
      --lr "${LR}" \
      --patience "${PATIENCE}" \
      --min-delta "${MIN_DELTA}" \
      --checkpoint "${RUN_DIR}" \
      --gpu "${GPU}" \
      --seed "${SEED}" \
      "${EXTRA_FLAGS[@]}"
  done
}


# 1) ResNet-50 best config: Sym4 - Level2 - SWT - LL+energy
# run_repeat_train \
#   "resnet50" \
#   "sym4" \
#   "2" \
#   "swt" \
#   "ll_energy" \
#   "resnet"


# 2) ConvNeXt-Tiny best config: Sym4 - Level1 - DWT - LL
run_repeat_train \
  "convnextv2_tiny" \
  "sym4" \
  "1" \
  "dwt" \
  "ll" \
  "convnext"
