#!/usr/bin/env bash
set -euo pipefail

# ====== COMMON SETTINGS ======
GPU="0"
SCRIPT="test_wavelet2.py"

IMG=224
BS=32
THRESH=0.5

BACKBONE="resnet50"
STREAM="wavelet"

# wavelet config
WAVELET="sym4"
LEVEL="2"
WTYPE="swt"
SUBBAND="ll_energy"

# train ckpt root
CKPT_ROOT="/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoint_wavelet2"

# test result root
TEST_ROOT="/home/oem/deepfake/Ourmethod/Frequency_step2/test_csv/wavelet"
# =============================

run_repeat_test () {
  local MODEL_ROOT="$1"   # ex) convnext
  local N_RUNS="$2"       # ex) 4

  local EXP_NAME="${WAVELET}_level${LEVEL}_${WTYPE}_${SUBBAND}"

  for RUN_NO in $(seq 1 "${N_RUNS}"); do
    local V="v${RUN_NO}"

    # 학습 때 저장한 체크포인트 경로
    local CKPT_DIR="${CKPT_ROOT}/${MODEL_ROOT}/${EXP_NAME}/${V}"
    local CKPT_FILE="${CKPT_DIR}/best_${BACKBONE}_${STREAM}.pth"

    # 테스트 결과 저장 폴더
    local OUT_DIR="${TEST_ROOT}/${MODEL_ROOT}/${EXP_NAME}/${V}"
    mkdir -p "${OUT_DIR}"

    if [[ ! -f "${CKPT_FILE}" ]]; then
      echo "[SKIP] checkpoint not found: ${CKPT_FILE}"
      continue
    fi

    echo "============================================================"
    echo "TEST: ${MODEL_ROOT}/${EXP_NAME}/${V}"
    echo "CKPT : ${CKPT_FILE}"
    echo "OUT  : ${OUT_DIR}"
    echo "============================================================"

    python3 "${SCRIPT}" \
      --gpu "${GPU}" \
      --backbone "${BACKBONE}" \
      --checkpoint "${CKPT_FILE}" \
      --img-size "${IMG}" \
      --stream "${STREAM}" \
      --wavelet "${WAVELET}" \
      --wavelet-level "${LEVEL}" \
      --wavelet-type "${WTYPE}" \
      --subband "${SUBBAND}" \
      --batch-size "${BS}" \
      --threshold "${THRESH}" \
      --csv "${OUT_DIR}"
  done
}

# convnext 4회 반복 학습된 가중치 테스트
run_repeat_test "resnet" 4