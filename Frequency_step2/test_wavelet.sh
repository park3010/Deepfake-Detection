#!/usr/bin/env bash
set -euo pipefail

# ========================
# EDIT THESE
# ========================
GPU="0"
BACKBONE="convnextv2_tiny"  # resnet50 convnextv2_tiny  
STREAM="wavelet"

# eval script path
EVAL_SCRIPT="test_wavelet2.py"

# train ckpt root (조합별 폴더에 best_*.pth가 들어있어야 함)
# (예: .../checkpoint_wavelet2/resnet/<W>/<W>_level<L>_<T>_<S>/best_resnet50_wavelet.pth)
TRAIN_CKPT_ROOT="/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoint_wavelet2/convnext"

# test csv root (요청 경로)
TEST_CSV_ROOT="/home/oem/deepfake/Ourmethod/Frequency_step2/test_csv/wavelet/convnext"

IMG=224
BATCH=32
THRESH=0.5

# optional flags
USE_CONVNEXT_CBAM=0     # convnextv2_tiny일 때만 의미
USE_WAVELET_GRAY=0
USE_STRICT=0            # 1이면 --strict (load strict=True)
USE_PRETRAINED=0        # 1이면 --pretrained (ResNet ImageNet pretrained로 모델 생성)

SKIP_DONE=1             # 1이면 이미 결과 CSV가 존재하면 스킵
# ========================

# ========================
# 어디부터 재개할지 여기서 제어
# - haar까지 했으면: WAVELETS=(sym4 db4 db8)
# - 처음부터면:       WAVELETS=(haar sym4 db4 db8)
# ========================
WAVELETS=(haar sym4 db4 db8)
LEVELS=(1 2)
TYPES=(dwt swt)
SUBBANDS=(ll high ll_energy)

mkdir -p "${TEST_CSV_ROOT}"

# optional flags (공통)
EXTRA_FLAGS=()
if [[ "${BACKBONE}" == "convnextv2_tiny" && "${USE_CONVNEXT_CBAM}" == "1" ]]; then
  EXTRA_FLAGS+=("--convnext-cbam")
fi
if [[ "${USE_WAVELET_GRAY}" == "1" ]]; then
  EXTRA_FLAGS+=("--wavelet-gray")
fi
if [[ "${USE_STRICT}" == "1" ]]; then
  EXTRA_FLAGS+=("--strict")
fi
if [[ "${USE_PRETRAINED}" == "1" ]]; then
  EXTRA_FLAGS+=("--pretrained")
fi

for W in "${WAVELETS[@]}"; do
  for L in "${LEVELS[@]}"; do
    for T in "${TYPES[@]}"; do
      for S in "${SUBBANDS[@]}"; do

        # ---- train ckpt 폴더(네 기존 train run shell 구조) ----
        RUN_DIR="${TRAIN_CKPT_ROOT}/${W}/${W}_level${L}_${T}_${S}"
        BEST_CKPT="${RUN_DIR}/best_${BACKBONE}_${STREAM}.pth"

        if [[ ! -f "${BEST_CKPT}" ]]; then
          echo "[SKIP] missing ckpt: ${BEST_CKPT}"
          continue
        fi

        # ---- test csv 저장 폴더 (조합별로 디렉터리 생성) ----
        OUT_DIR="${TEST_CSV_ROOT}/${W}/${W}_level${L}_${T}_${S}"
        mkdir -p "${OUT_DIR}"

        # eval 코드가 만드는 csv 파일명(tag 기반)을 정확히 예측하려면 tag 규칙을 그대로 맞춰야 함.
        # 다만, 스킵을 위해 "OUT_DIR 안에 *_results.csv 존재"로 충분히 처리 가능.
        if [[ "${SKIP_DONE}" == "1" ]]; then
          if compgen -G "${OUT_DIR}/*_results.csv" > /dev/null; then
            echo "[SKIP] already exists: ${OUT_DIR}/*_results.csv"
            continue
          fi
        fi

        echo "============================================================"
        echo "EVAL: ${BACKBONE} | ${STREAM} | ${W}_level${L}_${T}_${S}"
        echo "CKPT: ${BEST_CKPT}"
        echo "OUT : ${OUT_DIR}"
        echo "============================================================"

        python3 "${EVAL_SCRIPT}" \
          --gpu "${GPU}" \
          --backbone "${BACKBONE}" \
          --checkpoint "${BEST_CKPT}" \
          --img-size "${IMG}" \
          --stream "${STREAM}" \
          --wavelet "${W}" --wavelet-level "${L}" --wavelet-type "${T}" \
          --subband "${S}" \
          --batch-size "${BATCH}" \
          --threshold "${THRESH}" \
          --csv "${OUT_DIR}" \
          "${EXTRA_FLAGS[@]}"

      done
    done
  done
done