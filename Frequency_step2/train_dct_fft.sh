#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/home/oem/deepfake/hdd"
COMP="raw"

# 최종 저장 루트 (요청하신 경로)
CKPT_ROOT="/home/oem/deepfake/Ourmethod/Frequency_step2/ckpt_dct_fft"

# cache는 케이스/반복이 달라도 재사용 가능하니 공용으로 두는 걸 추천
# CACHE_ROOT="/home/oem/deepfake/Ourmethod/Frequency_step2/ckpt_dct_fft/cache_dct_fft"

EPOCHS=2000
BS=16
LR=3e-4
GPU=2

# 반복(v1~v5)마다 seed를 바꿔서 "일관성(variance)"를 보게 하는 게 일반적
# (원하면 아래 SEEDS만 바꿔도 됨)
SEEDS=(101 202 303 404 505)

VAL_RATIO=0.2
PATIENCE=5
MONITOR="f1_binary"
R1=0.12
R2=0.35

# mkdir -p "${CKPT_ROOT}" "${CACHE_ROOT}"

run () {
  echo ""
  echo "============================================================"
  echo "$*"
  echo "============================================================"
  "$@"
}

# 공통 옵션(ckpt-dir, seed는 케이스/반복별로 넣음)
BASE_COMMON=( \
  --data-dir "${DATA_DIR}" \
  --compression "${COMP}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BS}" \
  --lr "${LR}" \
  --val-ratio "${VAL_RATIO}" \
  --patience "${PATIENCE}" \
  --monitor "${MONITOR}" \
  --mode train \
  --gpu "${GPU}" \
)

# -----------------------------
# 실행 헬퍼
# -----------------------------
# usage:
#   run_case <backbone_dir> <case_dir> <seed> <extra_args...>
# 저장 경로:
#   ${CKPT_ROOT}/${backbone_dir}/${case_dir}/v{idx}/
run_case () {
  local backbone_dir="$1"; shift
  local case_dir="$1"; shift
  local seed="$1"; shift

  local ckpt_dir="${CKPT_ROOT}/${backbone_dir}/${case_dir}"
  mkdir -p "${ckpt_dir}"

  run python train_dct_fft2.py \
    "${BASE_COMMON[@]}" \
    --seed "${seed}" \
    --ckpt-dir "${ckpt_dir}" \
    "$@"
}

# -----------------------------
# 케이스 정의
# -----------------------------
# RepLKNet-B (4 cases)
# 1) ycbcr_block_dct
# 2) global_dct_low
# 3) global_dct_mid
# 4) global_dct_high
#
# ConvNeXt-Tiny (7 cases)
# 1) ycbcr_block_dct
# 2) global_dct_low
# 3) global_dct_mid
# 4) global_dct_high
# 5) global_fft_low
# 6) global_fft_mid
# 7) global_fft_high

# -----------------------------
# 5회 반복 루프
# -----------------------------
for i in 1 2 3 4 5; do
  seed="${SEEDS[$((i-1))]}"

  # ---------- RepLKNet ----------
  # A-1) RepLKNet-B + DCT(block, YCbCr, 8x8)
  run_case "replknet" "ycbcr_block_dct/v${i}" "${seed}" \
    --backbone replknet_b \
    --stream dct --dct-mode block --freq-in ycbcr --block-energy ac

  # A-2) RepLKNet-B + DCT(global, YCbCr) + low
  run_case "replknet" "global_dct_low/v${i}" "${seed}" \
    --backbone replknet_b \
    --stream dct --dct-mode global --freq-in ycbcr --band low --r1 "${R1}" --r2 "${R2}" \
    --cache-dir "${CACHE_ROOT}"

  # A-3) RepLKNet-B + DCT(global, YCbCr) + mid
  run_case "replknet" "global_dct_mid/v${i}" "${seed}" \
    --backbone replknet_b \
    --stream dct --dct-mode global --freq-in ycbcr --band mid --r1 "${R1}" --r2 "${R2}" \
    --cache-dir "${CACHE_ROOT}"

  # A-4) RepLKNet-B + DCT(global, YCbCr) + high
  run_case "replknet" "global_dct_high/v${i}" "${seed}" \
    --backbone replknet_b \
    --stream dct --dct-mode global --freq-in ycbcr --band high --r1 "${R1}" --r2 "${R2}" \
    --cache-dir "${CACHE_ROOT}"


  # ---------- ConvNeXt ----------
  # B-1) ConvNeXt-Tiny + DCT(block, YCbCr, 8x8)
  run_case "convnext" "ycbcr_block_dct/v${i}" "${seed}" \
    --backbone convnext_tiny \
    --stream dct --dct-mode block --freq-in ycbcr --block-energy ac

  # B-2) ConvNeXt-Tiny + DCT(global, YCbCr) + low
  run_case "convnext" "global_dct_low/v${i}" "${seed}" \
    --backbone convnext_tiny \
    --stream dct --dct-mode global --freq-in ycbcr --band low --r1 "${R1}" --r2 "${R2}" \
    --cache-dir "${CACHE_ROOT}"

  # B-3) ConvNeXt-Tiny + DCT(global, YCbCr) + mid
  run_case "convnext" "global_dct_mid/v${i}" "${seed}" \
    --backbone convnext_tiny \
    --stream dct --dct-mode global --freq-in ycbcr --band mid --r1 "${R1}" --r2 "${R2}" \
    --cache-dir "${CACHE_ROOT}"

  # B-4) ConvNeXt-Tiny + DCT(global, YCbCr) + high
  run_case "convnext" "global_dct_high/v${i}" "${seed}" \
    --backbone convnext_tiny \
    --stream dct --dct-mode global --freq-in ycbcr --band high --r1 "${R1}" --r2 "${R2}" \
    --cache-dir "${CACHE_ROOT}"

  # B-5) ConvNeXt-Tiny + FFT(global, YCbCr) + low
  run_case "convnext" "global_fft_low/v${i}" "${seed}" \
    --backbone convnext_tiny \
    --stream fft --fft-mode global --freq-in ycbcr --band low --r1 "${R1}" --r2 "${R2}" \
    --cache-dir "${CACHE_ROOT}"

  # B-6) ConvNeXt-Tiny + FFT(global, YCbCr) + mid
  run_case "convnext" "global_fft_mid/v${i}" "${seed}" \
    --backbone convnext_tiny \
    --stream fft --fft-mode global --freq-in ycbcr --band mid --r1 "${R1}" --r2 "${R2}" \
    --cache-dir "${CACHE_ROOT}"

  # B-7) ConvNeXt-Tiny + FFT(global, YCbCr) + high
  run_case "convnext" "global_fft_high/v${i}" "${seed}" \
    --backbone convnext_tiny \
    --stream fft --fft-mode global --freq-in ycbcr --band high --r1 "${R1}" --r2 "${R2}" \
    --cache-dir "${CACHE_ROOT}"

done

echo ""
echo "✅ All 11 cases × 5 runs finished."