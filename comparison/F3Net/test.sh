#!/usr/bin/env bash
set -euo pipefail

echo "=== F3Net Both test ==="
python test.py \
  --gpu 1 \
  --checkpoint /home/oem/deepfake/Ourmethod/comparison/_ckpt/f3net/both/f3net_both_best.pth \
  --f3net-root /home/oem/deepfake/Ourmethod/comparison/F3Net \
  --f3net-mode Both \
  --csv /home/oem/deepfake/Ourmethod/comparison/_result/f3net/both

echo "=== F3Net FAD test ==="
python test.py \
  --gpu 1 \
  --checkpoint /home/oem/deepfake/Ourmethod/comparison/_ckpt/f3net/fad/f3net_fad_best.pth \
  --f3net-root /home/oem/deepfake/Ourmethod/comparison/F3Net \
  --f3net-mode FAD \
  --csv /home/oem/deepfake/Ourmethod/comparison/_result/f3net/fad

echo "=== F3Net LFS test ==="
python test.py \
  --gpu 1 \
  --checkpoint /home/oem/deepfake/Ourmethod/comparison/_ckpt/f3net/lfs/f3net_lfs_best.pth \
  --f3net-root /home/oem/deepfake/Ourmethod/comparison/F3Net \
  --f3net-mode LFS \
  --csv /home/oem/deepfake/Ourmethod/comparison/_result/f3net/lfs

echo "=== All tests completed ==="