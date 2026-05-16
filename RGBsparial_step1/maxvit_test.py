#!/usr/bin/env python3
"""
Video-level DeepFake inference with **MaxViT** (RGB only).

· 각 비디오 디렉터리(프레임들) → 프레임별 예측 → 평균 확률로 라벨 결정  
· 입력 프레임은 224×224 RGB, 0-1 스케일 후 (x – 0.5)/0.5 정규화  
· 결과 CSV(ID, label) 저장
"""

import os, sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# 0. 모듈 경로 & 모델 불러오기
# ──────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ROOT_SRC = "/home/oem/deepfake/Ourmethod/RGBsparial_step1"   # MaxViT 소스 경로
sys.path.append(ROOT_SRC)

from maxvit.maxvit import MaxViT          # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# 1. 장치 & 모델 로드
# ──────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] using device → {DEVICE}")

# 필요하면 use_cbam=True 로 변경
model = MaxViT(num_classes=2, use_cbam=False)
ckpt_path = (
    "/home/oem/deepfake/Ourmethod/RGBsparial_step1/checkpoints/"
    "maxvit_best.pth"                    # ← 학습 스크립트에서 저장된 이름으로 수정
)
state = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.to(DEVICE).eval()
print(f"[INFO] checkpoint loaded from {ckpt_path}")

# ──────────────────────────────────────────────────────────────────────
# 2. 테스트 프레임 디렉터리
# ──────────────────────────────────────────────────────────────────────
TEST_DIR = Path("/home/oem/deepfake/hdd/test_sample_frames_5")
print(f"[INFO] processing videos in {TEST_DIR.resolve()}")

def preprocess_pil(pil: Image.Image) -> torch.Tensor:
    """PIL(RGB) → Tensor(1,3,224,224) normalized to −1 ~ 1."""
    pil = pil.resize((224, 224), Image.LANCZOS)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)   # 1×3×H×W

# ──────────────────────────────────────────────────────────────────────
# 3. 비디오별 추론
# ──────────────────────────────────────────────────────────────────────
results = []
for video_dir in tqdm(sorted(TEST_DIR.iterdir()), desc="Videos"):
    if not video_dir.is_dir():
        continue

    vid_id = video_dir.name + ".mp4"
    frame_paths = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not frame_paths:
        results.append((vid_id, 1))            # 프레임 없음 → fake
        continue

    probs = []
    for fp in tqdm(frame_paths, desc=f"  {vid_id}", leave=False, ncols=80):
        pil = Image.open(fp).convert("RGB")
        inp = preprocess_pil(pil).to(DEVICE)

        with torch.no_grad():
            logits = model(inp)
            p_fake = torch.softmax(logits, dim=1)[0, 1].item()
        probs.append(p_fake)

    avg_p = float(np.mean(probs))
    label = 0 if avg_p > 0.5 else 1            # 0=real, 1=fake
    results.append((vid_id, label))

# ──────────────────────────────────────────────────────────────────────
# 4. CSV 저장
# ──────────────────────────────────────────────────────────────────────
out_csv = Path("/home/oem/deepfake/Ourmethod/RGBsparial_step1/test_result/test_maxvit.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results, columns=["ID", "label"]).to_csv(out_csv, index=False)
print(f"✓ submission saved → {out_csv}")
