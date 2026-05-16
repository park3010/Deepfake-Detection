#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ─── 설정 ───────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 사용자 설정 ─────────────────────────────────────────
# 주파수 처리 방식 선택: 'fft' 또는 'dct'
FREQ_METHOD = 'fft'  # 변경하여 사용할 수 있습니다.

# ─── 경로 설정 ───────────────────────────────────────────
TEST_DIR   = Path("/home/oem/deepfake/hdd/test_sample_frames_5")
OUT_CSV    = Path("/home/oem/deepfake/Ourmethod/results/2model_submission.csv")
RGB_CKPT   = Path("/home/oem/deepfake/Ourmethod/RGBsparial_step1/checkpoints/hornet_base_ddp_best.pth")
FREQ_CKPT  = Path("/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoints/convnext_fft_1_best.pth")
MLP_CKPT   = Path("/home/oem/deepfake/Ourmethod/ensemble_results/mlp_fusion_2model_best.pth")

# ─── 로컬 임포트 ─────────────────────────────────────────
import sys
sys.path.append('/home/oem/deepfake/Ourmethod/Frequency_step2')
from Frequency_step2.models.convnextv2 import convnextv2_large
sys.path.append('/home/oem/deepfake/Ourmethod/RGBsparial_step1')
from RGBsparial_step1.hornet.hornet import hornet_base_gf

# ─── FFT / DCT 함수 ───────────────────────────────────────
def extract_fft(bgr: np.ndarray) -> np.ndarray:
    chans = []
    for ch in cv2.split(bgr):
        dft   = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        shift = np.fft.fftshift(dft)
        mag   = cv2.magnitude(shift[:,:,0], shift[:,:,1])
        chans.append((20 * np.log(mag + 1)).astype(np.float32))
    return np.stack(chans, axis=2)

def extract_dct(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    return cv2.dct(gray).astype(np.float32)

# ─── 베이스 모델 로드 ────────────────────────────────────
def load_base(kind, ckpt_path):
    sd = torch.load(ckpt_path, map_location=DEVICE)
    sd = {k.replace('model.',''): v for k,v in sd.items()}

    if kind == 'hornet':
        m = hornet_base_gf(num_classes=2)
    elif kind == 'freq':
        stem_w = next(v for k,v in sd.items() if k.endswith('downsample_layers.0.0.weight'))
        m = convnextv2_large(in_chans=stem_w.shape[1], num_classes=2, use_cbam=False)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE).eval()

# ─── Meta-MLP ─────────────────────────────────────────────
class MetaMLP(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

# ─── Confidence-Tuned 보정 ───────────────────────────────
def apply_confidence_tuning(mlp_probs: np.ndarray) -> np.ndarray:
    fake = mlp_probs[:,1]
    conf = 2 * np.abs(fake - 0.5)
    return np.where(fake < 0.5,
                    fake * conf + 0.5 * (1 - conf),
                    fake)

# ─── 테스트 루프 ─────────────────────────────────────────
def run_test():
    # 1) 모델 불러오기
    rgb_model  = load_base('hornet', RGB_CKPT)
    freq_model = load_base('freq',   FREQ_CKPT)

    # 2) MLP 로드
    mlp = MetaMLP(in_dim=4).to(DEVICE)
    mlp.load_state_dict(torch.load(MLP_CKPT, map_location=DEVICE))
    mlp.eval()

    # 3) 전처리
    resize_tf = T.Resize((224,224))
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    softmax = nn.Softmax(dim=1)

    results = []
    for vid_dir in tqdm(sorted(TEST_DIR.iterdir()), desc="Test videos"):
        if not vid_dir.is_dir(): continue
        vid = vid_dir.name + ".mp4"
        frame_scores = []

        for img_path in sorted(vid_dir.iterdir()):
            img = Image.open(img_path).convert("RGB")
            img = resize_tf(img)

            # — RGB 모델 예측
            inp_r = to_tensor(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): pr = softmax(rgb_model(inp_r))[0].cpu().numpy()
            torch.cuda.empty_cache()

            # — 주파수 모델 입력 생성
            arr      = np.array(img).astype(np.float32) / 255.0
            bgr      = arr[:,:,::-1]
            if FREQ_METHOD == 'fft':
                freq_map = extract_fft(bgr)              # H×W×3
            else:  # 'dct'
                bgr_uint8 = (bgr * 255).astype(np.uint8)
                dctm      = extract_dct(bgr_uint8)       # H×W
                freq_map  = np.stack([dctm]*3, axis=2)    # H×W×3

            freq_in = np.concatenate([bgr, freq_map], axis=2)  # H×W×6
            inp_f   = torch.from_numpy(freq_in.transpose(2,0,1)[None]).to(DEVICE)
            with torch.no_grad(): pf = softmax(freq_model(inp_f))[0].cpu().numpy()
            torch.cuda.empty_cache()

            # — MLP Fusion + Confidence 튜닝
            fusion   = np.concatenate([pr, pf])         # (4,)
            with torch.no_grad():
                mlpo  = softmax(mlp(torch.from_numpy(fusion).float().unsqueeze(0).to(DEVICE)))[0].cpu().numpy()
            score    = apply_confidence_tuning(mlpo[np.newaxis,:])[0]
            frame_scores.append(score)

        video_score = float(np.mean(frame_scores)) if frame_scores else 0.0
        results.append((vid, int(video_score > 0.5)))

    # 4) CSV 저장
    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(results, columns=["ID","label"]).to_csv(OUT_CSV, index=False)
    print("✅ Test submission written to", OUT_CSV)

if __name__=='__main__':
    run_test()
