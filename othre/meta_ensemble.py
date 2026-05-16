#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from netcal.metrics import ECE
import timm

# 로컬 경로에 맞게 조정
import sys
sys.path.append('/home/oem/deepfake/Ourmethod/Frequency_step2')
from Frequency_step2.models.convnextv2 import convnextv2_large
from RGBsparial_step1.hornet.hornet import  hornet_base_gf

# ─── 설정 ───────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터/모델 경로
TEST_DIR   = Path("/home/oem/deepfake/hdd/test_sample_frames_5")
OUT_CSV    = Path("/home/oem/deepfake/Ourmethod/results/submission_model2_57920205_focal.csv")
RGB_CKPT   = Path("/home/oem/deepfake/Ourmethod/RGBsparial_step1/checkpoints/hornet_base_ddp_best.pth")
FREQ_CKPT  = Path("/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoints/convnext_fft_1_best.pth")
MLP_CKPT   = Path("/home/oem/deepfake/Ourmethod/ensemble_results/mlp_fusion_focal_loss_exclude_best.pth")

# ─── FFT 추출 함수 ───────────────────────────────────────
def extract_fft(bgr: np.ndarray) -> np.ndarray:
    chans = []
    for ch in cv2.split(bgr):
        dft   = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        shift = np.fft.fftshift(dft)
        mag   = cv2.magnitude(shift[:,:,0], shift[:,:,1])
        chans.append((20 * np.log(mag + 1)).astype(np.float32))
    return np.stack(chans, axis=2)

# ─── 베이스 모델 로드 ───────────────────────────────────
def load_base(kind, ckpt_path):
    sd = torch.load(ckpt_path, map_location=DEVICE)
    sd = {k.replace('model.',''): v for k,v in sd.items()}

    if kind=='hornet':
        m =  hornet_base_gf(num_classes=2)
    elif kind=='convnext':
        stem_w = next(v for k,v in sd.items() if k.endswith('downsample_layers.0.0.weight'))
        m = convnextv2_large(in_chans=stem_w.shape[1], num_classes=2, use_cbam=False)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE).eval()

# ─── Meta-MLP 정의 ─────────────────────────────────────
class MetaMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,)
        logpt = -self.ce(logits, targets)          # -CE = log p_t
        pt    = torch.exp(logpt)                   # p_t
        loss  = -self.alpha * (1 - pt)**self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ─── MLP Fusion 학습 루틴 ───────────────────────────────
def train_mlp(rgb_probs, freq_probs, y_true):
    # 1) fusion 입력 준비
    N = min(len(rgb_probs), len(freq_probs), len(y_true))
    rgb_probs  = rgb_probs[:N]
    freq_probs = freq_probs[:N]
    y_true     = y_true[:N]
    
    probs = np.concatenate([rgb_probs, freq_probs], axis=1)
    X = torch.from_numpy(probs).float().to(DEVICE)
    y = torch.from_numpy(y_true).long().to(DEVICE)

    # 2) DataLoader
    ds     = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    # 3) 모델/옵티마이저/손실
    meta_model = MetaMLP(in_dim=probs.shape[1]).to(DEVICE)
    optimiser  = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
    criterion  = FocalLoss(alpha=1.0, gamma=2.0)

    # 4) 학습
    patience = 10
    epochs_no_improve = 0
    meta_model.train()
    best_loss = float('inf')
    
    for epoch in range(1, 1001):
        total_loss = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            logits = meta_model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"[MLP Epoch {epoch:2d}] loss: {avg:.4f}")
        # 체크포인트 저장
        if avg < best_loss:
            best_loss = avg
            epochs_no_improve = 0    
            torch.save(meta_model.state_dict(), MLP_CKPT)
        else:
            epochs_no_improve += 1
            print(f"  ▶ No improvement for {epochs_no_improve}/{patience} epochs.")
        
        if epochs_no_improve >= patience:
            print(f"▶ Early stopping triggered after {epoch} epochs.")
            break
    print("✅ MLP training complete. Best loss:", best_loss)
    print("✅ MLP model saved to", MLP_CKPT)
    return

# ─── Conditional Confidence-Tuned 보정 ────────────────
def apply_confidence_tuning(mlp_probs):
    # mlp_probs: np.ndarray shape (M,2)
    fake_scores  = mlp_probs[:,1]
    confidence   = 2 * np.abs(fake_scores - 0.5)
    final_score  = np.zeros_like(fake_scores)
    for i, s in enumerate(fake_scores):
        if s < 0.5:
            final_score[i] = s * confidence[i] + 0.5 * (1 - confidence[i])
        else:
            final_score[i] = s
    return final_score

# ─── 테스트 프레임 처리 ─────────────────────────────────
def run_test():
    # 1) 베이스 모델 불러오기
    rgb_model  = load_base('hornet',   RGB_CKPT)
    freq_model = load_base('convnext', FREQ_CKPT)

    # 2) MLP 불러오기
    mlp = MetaMLP(in_dim=4).to(DEVICE)  # [p_r0,p_r1,p_f0,p_f1]
    mlp.load_state_dict(torch.load(MLP_CKPT, map_location=DEVICE))
    mlp.eval()

    # 3) 전처리
    resize_tf = T.Resize((224,224))
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],
                    [0.229,0.224,0.225])
    ])
    softmax = nn.Softmax(dim=1)

    results = []
    for vid_dir in tqdm(sorted(TEST_DIR.iterdir()), desc="Test videos"):
        if not vid_dir.is_dir(): continue
        vid = vid_dir.name + ".mp4"

        frame_scores = []
        for img_path in sorted(vid_dir.iterdir()):
            # 1) RGB 스트림
            img = Image.open(img_path).convert("RGB")
            img = resize_tf(img)
            inp_rgb = to_tensor(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pr = softmax(rgb_model(inp_rgb))[0].cpu().numpy()  # shape (2,)

            # 메모리 정리
            torch.cuda.empty_cache()

            # 2) FFT 스트림: BGR + FFT 합쳐서 6채널
            arr = np.array(img).astype(np.float32) / 255.0   # H×W×3 RGB
            bgr = arr[:, :, ::-1]                            # H×W×3 BGR
            fft_m = extract_fft(bgr)                        # H×W×3
            # 두 맵을 채널 축으로 합치기 → H×W×6
            combined = np.concatenate([bgr, fft_m], axis=2) # H×W×6
            inp_fft = torch.from_numpy(
                combined.transpose(2,0,1)  # 6×H×W
            ).unsqueeze(0).to(DEVICE)      # 1×6×H×W

            with torch.no_grad():
                pf = softmax(freq_model(inp_fft))[0].cpu().numpy()  # shape (2,)

            torch.cuda.empty_cache()

            # 3) MLP Fusion
            fusion_in = torch.from_numpy(np.concatenate([pr, pf])).float().to(DEVICE)
            with torch.no_grad():
                mlp_out = softmax(mlp(fusion_in.unsqueeze(0)))[0].cpu().numpy()

            # 4) Conditional Confidence-Tuned MLP
            score = apply_confidence_tuning(mlp_out[np.newaxis, :])[0]
            frame_scores.append(score)

        # 비디오 단위 평균 → 최종 라벨
        #video_score = float(np.mean(frame_scores)) if frame_scores else 0.0
        #video_score = float(np.min(frame_scores)) if frame_scores else 0.0
        video_score = float(np.max(frame_scores)) if frame_scores else 0.0
        label_0       = int(video_score > 0.57920205)
        label_1       = int(video_score > 0.579202051)  
        label_2       = int(video_score > 0.579202052)
        label_3       = int(video_score > 0.579202053)
        label_4       = int(video_score > 0.579202054)
        label_5       = int(video_score > 0.579202055)
        label_6       = int(video_score > 0.579202056)
        label_7       = int(video_score > 0.579202057)
        label_8       = int(video_score > 0.579202058)
        label_9       = int(video_score > 0.579202059)
        label_10       = int(video_score > 57920206)
        
        results.append((vid, label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9, label_10))

    # CSV 저장
    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(results, columns=["ID","0.57920205", "0.579202051", "0.579202052", "0.579202053", "0.579202054", "0.579202055", "0.579202056", "0.579202057", "0.579202058", "0.579202059", "57920206"]).to_csv(OUT_CSV, index=False)
    print("✅ Test submission written to", OUT_CSV)


# ─── 메인 ───────────────────────────────────────────────
if __name__=='__main__':
    #   train_mlp(np.load('/home/oem/deepfake/Ourmethod/RGBsparial_step1/result/hornet_exclude_rgb_probs.npy'),
    #             np.load('/home/oem/deepfake/Ourmethod/Frequency_step2/result/convnext_exclude_freq_probs.npy'),
    #             np.load('/home/oem/deepfake/Ourmethod/Frequency_step2/result/convnext_exclude_y_true.npy'))
    run_test()
