#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습된 딥페이크 모델을 사용해 여러 테스트 데이터셋을 평가하는 스크립트
- 지원 모델 : ConvNeXt-V2, RepLKNet31B, Xception, MaxViT, HorNet, CoaTNet
- 입력: 각 데이터셋별 real/fake 폴더에 저장된 프레임 이미지
- 옵션: --use-fft 또는 --use-dct (상호 배타적)
- 출력: 각 데이터셋 및 전체 통합 Accuracy/Precision/Recall/F1을 터미널에 출력
- CSV 저장: evaluation_results.csv에 결과 저장
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.cuda.amp import autocast
import pywt

base = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base, "Frequency_step2"))
sys.path.append(os.path.join(base, "RGBsparial_step1"))
sys.path.append(os.path.join(base, "mlp"))


# ------------------------------------------------------------------------
# FFT / DCT 추출 함수
# ------------------------------------------------------------------------
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

# ------------------------------------------------------------------------
# Wavelet (SWT) 유틸
# ------------------------------------------------------------------------
def _robust_norm01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    denom = max(p99 - p1, eps)
    y = (x - p1) / denom
    return np.clip(y, 0.0, 1.0)

def _wavelet_maps_per_channel(ch_2d: np.ndarray, wavelet: str, level: int,
                              details: str, include_approx: bool):
    coeffs = pywt.swt2(ch_2d, wavelet=wavelet, level=level, norm=True)
    feat_maps = []

    if include_approx:
        cA_last = coeffs[-1][0]
        feat_maps.append(cA_last.astype(np.float32))

    if details == 'separate':
        for (cA, (cH, cV, cD)) in coeffs:
            feat_maps.extend([
                np.abs(cH).astype(np.float32),
                np.abs(cV).astype(np.float32),
                np.abs(cD).astype(np.float32),
            ])
    elif details == 'energy':
        for (cA, (cH, cV, cD)) in coeffs:
            energy = np.sqrt(cH.astype(np.float32)**2 +
                             cV.astype(np.float32)**2 +
                             cD.astype(np.float32)**2)
            feat_maps.append(energy)
    else:
        raise ValueError("wavelet_details must be 'separate' or 'energy'")

    feat_maps = [_robust_norm01(m) for m in feat_maps]
    return feat_maps

def _wavelet_features(arr_bgr: np.ndarray, wavelet: str, level: int,
                      gray: bool, details: str, include_approx: bool) -> np.ndarray:
    if gray:
        g = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        maps = _wavelet_maps_per_channel(g, wavelet, level, details, include_approx)
        w = np.stack(maps, axis=0)
        return w
    else:
        b, g, r = cv2.split(arr_bgr)
        wb = _wavelet_maps_per_channel(b, wavelet, level, details, include_approx)
        wg = _wavelet_maps_per_channel(g, wavelet, level, details, include_approx)
        wr = _wavelet_maps_per_channel(r, wavelet, level, details, include_approx)
        w = np.stack(wb + wg + wr, axis=0)
        return w

def calc_wavelet_channels(gray: bool, include_approx: bool, details: str, level: int) -> int:
    level = max(1, int(level))
    approx = 1 if include_approx else 0
    if details == 'separate':
        per_stream = approx + 3 * level
    elif details == 'energy':
        per_stream = approx + 1 * level
    else:
        raise ValueError("wavelet_details must be 'separate' or 'energy'")
    return per_stream if gray else per_stream * 3

# ------------------------------------------------------------------------
# VideoFrameDataset
# ------------------------------------------------------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, frame_paths, resize, use_fft=False, use_dct=False, 
                 use_wavelet=False, wavelet='db2', wavelet_level=2,
                 wavelet_gray=False, wavelet_details='energy',
                 wavelet_include_approx=False):
        
        self.frames  = frame_paths
        self.resize  = resize
        self.use_fft = use_fft
        self.use_dct = use_dct

        self.use_wavelet = use_wavelet
        self.wavelet = wavelet
        self.wavelet_level = max(1, int(wavelet_level))
        self.wavelet_gray = wavelet_gray
        self.wavelet_details = wavelet_details
        self.wavelet_include_approx = wavelet_include_approx

        self.normalize = transforms.Normalize([0.485,0.456,0.406],
                                              [0.229,0.224,0.225])
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        path = self.frames[idx]
        img  = Image.open(path).convert('RGB')
        img  = self.resize(img)
        arr  = np.array(img)[:, :, ::-1].astype(np.float32)  # BGR

        if self.use_fft:
            x_np = extract_fft(arr)
            x = torch.from_numpy(x_np.transpose(2,0,1))
            return x  # FFT는 normalize X (학습 입력과 동일한 스케일)
        elif self.use_dct:
            dct  = extract_dct(arr)
            x_np = dct[:, :, None]
            x = torch.from_numpy(x_np.transpose(2,0,1))
            return x  # DCT도 normalize X
        elif self.use_wavelet:
            # 학습 코드와 동일하게 RGB(/255) + Wavelet concat
            base = (arr.transpose(2,0,1) / 255.0).astype(np.float32)
            w = _wavelet_features(arr, self.wavelet, self.wavelet_level,
                                  self.wavelet_gray, self.wavelet_details,
                                  self.wavelet_include_approx).astype(np.float32)
            x_np = np.concatenate([base, w], axis=0)
            x = torch.from_numpy(x_np)
            return x  # Wavelet도 normalize X
        else:
            # 순수 RGB
            x_np = (arr / 255.0).transpose(2,0,1).astype(np.float32)
            x = torch.from_numpy(x_np)
            x = self.normalize(x)
            return x

# ------------------------------------------------------------------------
# TEST_DATASETS 정의
# ------------------------------------------------------------------------
TEST_DATASETS = {
    "Celeb": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-real",
            "/home/oem/deepfake/hdd_5TB/Celeb/YouTube-real"
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-synthesis"
        ],
    },
    "DeepfakeTIMIT": {
        "real": [],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality"
        ],
    },
    "DFD": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences"
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences"
        ],
    },
    "WildDeepfake": {
        "root":   "/home/oem/deepfake/hdd_5TB/WildDeepfake",
        "splits": ["train","test"],
    },
}

# ------------------------------------------------------------------------
# 평가 루프 (tqdm + FP16 + empty_cache)
# ------------------------------------------------------------------------
def evaluate_dataset(model, device, resize, roots, label_value,
                     batch_size=1, threshold=0.5, use_fft=False, use_dct=False,
                     use_wavelet=False, wavelet='db2', wavelet_level=2,
                     wavelet_gray=False, wavelet_details='energy',
                     wavelet_include_approx=False):
    
    y_true, y_pred = [], []
    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] 경로 없음: {root}")
            continue
        vids = sorted(os.listdir(root))
        for vid in tqdm(vids, desc=f"[{label_value}] {os.path.basename(root)}"):
            vid_dir = os.path.join(root, vid)
            if not os.path.isdir(vid_dir): continue

            frames = sorted(glob.glob(os.path.join(vid_dir, "*.png")))
            frames += sorted(glob.glob(os.path.join(vid_dir, "*.jpg")))
            if not frames: continue

            ds     = VideoFrameDataset(frames, resize, use_fft, use_dct,use_wavelet=use_wavelet, wavelet=wavelet,
                                        wavelet_level=wavelet_level, wavelet_gray=wavelet_gray,
                                        wavelet_details=wavelet_details,
                                        wavelet_include_approx=wavelet_include_approx)
            loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
            probs  = []

            for batch in tqdm(loader, desc=f" frames of {vid}", leave=False):
                batch = batch.to(device)
                with autocast():
                    logits = model(batch)
                p = torch.softmax(logits, dim=1)[:,1]
                probs.append(p.detach().cpu().numpy())

            avg_p = float(np.concatenate(probs).mean())
            pred  = 1 if avg_p >= threshold else 0
            y_true.append(label_value)
            y_pred.append(pred)

            # 예약 메모리 해제
            torch.cuda.empty_cache()

    return y_true, y_pred

# ------------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        type=int, default=0, help='GPU 번호')
    parser.add_argument('--model',
        choices=['xception','maxvit','hornet','coatnet','convnext','replknet31b', 'efficientnet', 'mlp'],
        required=True)
    parser.add_argument('--checkpoint', required=True, help='.pth 파일 경로')

    # FFT/DCT/Wavelet 상호 배타
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-fft', action='store_true', help='FFT 입력 사용')
    group.add_argument('--use-dct', action='store_true', help='DCT 입력 사용')
    group.add_argument('--use-wavelet', action='store_true', help='Wavelet(SWT) 입력 추가(RGB+W)')

    # NEW: wavelet 세부 옵션 (학습 스크립트와 동일 기본값)
    parser.add_argument('--wavelet', type=str, default='db2')
    parser.add_argument('--wavelet-level', type=int, default=2)
    parser.add_argument('--wavelet-gray', action='store_true')
    parser.add_argument('--wavelet-details', choices=['separate','energy'], default='energy')
    parser.add_argument('--wavelet-include-approx', action='store_true')

    parser.add_argument('--batch-size', type=int,   default=4)
    parser.add_argument('--threshold',  type=float, default=0.5)
    parser.add_argument('--csv',        type=str,
                        default="/home/sujin/psj2003/deepfake/code/result",
                        help='결과 CSV 저장 디렉터리')
    args = parser.parse_args()

    os.makedirs(args.csv, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}")
    torch.cuda.empty_cache()

    # 입력 채널 계산
    if args.use_fft:
        in_ch = 3
    elif args.use_dct:
        in_ch = 1
    elif args.use_wavelet:
        wch = calc_wavelet_channels(args.wavelet_gray,
                                    args.wavelet_include_approx,
                                    args.wavelet_details,
                                    max(1, int(args.wavelet_level)))
        in_ch = 3 + wch
        print(f"▶ Wavelet on: +{wch}ch → in_channels = {in_ch}")
    else:
        in_ch = 3

    # 모델 인스턴스 생성
    if args.model == 'convnext':
        from Frequency_step2.models.convnextv2 import convnextv2_large
        model = convnextv2_large(
            in_chans=in_ch,
            num_classes=2, use_cbam=False
        )
    elif args.model == 'replknet31b':
        from Frequency_step2.models.replknet import create_RepLKNet31B
        model = create_RepLKNet31B(
            num_classes=2,
            in_channels=in_ch,
            use_cbam=False
        )
    elif args.model == 'xception':
        from RGBsparial_step1.Xception.xception import xception
        model = xception(num_classes=2, use_cbam=False,)
    elif args.model == 'maxvit':
        from RGBsparial_step1.maxvit.maxvit     import MaxViT
        model = MaxViT  (num_classes=2, use_cbam=False)
    elif args.model == 'hornet':
        from RGBsparial_step1.hornet.hornet     import hornet_base_gf, hornet_large_gf
        model = hornet_large_gf(num_classes=2, use_cbam=False)
    elif args.model == 'coatnet':
        from RGBsparial_step1.coatnet.coatnet   import coatnet_0
        model = coatnet_0(num_classes=2)
    elif args.model == 'efficientnet':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)
    elif args.model == 'mlp':
        from mlp.ensemble_mlp import MetaMLP
        model = MetaMLP(in_dim=2)

    # ─── 체크포인트 로드 & stem weight 보정 ───
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('model_state', ckpt)
    stem_key = 'downsample_layers.0.0.weight'
    if stem_key in state_dict:
        w_chk = state_dict[stem_key]
        w_cur = model.state_dict()[stem_key]
        c_chk, c_cur = w_chk.shape[1], w_cur.shape[1]
        if c_chk != c_cur:
            if c_chk == 1 and c_cur > 1:
                state_dict[stem_key] = w_chk.repeat(1, c_cur, 1, 1)
                print(f"[Info] stem weight: 1→{c_cur} 채널로 복제")
            elif c_cur == 1 and c_chk > 1:
                state_dict[stem_key] = w_chk.mean(dim=1, keepdim=True)
                print(f"[Info] stem weight: {c_chk}→1 채널로 평균")
            else:
                state_dict[stem_key] = w_chk.mean(dim=1, keepdim=True)
                print(f"[Info] stem weight: mismatch({c_chk}→{c_cur}), 평균 적용")
    # state_dict.pop(stem_key, None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"▶ Loaded model from {args.checkpoint}")
    print(f"Model {args.model} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    torch.cuda.empty_cache()

    resize = transforms.Resize((224,224))
    results, all_true, all_pred = [], [], []

    # 각 데이터셋 평가
    for ds_name, cfg in TEST_DATASETS.items():
        if ds_name == "WildDeepfake":
            real_roots, fake_roots = [], []
            for split in cfg['splits']:
                sd = os.path.join(cfg['root'], split)
                if not os.path.isdir(sd): continue
                for m in os.listdir(sd):
                    base = os.path.join(sd, m)
                    r,f  = os.path.join(base,"real"), os.path.join(base,"fake")
                    if os.path.isdir(r): real_roots.append(r)
                    if os.path.isdir(f): fake_roots.append(f)
            ds_paths = {'real': real_roots, 'fake': fake_roots}
        elif ds_name == "DeepfakeTIMIT":
            fake_roots = []
            for quality_root in cfg['fake']:
                if not os.path.isdir(quality_root):
                    continue

                # speaker: fadg0, faks0, …
                for speaker in os.listdir(quality_root):
                    sp_path = os.path.join(quality_root, speaker)
                    if os.path.isdir(sp_path):
                        # print()
                        # print(sp_path)
                        # print('----' * 30)
                        fake_roots.append(sp_path)

                    # # video-frames: sa1-video-fram1, sx109-video-fedw0, …
                    # for vid in os.listdir(sp_path):
                    #     vid_path = os.path.join(sp_path, vid)
                    #     if not os.path.isdir(vid_path):
                    #         # print(vid_path)
                    #         # fake_roots.append(vid_path)
                    #         continue

                    #     for root, _, files in os.walk(vid_path):
                    #         for f in files:
                    #             if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    #                 fake_roots.append(os.path.join(root, f))
                            
            ds_paths = {'real': [], 'fake': fake_roots}
        else:
            ds_paths = cfg

        print(f"\n>>> Evaluating {ds_name}")
        rt, rp = evaluate_dataset(model, device, resize,
                                   ds_paths.get('real', []), 0,
                                   args.batch_size, args.threshold,
                                   args.use_fft, args.use_dct,
                                   use_wavelet=args.use_wavelet, wavelet=args.wavelet,
                                   wavelet_level=args.wavelet_level, wavelet_gray=args.wavelet_gray,
                                   wavelet_details=args.wavelet_details,
                                   wavelet_include_approx=args.wavelet_include_approx)
        ft, fp = evaluate_dataset(model, device, resize,
                                   ds_paths.get('fake', []), 1,
                                   args.batch_size, args.threshold,
                                   args.use_fft, args.use_dct,
                                   use_wavelet=args.use_wavelet, wavelet=args.wavelet,
                                   wavelet_level=args.wavelet_level, wavelet_gray=args.wavelet_gray,
                                   wavelet_details=args.wavelet_details,
                                   wavelet_include_approx=args.wavelet_include_approx)
        y_t, y_p = rt + ft, rp + fp

        acc  = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec  = recall_score(y_t, y_p, zero_division=0)
        f1_macro = f1_score(y_t, y_p, average='macro',  zero_division=0)
        f1_bin   = f1_score(y_t, y_p, average='binary', zero_division=0) 
        print(f"[{ds_name}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1-macro={f1_macro:.4f}  F1-binary={f1_bin:.4f}")
        
        results.append({
            'dataset':  ds_name,
            'accuracy': acc,
            'precision':prec,
            'recall':   rec,
            'f1_macro': f1_macro,
            'f1_binary': f1_bin
        })
        all_true.extend(y_t)
        all_pred.extend(y_p)

    # 전체 통합 지표
    if all_true:
        oa   = accuracy_score(all_true, all_pred)
        op   = precision_score(all_true, all_pred, zero_division=0)
        or_  = recall_score(all_true, all_pred, zero_division=0)
        of1_m  = f1_score(all_true, all_pred, average='macro',  zero_division=0)
        of1_b  = f1_score(all_true, all_pred, average='binary',  zero_division=0)
        print(f"\n=== Overall Metrics ===")
        print(f"Acc   = {oa:.4f}  Prec={op:.4f}  Rec={or_:.4f}  F1-Macro={of1_m:.4f}  F1-Binary={of1_b:.4f}")
        results.append({
            'dataset':  'Overall',
            'accuracy': oa,
            'precision':op,
            'recall':   or_,
            'f1_macro': of1_m,
            'f1_binary': of1_b
        })

    # CSV 저장
    if args.use_fft:
        mode = 'fft'
    elif args.use_dct:
        mode = 'dct'
    elif args.use_wavelet:
        mode = 'wavelet'
    else:
        mode = 'rgb'
    out_path = f"{args.model}_{mode}_results.csv"
    csv_path = os.path.join(args.csv, out_path)
    pd.DataFrame(results, columns=['dataset','accuracy','precision','recall','f1_macro', 'f1_binary']).to_csv(csv_path, index=False)
    print(f"\n▶ Saved metrics to {csv_path}")

if __name__ == '__main__':
    main()