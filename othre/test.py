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

base = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base, "Frequency_step2"))
sys.path.append(os.path.join(base, "RGBsparial_step1"))


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
# VideoFrameDataset
# ------------------------------------------------------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, frame_paths, resize, use_fft=False, use_dct=False):
        self.frames  = frame_paths
        self.resize  = resize
        self.use_fft = use_fft
        self.use_dct = use_dct
        self.normalize = transforms.Normalize([0.485,0.456,0.406],
                                              [0.229,0.224,0.225])
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        path = self.frames[idx]
        img  = Image.open(path).convert('RGB')
        img  = self.resize(img)
        arr  = np.array(img)[:, :, ::-1]  # BGR

        if self.use_fft:
            x_np = extract_fft(arr)
        elif self.use_dct:
            dct  = extract_dct(arr)
            x_np = dct[:, :, None]
        else:
            x_np = arr.astype(np.float32) / 255.0

        x = torch.from_numpy(x_np.transpose(2,0,1))
        if not (self.use_fft or self.use_dct):
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
    "DFD": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences"
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences"
        ],
    },
    "DeepfakeTIMIT": {
        "real": [],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality"
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
                     batch_size=1, threshold=0.5, use_fft=False, use_dct=False):
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

            ds     = VideoFrameDataset(frames, resize, use_fft, use_dct)
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
        choices=['xception','maxvit','hornet','coatnet','convnext','replknet31b'],
        required=True)
    parser.add_argument('--checkpoint', required=True, help='.pth 파일 경로')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-fft', action='store_true', help='FFT 입력 사용')
    group.add_argument('--use-dct', action='store_true', help='DCT 입력 사용')
    parser.add_argument('--batch-size', type=int,   default=1)
    parser.add_argument('--threshold',  type=float, default=0.5)
    parser.add_argument('--csv',        type=str,
                        default="/home/oem/deepfake/Ourmethod/results",
                        help='결과 CSV 저장 디렉터리')
    args = parser.parse_args()

    os.makedirs(args.csv, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}")
    torch.cuda.empty_cache()

    # 모델 인스턴스 생성
    if args.model == 'convnext':
        from Frequency_step2.models.convnextv2 import convnextv2_large
        model = convnextv2_large(
            in_chans=3 if not (args.use_fft or args.use_dct)
                    else (3 if args.use_fft else 1),
            num_classes=2, use_cbam=False
        )
    elif args.model == 'replknet31b':
        from Frequency_step2.models.replknet import create_RepLKNet31B
        model = create_RepLKNet31B(
            num_classes=2,
            in_channels=3 if not (args.use_fft or args.use_dct)
                        else (3 if args.use_fft else 1),
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

    # ─── 체크포인트 로드 & stem weight 보정 ───
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('model_state', ckpt)
    stem_key = 'stem.0.0.0.weight'
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
    state_dict.pop(stem_key, None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"▶ Loaded model from {args.checkpoint}")
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
        elif ds_name == "DeepfkeTIMIT":
            real_roots = []
            fake_roots = []
            for q in cfg['real']:
                for grp in os.listdir(q):
                    grp_p = os.path.join(q, grp)
                    if os.path.isdir(grp_p):
                        for vid in os.listdir(grp_p):
                            p = os.path.join(grp_p, vid)
                            if os.path.isdir(p): real_roots.append(p)
            for q in cfg['fake']:
                for grp in os.listdir(q):
                    grp_p = os.path.join(q, grp)
                    if os.path.isdir(grp_p):
                        for vid in os.listdir(grp_p):
                            p = os.path.join(grp_p, vid)
                            if os.path.isdir(p): fake_roots.append(p)
            ds_paths = {'real': real_roots, 'fake': fake_roots}
        else:
            ds_paths = cfg

        print(f"\n>>> Evaluating {ds_name}")
        rt, rp = evaluate_dataset(model, device, resize,
                                   ds_paths.get('real', []), 0,
                                   args.batch_size, args.threshold,
                                   args.use_fft, args.use_dct)
        ft, fp = evaluate_dataset(model, device, resize,
                                   ds_paths.get('fake', []), 1,
                                   args.batch_size, args.threshold,
                                   args.use_fft, args.use_dct)
        y_t, y_p = rt + ft, rp + fp

        acc  = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec  = recall_score(y_t, y_p, zero_division=0)
        f1   = f1_score(y_t, y_p, average='macro')
        print(f"[{ds_name}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
        
        results.append({
            'dataset':  ds_name,
            'accuracy': acc,
            'precision':prec,
            'recall':   rec,
            'f1_macro': f1
        })
        all_true.extend(y_t)
        all_pred.extend(y_p)

    # 전체 통합 지표
    if all_true:
        oa   = accuracy_score(all_true, all_pred)
        op   = precision_score(all_true, all_pred, zero_division=0)
        or_  = recall_score(all_true, all_pred, zero_division=0)
        of1  = f1_score(all_true, all_pred, average='macro')
        print(f"\n=== Overall Metrics ===")
        print(f"Acc   = {oa:.4f}  Prec={op:.4f}  Rec={or_:.4f}  F1={of1:.4f}")
        results.append({
            'dataset':  'Overall',
            'accuracy': oa,
            'precision':op,
            'recall':   or_,
            'f1_macro': of1
        })

    # CSV 저장
    mode = 'fft' if args.use_fft else 'dct' if args.use_dct else 'rgb'
    out_path = f"{args.model}_{mode}_results.csv"
    csv_path = os.path.join(args.csv, out_path)
    pd.DataFrame(results, columns=['dataset','accuracy','precision','recall','f1_macro']).to_csv(csv_path, index=False)
    print(f"\n▶ Saved metrics to {csv_path}")

if __name__ == '__main__':
    main()