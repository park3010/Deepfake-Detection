#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB-스트림 + Wavelet 옵션 학습/검증 스크립트
- 지원 모델 : ConvNeXt-V2, RepLKNet31B
- 옵션      : Early-Stopping, tqdm 진행바, Wavelet(SWT)
- 매 epoch마다 모델 가중치(.pth) 저장
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, random_split, default_collate
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from models.convnextv2 import convnextv2_large
from models.replknet import create_RepLKNet31B

# pip install PyWavelets
import pywt


# --------------------- EarlyStopping ---------------------
class EarlyStopping:
    """val_loss가 patience epoch간 개선되지 않으면 학습 조기 종료"""
    def __init__(self, patience=5, min_delta=0.0, verbose=False, path='checkpoint_es.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.checkpoint_path = path

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(f"[EarlyStopping] val_loss improved → {val_loss:.4f} (ckpt saved)")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] no improve ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


# --------------------- Dataset ---------------------
class FFPPFrameDataset(torch.utils.data.Dataset):
    """FF++ (mtcnn crop) 프레임 데이터셋 (RGB + Wavelet 옵션)"""
    def __init__(
        self,
        root_dir,
        compression='raw',
        transform=None,
        use_wavelet=False,
        wavelet='db2',
        wavelet_level=1,
        wavelet_gray=False,
        wavelet_details='energy',
        wavelet_include_approx=True
    ):
        self.transform = transform
        self.samples = []
        self.use_wavelet = use_wavelet
        self.wavelet = wavelet
        self.wavelet_level = max(1, int(wavelet_level))
        self.wavelet_gray = wavelet_gray
        self.wavelet_details = wavelet_details
        self.wavelet_include_approx = wavelet_include_approx

        bases = [
            os.path.join(root_dir, 'original_sequences'),
            os.path.join(root_dir, 'manipulated_sequences')
        ]
        for label, base in enumerate(bases):
            if not os.path.isdir(base):
                continue
            for method in os.listdir(base):
                full = os.path.join(base, method, compression, 'mtcnn')
                if not os.path.isdir(full):
                    continue
                for sub, _, files in os.walk(full):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(sub, f), label))
        print(f"총 샘플 수: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _robust_norm01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
        denom = max(p99 - p1, eps)
        y = (x - p1) / denom
        return np.clip(y, 0.0, 1.0)

    def _wavelet_maps_per_channel(self, ch_2d: np.ndarray):
        coeffs = pywt.swt2(ch_2d, wavelet=self.wavelet, level=self.wavelet_level, norm=True)
        feat_maps = []

        if self.wavelet_include_approx:
            cA_last = coeffs[-1][0]
            feat_maps.append(cA_last.astype(np.float32))

        if self.wavelet_details == 'separate':
            for (cA, (cH, cV, cD)) in coeffs:
                feat_maps.extend([
                    np.abs(cH).astype(np.float32),
                    np.abs(cV).astype(np.float32),
                    np.abs(cD).astype(np.float32),
                ])
        elif self.wavelet_details == 'energy':
            for (cA, (cH, cV, cD)) in coeffs:
                energy = np.sqrt(cH.astype(np.float32) ** 2 +
                                 cV.astype(np.float32) ** 2 +
                                 cD.astype(np.float32) ** 2)
                feat_maps.append(energy)
        else:
            raise ValueError("wavelet_details must be 'separate' or 'energy'")

        feat_maps = [self._robust_norm01(m) for m in feat_maps]
        return feat_maps

    def _wavelet_features(self, arr_bgr: np.ndarray) -> np.ndarray:
        if self.wavelet_gray:
            gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            maps = self._wavelet_maps_per_channel(gray)
            w = np.stack(maps, axis=0)
            return w
        else:
            b, g, r = cv2.split(arr_bgr)
            wb = self._wavelet_maps_per_channel(b)
            wg = self._wavelet_maps_per_channel(g)
            wr = self._wavelet_maps_per_channel(r)
            w = np.stack(wb + wg + wr, axis=0)
            return w

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            return None

        if self.transform:
            img = self.transform(img)

        arr = np.array(img)[:, :, ::-1].astype(np.float32)
        base = (arr.transpose(2, 0, 1) / 255.0).astype(np.float32)
        chans = [base]

        if self.use_wavelet:
            w = self._wavelet_features(arr).astype(np.float32)
            chans.append(w)

        x = np.concatenate(chans, axis=0)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


# --------------------- Metrics ---------------------
def compute_metrics(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            out = model(x)
            p = out.argmax(1).cpu().tolist()
            preds.extend(p)
            trues.extend(y.tolist())
    return {
        'acc': accuracy_score(trues, preds),
        'f1':  f1_score(trues, preds, average='macro'),
        'prec': precision_score(trues, preds, average='macro', zero_division=0),
        'recall': recall_score(trues, preds, average='macro', zero_division=0)
    }


# --------------------- Wavelet channel calculator ---------------------
def calc_wavelet_channels(gray: bool, include_approx: bool, details: str, level: int) -> int:
    approx = 1 if include_approx else 0
    if details == 'separate':
        per_stream = approx + 3 * level
    elif details == 'energy':
        per_stream = approx + 1 * level
    else:
        raise ValueError("wavelet_details must be 'separate' or 'energy'")
    return per_stream if gray else per_stream * 3


# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['convnext', 'replknet31b'], default='convnext')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--compression', type=str, default='raw')
    parser.add_argument('--epochs',   type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--mode',     choices=['train', 'val'], default='train')
    parser.add_argument('--ckpt',     type=str, help='val 모드에서 불러올 체크포인트(.pth) 경로')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--checkpoint', type=str, default="./checkpoints")

    # Wavelet options
    parser.add_argument('--use-wavelet', action='store_true')
    parser.add_argument('--wavelet', type=str, default='db2')
    parser.add_argument('--wavelet-level', type=int, default=2)
    parser.add_argument('--wavelet-gray', action='store_true')
    parser.add_argument('--wavelet-details', choices=['separate', 'energy'], default='energy')
    parser.add_argument('--wavelet-include-approx', action='store_true')
    
    # 첫 wavlet option은 아래 옵션으로 실험 부탁드려요
    # --use-wavelet --wavelet sym4 --wavelet-level 2 --wavelet-details energy --wavelet-include-approx

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}\n")

    tfm = transforms.Resize((224, 224))

    ds = FFPPFrameDataset(
        root_dir=args.data_dir,
        compression=args.compression,
        transform=tfm,
        use_wavelet=args.use_wavelet,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        wavelet_gray=args.wavelet_gray,
        wavelet_details=args.wavelet_details,
        wavelet_include_approx=args.wavelet_include_approx
    )
    tr_n = int(0.8 * len(ds))
    va_n = len(ds) - tr_n
    tr_ds, va_ds = random_split(ds, [tr_n, va_n], generator=torch.Generator().manual_seed(42))

    tr_ld = DataLoader(tr_ds, args.batch_size, True,
                       num_workers=4, pin_memory=True,
                       collate_fn=lambda b: default_collate([x for x in b if x]))
    va_ld = DataLoader(va_ds, args.batch_size, False,
                       num_workers=2, pin_memory=True,
                       collate_fn=lambda b: default_collate([x for x in b if x]))

    in_ch = 3
    if args.use_wavelet:
        wch = calc_wavelet_channels(
            gray=args.wavelet_gray,
            include_approx=args.wavelet_include_approx,
            details=args.wavelet_details,
            level=max(1, int(args.wavelet_level))
        )
        in_ch += wch
        print(f"▶ Wavelet on: +{wch}ch → in_channels = {in_ch}")

    if args.model == 'convnext':
        model = convnextv2_large(in_chans=in_ch, num_classes=2)
    else:
        model = create_RepLKNet31B(num_classes=2, in_channels=in_ch)
    model.to(device)
    print(f"▶ 모델: {args.model}\n")

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    es_ckpt = ckpt_dir / f"es_{args.model}.pth"

    early_stop = EarlyStopping(patience=args.patience,
                               min_delta=args.min_delta,
                               verbose=True,
                               path=str(es_ckpt))

    if args.mode == 'train':
        crit = nn.CrossEntropyLoss()
        opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        best_f1 = 0.0

        for ep in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for x, y in tqdm(tr_ld, desc=f"Epoch {ep}/{args.epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                running_loss += loss.item()

            tr_m = compute_metrics(model, tr_ld, device)
            va_m = compute_metrics(model, va_ld, device)
            avg_loss = running_loss / len(tr_ld)
            print(f"[{ep}] loss:{avg_loss:.4f} "
                  f"Tr_acc:{tr_m['acc']:.4f} Tr_f1:{tr_m['f1']:.4f} | "
                  f"Va_acc:{va_m['acc']:.4f} Va_f1:{va_m['f1']:.4f}")

            torch.save({
                'model_state': model.state_dict(),
                'optim_state': opt.state_dict(),
                'epoch': ep,
            }, ckpt_dir / f"epoch_{ep:03d}.pth")

            if va_m['f1'] > best_f1:
                best_f1 = va_m['f1']
                torch.save({
                    'model_state': model.state_dict(),
                    'optim_state': opt.state_dict(),
                    'epoch': ep,
                    'best_f1': best_f1,
                }, ckpt_dir / f"best_{args.model}.pth")
                print(" ▶ best ckpt 저장")

            early_stop(1 - va_m['f1'], model)
            if early_stop.early_stop:
                print("▶ EarlyStopping 발동")
                break

        print(f"\n학습 완료. Best F1: {best_f1:.4f}")

    else:
        assert args.ckpt, "--mode val 시 --ckpt 지정 필요"
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        m = compute_metrics(model, va_ld, device)
        print("\n=== Validation Metrics ===")
        print(f"Accuracy : {m['acc']:.4f}")
        print(f"F1 score : {m['f1']:.4f}")
        print(f"Precision: {m['prec']:.4f}")
        print(f"Recall   : {m['recall']:.4f}")


if __name__ == '__main__':
    main()
