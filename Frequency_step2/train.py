#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB-스트림 모델 학습 및 검증 스크립트
- 지원 모델 : ConvNeXt-V2, RepLKNet31B
- 옵션      : FFT/DCT 스트림, CBAM, Early-Stopping, tqdm 진행바
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
    """FF++ (mtcnn crop) 프레임 데이터셋"""
    def __init__(self, root_dir, compression='raw', use_fft=False, use_dct=False, transform=None):
        self.use_fft = use_fft
        self.use_dct = use_dct
        self.transform = transform
        self.samples = []
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
                        if f.lower().endswith(('.png','.jpg','.jpeg')):
                            self.samples.append((os.path.join(sub, f), label))
        print(f"총 샘플 수: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            return None
        # 오직 resize만 수행, ToTensor/Normalize는 사용 안 함
        if self.transform:
            img = self.transform(img)

        arr = np.array(img)[:, :, ::-1].astype(np.float32)  # BGR 순으로
        chans = [arr.transpose(2,0,1) / 255.0]
        if self.use_fft:
            # FFT: arr*255 → 채널별 DFT → magnitude → 로그 스케일
            b,g,r = cv2.split(arr)
            def _fft(ch):
                dft = cv2.dft(ch.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
                mag = cv2.magnitude(dft[:,:,0], dft[:,:,1])
                return 20 * np.log(mag + 1)
            fft_map = np.stack([_fft(b), _fft(g), _fft(r)], axis=2)
            chans.append(fft_map.transpose(2,0,1) / np.max(fft_map))  # 정규화

        if self.use_dct:
            # DCT: 그레이스케일로 변환 → DCT → 정규화
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            dct_map = cv2.dct(gray.astype(np.float32))
            chans.append(dct_map[None] / np.max(dct_map))

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


# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--use-fft', action='store_true', help='FFT 스트림 사용')
    group.add_argument('--use-dct', action='store_true', help='DCT 스트림 사용')
    parser.add_argument('--model', choices=['convnext','replknet31b'], default='convnext')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--epochs',   type=int, default=20)
    parser.add_argument('--batch-size',type=int, default=16)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--use-cbam', action='store_true', help='CBAM 적용')
    parser.add_argument('--mode',     choices=['train','val'], default='train')
    parser.add_argument('--ckpt',     type=str, help='val 모드에서 불러올 체크포인트(.pth) 경로')
    parser.add_argument('--patience', type=int, default=5, help='Early-Stopping patience')
    parser.add_argument('--min-delta',type=float, default=0.0, help='Early-Stopping min_delta')
    parser.add_argument('--checkpoint',type=str,
                        default="./checkpoints",
                        help='학습 중 저장할 체크포인트 디렉토리')
    args = parser.parse_args()

    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}\n")

    # transform: 오직 resize만
    tfm = transforms.Resize((224,224))

    # dataset & split
    ds = FFPPFrameDataset(
        root_dir   = args.data_dir,
        use_fft    = args.use_fft,
        use_dct    = args.use_dct,
        transform  = tfm
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

    # model
    in_ch = 3 + (3 if args.use_fft else 1 if args.use_dct else 0)
    if args.model == 'convnext':
        model = convnextv2_large(in_chans=in_ch, num_classes=2, use_cbam=args.use_cbam)
    else:
        model = create_RepLKNet31B(num_classes=2, in_channels=in_ch, use_cbam=args.use_cbam)
    model.to(device)
    print(f"▶ 모델: {args.model}\n")

    # checkpoint dirs
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

        for ep in range(1, args.epochs+1):
            model.train()
            running_loss = 0.0
            for x,y in tqdm(tr_ld, desc=f"Epoch {ep}/{args.epochs}", leave=False):
                x,y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = crit(out,y)
                loss.backward()
                opt.step()
                running_loss += loss.item()

            tr_m = compute_metrics(model, tr_ld, device)
            va_m = compute_metrics(model, va_ld, device)
            avg_loss = running_loss / len(tr_ld)
            print(f"[{ep}] loss:{avg_loss:.4f} "
                  f"Tr_acc:{tr_m['acc']:.4f} Tr_f1:{tr_m['f1']:.4f} | "
                  f"Va_acc:{va_m['acc']:.4f} Va_f1:{va_m['f1']:.4f}")

            # --- epoch checkpoint 저장
            torch.save({
                'model_state': model.state_dict(),
                'optim_state': opt.state_dict(),
                'epoch': ep,
            }, ckpt_dir / f"epoch_{ep:03d}.pth")

            # --- best checkpoint 갱신
            if va_m['f1'] > best_f1:
                best_f1 = va_m['f1']
                torch.save({
                    'model_state': model.state_dict(),
                    'optim_state': opt.state_dict(),
                    'epoch': ep,
                    'best_f1': best_f1,
                }, ckpt_dir / f"best_{args.model}.pth")
                print(" ▶ best ckpt 저장")

            # --- EarlyStopping
            early_stop(1 - va_m['f1'], model)
            if early_stop.early_stop:
                print("▶ EarlyStopping 발동")
                break

        print(f"\n학습 완료. Best F1: {best_f1:.4f}")

    else:  # val 모드
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
