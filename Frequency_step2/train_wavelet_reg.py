#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet-50 + RGB/Wavelet 학습/검증 스크립트
- 지원 백본 : ResNet-50 only
- Wavelet(SWT) 옵션 지원
- 정규화 실험 : none / label_down / label_up

정규화 정의(실행 가능한 형태)
- none:
    표준 CrossEntropyLoss
- label_down:
    label smoothing 형태의 soft target 사용
    true class confidence = 1 - lambda
- label_up:
    표준 CE + beta * (1 - p_true)^2
    -> 정답 클래스 확률을 1로 더 강하게 밀어주는 confidence-up regularizer
"""

import os
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, default_collate, random_split
from torchvision import transforms
from tqdm import tqdm

from models.resnet_cbam import resnet50


# --------------------- Reproducibility ---------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- Utils -------------------------
def _find_first_conv(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None


def adapt_first_conv_in_channels(model: torch.nn.Module, in_ch: int):
    first_conv = _find_first_conv(model)
    if first_conv is None:
        print("[adapt_first_conv] Conv2d를 찾지 못했어요(스킵).")
        return model

    if first_conv.in_channels == in_ch:
        return model

    with torch.no_grad():
        old_weight = first_conv.weight
        out_c, old_in_c, _, _ = old_weight.shape
        bias = first_conv.bias is not None

        new_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_c,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=bias,
            padding_mode=first_conv.padding_mode,
        )

        if in_ch > old_in_c:
            mean_w = old_weight.mean(dim=1, keepdim=True)
            new_weight = mean_w.repeat(1, in_ch, 1, 1).clone()
        else:
            reduced = old_weight[:, :in_ch, :, :]
            if reduced.shape[1] < in_ch:
                mean_w = old_weight.mean(dim=1, keepdim=True)
                pad = mean_w.repeat(1, in_ch - reduced.shape[1], 1, 1)
                reduced = torch.cat([reduced, pad], dim=1)
            new_weight = reduced.clone()

        new_conv.weight.copy_(new_weight)
        if bias:
            new_conv.bias.copy_(first_conv.bias.data)

    def _replace_first_conv(parent):
        for name, child in parent.named_children():
            if child is first_conv:
                setattr(parent, name, new_conv)
                return True
            if _replace_first_conv(child):
                return True
        return False

    replaced = _replace_first_conv(model)
    if replaced:
        print(f"[adapt_first_conv] 첫 Conv 입력 채널 {old_in_c} → {in_ch} 교체 완료.")
    else:
        print("[adapt_first_conv] 교체 실패(스킵).")
    return model


# --------------------- EarlyStopping ---------------------
class EarlyStopping:
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
            for (_, (cH, cV, cD)) in coeffs:
                feat_maps.extend([
                    np.abs(cH).astype(np.float32),
                    np.abs(cV).astype(np.float32),
                    np.abs(cD).astype(np.float32),
                ])
        elif self.wavelet_details == 'energy':
            for (_, (cH, cV, cD)) in coeffs:
                energy = np.sqrt(cH.astype(np.float32) ** 2 +
                                 cV.astype(np.float32) ** 2 +
                                 cD.astype(np.float32) ** 2)
                feat_maps.append(energy)
        else:
            raise ValueError("wavelet_details must be 'separate' or 'energy'")

        return [self._robust_norm01(m) for m in feat_maps]

    def _wavelet_features(self, arr_bgr: np.ndarray) -> np.ndarray:
        if self.wavelet_gray:
            gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            maps = self._wavelet_maps_per_channel(gray)
            return np.stack(maps, axis=0)

        b, g, r = cv2.split(arr_bgr)
        wb = self._wavelet_maps_per_channel(b)
        wg = self._wavelet_maps_per_channel(g)
        wr = self._wavelet_maps_per_channel(r)
        return np.stack(wb + wg + wr, axis=0)

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
            chans.append(self._wavelet_features(arr).astype(np.float32))

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
        'f1': f1_score(trues, preds, average='macro'),
        'prec': precision_score(trues, preds, average='macro', zero_division=0),
        'recall': recall_score(trues, preds, average='macro', zero_division=0),
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


# --------------------- Regularization loss ---------------------
def build_soft_targets(y: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    if not (0.0 <= smoothing < 1.0):
        raise ValueError(f"label_down smoothing(lambda) must be in [0,1). got {smoothing}")

    with torch.no_grad():
        soft = torch.full((y.size(0), num_classes), smoothing / (num_classes - 1),
                          device=y.device, dtype=torch.float32)
        soft.scatter_(1, y.unsqueeze(1), 1.0 - smoothing)
    return soft


def classification_loss(logits: torch.Tensor,
                        y: torch.Tensor,
                        reg_mode: str,
                        reg_lambda: float,
                        label_up_beta: float):
    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)

    ce_loss = F.cross_entropy(logits, y)
    aux_value = torch.tensor(0.0, device=logits.device)

    if reg_mode == 'none':
        total_loss = ce_loss
    elif reg_mode == 'label_down':
        soft_targets = build_soft_targets(y, num_classes=num_classes, smoothing=reg_lambda)
        total_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        aux_value = total_loss - ce_loss
    elif reg_mode == 'label_up':
        p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1)
        up_reg = ((1.0 - p_true) ** 2).mean()
        total_loss = ce_loss + label_up_beta * up_reg
        aux_value = up_reg
    else:
        raise ValueError(f"Unknown reg_mode: {reg_mode}")

    stats = {
        'ce': float(ce_loss.detach().item()),
        'aux': float(aux_value.detach().item()),
        'total': float(total_loss.detach().item()),
    }
    return total_loss, stats


# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--compression', type=str, default='raw')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--mode', choices=['train', 'val'], default='train')
    parser.add_argument('--ckpt', type=str, help='val 모드에서 불러올 체크포인트(.pth) 경로')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints')
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true', help='ResNet50 ImageNet pretrained 사용')

    parser.add_argument('--use-wavelet', action='store_true')
    parser.add_argument('--wavelet', type=str, default='db2')
    parser.add_argument('--wavelet-level', type=int, default=2)
    parser.add_argument('--wavelet-gray', action='store_true')
    parser.add_argument('--wavelet-details', choices=['separate', 'energy'], default='energy')
    parser.add_argument('--wavelet-include-approx', action='store_true')

    parser.add_argument('--reg-mode', choices=['none', 'label_down', 'label_up'], default='none')
    parser.add_argument('--reg-lambda', type=float, default=0.0,
                        help='label_down용 lambda. true class confidence = 1-lambda')
    parser.add_argument('--label-up-beta', type=float, default=0.1,
                        help='label_up용 confidence-up regularizer weight')

    args = parser.parse_args()
    set_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}\n")
    print(f"▶ Backbone: resnet50 | pretrained={args.pretrained}")
    print(f"▶ Reg mode: {args.reg_mode} | reg_lambda={args.reg_lambda} | label_up_beta={args.label_up_beta}")

    if args.reg_mode != 'label_down' and args.reg_lambda != 0.0:
        print('[WARN] reg_lambda is only used for reg_mode=label_down. current value will be ignored.')

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
        wavelet_include_approx=args.wavelet_include_approx,
    )
    tr_n = int(0.8 * len(ds))
    va_n = len(ds) - tr_n
    split_gen = torch.Generator().manual_seed(args.seed)
    tr_ds, va_ds = random_split(ds, [tr_n, va_n], generator=split_gen)

    tr_ld = DataLoader(
        tr_ds, args.batch_size, True,
        num_workers=4, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda b: default_collate([x for x in b if x])
    )
    va_ld = DataLoader(
        va_ds, args.batch_size, False,
        num_workers=2, pin_memory=True,
        collate_fn=lambda b: default_collate([x for x in b if x])
    )

    in_ch = 3
    if args.use_wavelet:
        wch = calc_wavelet_channels(
            gray=args.wavelet_gray,
            include_approx=args.wavelet_include_approx,
            details=args.wavelet_details,
            level=max(1, int(args.wavelet_level)),
        )
        in_ch += wch
        print(f"▶ Wavelet on: +{wch}ch → in_channels = {in_ch}")

    model = resnet50(pretrained=args.pretrained, num_classes=2)
    model = adapt_first_conv_in_channels(model, in_ch)
    model.to(device)

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    es_ckpt = ckpt_dir / f"es_resnet50_{args.reg_mode}.pth"

    early_stop = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True,
        path=str(es_ckpt),
    )

    if args.mode == 'train':
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        best_f1 = 0.0

        for ep in range(1, args.epochs + 1):
            model.train()
            running_total_loss = 0.0
            running_ce_loss = 0.0
            running_aux = 0.0

            for x, y in tqdm(tr_ld, desc=f"Epoch {ep}/{args.epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss, stats = classification_loss(
                    out, y,
                    reg_mode=args.reg_mode,
                    reg_lambda=args.reg_lambda,
                    label_up_beta=args.label_up_beta,
                )
                loss.backward()
                opt.step()

                running_total_loss += stats['total']
                running_ce_loss += stats['ce']
                running_aux += stats['aux']

            tr_m = compute_metrics(model, tr_ld, device)
            va_m = compute_metrics(model, va_ld, device)
            denom = max(1, len(tr_ld))
            avg_total_loss = running_total_loss / denom
            avg_ce_loss = running_ce_loss / denom
            avg_aux = running_aux / denom

            print(
                f"[{ep}] total_loss:{avg_total_loss:.4f} ce:{avg_ce_loss:.4f} aux:{avg_aux:.4f} "
                f"Tr_acc:{tr_m['acc']:.4f} Tr_f1:{tr_m['f1']:.4f} | "
                f"Va_acc:{va_m['acc']:.4f} Va_f1:{va_m['f1']:.4f}"
            )

            torch.save({
                'model_state': model.state_dict(),
                'optim_state': opt.state_dict(),
                'epoch': ep,
                'best_f1': best_f1,
                'args': vars(args),
            }, ckpt_dir / f"epoch_{ep:03d}.pth")

            if va_m['f1'] > best_f1:
                best_f1 = va_m['f1']
                torch.save({
                    'model_state': model.state_dict(),
                    'optim_state': opt.state_dict(),
                    'epoch': ep,
                    'best_f1': best_f1,
                    'args': vars(args),
                }, ckpt_dir / f"best_resnet50_{args.reg_mode}.pth")
                print(' ▶ best ckpt 저장')

            early_stop(1 - va_m['f1'], model)
            if early_stop.early_stop:
                print('▶ EarlyStopping 발동')
                break

        print(f"\n학습 완료. Best F1: {best_f1:.4f}")

    else:
        assert args.ckpt, '--mode val 시 --ckpt 지정 필요'
        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt.get('model_state', ckpt)
        model.load_state_dict(state_dict, strict=False)
        m = compute_metrics(model, va_ld, device)
        print('\n=== Validation Metrics ===')
        print(f"Accuracy : {m['acc']:.4f}")
        print(f"F1 score : {m['f1']:.4f}")
        print(f"Precision: {m['prec']:.4f}")
        print(f"Recall   : {m['recall']:.4f}")


if __name__ == '__main__':
    main()
