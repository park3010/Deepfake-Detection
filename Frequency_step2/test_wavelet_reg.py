#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet-50 + RGB/Wavelet 테스트 스크립트
- 지원 백본 : ResNet-50 only
- Wavelet 입력 옵션 지원
- reg-mode / reg-lambda / label-up-beta 메타데이터를 CSV와 파일명에 반영
"""

import argparse
import gc
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

base = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base, 'Frequency_step2'))
sys.path.append(os.path.join(base, 'RGBsparial_step1'))
sys.path.append(os.path.join(base, 'mlp'))

from models.resnet_cbam import resnet50


def _find_first_conv(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None


def adapt_first_conv_in_channels(model: torch.nn.Module, in_ch: int):
    first_conv = _find_first_conv(model)
    if first_conv is None or first_conv.in_channels == in_ch:
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

    _replace_first_conv(model)
    return model


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
        for (_, (cH, cV, cD)) in coeffs:
            feat_maps.extend([
                np.abs(cH).astype(np.float32),
                np.abs(cV).astype(np.float32),
                np.abs(cD).astype(np.float32),
            ])
    elif details == 'energy':
        for (_, (cH, cV, cD)) in coeffs:
            energy = np.sqrt(cH.astype(np.float32) ** 2 +
                             cV.astype(np.float32) ** 2 +
                             cD.astype(np.float32) ** 2)
            feat_maps.append(energy)
    else:
        raise ValueError("wavelet_details must be 'separate' or 'energy'")

    return [_robust_norm01(m) for m in feat_maps]


def _wavelet_features(arr_bgr: np.ndarray, wavelet: str, level: int,
                      gray: bool, details: str, include_approx: bool) -> np.ndarray:
    if gray:
        g = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        maps = _wavelet_maps_per_channel(g, wavelet, level, details, include_approx)
        return np.stack(maps, axis=0)

    b, g, r = cv2.split(arr_bgr)
    wb = _wavelet_maps_per_channel(b, wavelet, level, details, include_approx)
    wg = _wavelet_maps_per_channel(g, wavelet, level, details, include_approx)
    wr = _wavelet_maps_per_channel(r, wavelet, level, details, include_approx)
    return np.stack(wb + wg + wr, axis=0)


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


class VideoFrameDataset(Dataset):
    def __init__(self, frame_paths, resize,
                 use_wavelet=False, wavelet='db2', wavelet_level=2,
                 wavelet_gray=False, wavelet_details='energy',
                 wavelet_include_approx=False):
        self.frames = frame_paths
        self.resize = resize
        self.use_wavelet = use_wavelet
        self.wavelet = wavelet
        self.wavelet_level = max(1, int(wavelet_level))
        self.wavelet_gray = wavelet_gray
        self.wavelet_details = wavelet_details
        self.wavelet_include_approx = wavelet_include_approx
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path = self.frames[idx]
        img = Image.open(path).convert('RGB')

        if self.use_wavelet:
            img_resized = self.resize(img)
            arr = np.array(img_resized)[:, :, ::-1].astype(np.float32)
            base = (arr.transpose(2, 0, 1) / 255.0).astype(np.float32)
            w = _wavelet_features(arr, self.wavelet, self.wavelet_level,
                                  self.wavelet_gray, self.wavelet_details,
                                  self.wavelet_include_approx).astype(np.float32)
            x_np = np.concatenate([base, w], axis=0)
            return torch.from_numpy(x_np)

        return self.rgb_transform(img)


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


def evaluate_dataset(model, device, resize, roots, label_value,
                     batch_size=1, threshold=0.5,
                     use_wavelet=False, wavelet='db2', wavelet_level=2,
                     wavelet_gray=False, wavelet_details='energy',
                     wavelet_include_approx=False):
    y_true, y_pred = [], []
    use_amp = (device.type == 'cuda')

    for root in roots:
        if not os.path.isdir(root):
            print(f'[WARN] 경로 없음: {root}')
            continue
        vids = sorted(os.listdir(root))
        for vid in tqdm(vids, desc=f'[{label_value}] {os.path.basename(root)}'):
            vid_dir = os.path.join(root, vid)
            if not os.path.isdir(vid_dir):
                continue

            frames = sorted(glob.glob(os.path.join(vid_dir, '*.png')))
            frames += sorted(glob.glob(os.path.join(vid_dir, '*.jpg')))
            if not frames:
                continue

            ds = VideoFrameDataset(
                frames, resize,
                use_wavelet=use_wavelet,
                wavelet=wavelet,
                wavelet_level=wavelet_level,
                wavelet_gray=wavelet_gray,
                wavelet_details=wavelet_details,
                wavelet_include_approx=wavelet_include_approx,
            )
            loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
            sum_p = 0.0
            cnt = 0

            for batch in tqdm(loader, desc=f' frames of {vid}', leave=False):
                batch = batch.to(device)
                with torch.inference_mode():
                    with autocast(enabled=use_amp):
                        logits = model(batch)
                        p = torch.softmax(logits, dim=1)[:, 1]

                sum_p += float(p.sum().item())
                cnt += int(p.numel())
                del batch, logits, p

            if cnt == 0:
                continue

            avg_p = sum_p / cnt
            pred = 1 if avg_p >= threshold else 0
            y_true.append(label_value)
            y_pred.append(pred)

            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    return y_true, y_pred


def make_run_tag(args) -> str:
    if args.reg_mode == 'none':
        return 'none'
    if args.reg_mode == 'label_down':
        return f"label_down_lam{str(args.reg_lambda).replace('.', 'p')}"
    return f"label_up_beta{str(args.label_up_beta).replace('.', 'p')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU 번호')
    parser.add_argument('--checkpoint', required=True, help='.pth 파일 경로')
    parser.add_argument('--pretrained', action='store_true', help='모델 생성 시 pretrained=True')

    parser.add_argument('--use-wavelet', action='store_true', help='Wavelet(SWT) 입력 추가(RGB+W)')
    parser.add_argument('--wavelet', type=str, default='db2')
    parser.add_argument('--wavelet-level', type=int, default=2)
    parser.add_argument('--wavelet-gray', action='store_true')
    parser.add_argument('--wavelet-details', choices=['separate', 'energy'], default='energy')
    parser.add_argument('--wavelet-include-approx', action='store_true')

    parser.add_argument('--reg-mode', choices=['none', 'label_down', 'label_up'], default='none')
    parser.add_argument('--reg-lambda', type=float, default=0.0)
    parser.add_argument('--label-up-beta', type=float, default=0.1)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--csv', type=str, default='./result',
                        help='결과 CSV 저장 디렉터리')
    args = parser.parse_args()

    os.makedirs(args.csv, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'▶ Device: {device}')
    torch.cuda.empty_cache()

    in_ch = 3
    if args.use_wavelet:
        wch = calc_wavelet_channels(args.wavelet_gray,
                                    args.wavelet_include_approx,
                                    args.wavelet_details,
                                    max(1, int(args.wavelet_level)))
        in_ch = 3 + wch
        print(f'▶ Wavelet on: +{wch}ch → in_channels = {in_ch}')

    model = resnet50(pretrained=args.pretrained, num_classes=2)
    model = adapt_first_conv_in_channels(model, in_ch)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f'▶ Loaded model from {args.checkpoint}')
    print(f"Model resnet50 params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    torch.cuda.empty_cache()

    resize = transforms.Resize((224, 224))
    results, all_true, all_pred = [], [], []

    for ds_name, cfg in TEST_DATASETS.items():
        if ds_name == 'WildDeepfake':
            real_roots, fake_roots = [], []
            for split in cfg['splits']:
                sd = os.path.join(cfg['root'], split)
                if not os.path.isdir(sd):
                    continue
                for m in os.listdir(sd):
                    base_dir = os.path.join(sd, m)
                    r, f = os.path.join(base_dir, 'real'), os.path.join(base_dir, 'fake')
                    if os.path.isdir(r):
                        real_roots.append(r)
                    if os.path.isdir(f):
                        fake_roots.append(f)
            ds_paths = {'real': real_roots, 'fake': fake_roots}
        elif ds_name == 'DeepfakeTIMIT':
            fake_roots = []
            for quality_root in cfg['fake']:
                if not os.path.isdir(quality_root):
                    continue
                for speaker in os.listdir(quality_root):
                    sp_path = os.path.join(quality_root, speaker)
                    if os.path.isdir(sp_path):
                        fake_roots.append(sp_path)
            ds_paths = {'real': [], 'fake': fake_roots}
        else:
            ds_paths = cfg

        print(f'\n>>> Evaluating {ds_name}')
        rt, rp = evaluate_dataset(model, device, resize,
                                  ds_paths.get('real', []), 0,
                                  args.batch_size, args.threshold,
                                  use_wavelet=args.use_wavelet, wavelet=args.wavelet,
                                  wavelet_level=args.wavelet_level, wavelet_gray=args.wavelet_gray,
                                  wavelet_details=args.wavelet_details,
                                  wavelet_include_approx=args.wavelet_include_approx)
        ft, fp = evaluate_dataset(model, device, resize,
                                  ds_paths.get('fake', []), 1,
                                  args.batch_size, args.threshold,
                                  use_wavelet=args.use_wavelet, wavelet=args.wavelet,
                                  wavelet_level=args.wavelet_level, wavelet_gray=args.wavelet_gray,
                                  wavelet_details=args.wavelet_details,
                                  wavelet_include_approx=args.wavelet_include_approx)
        y_t, y_p = rt + ft, rp + fp

        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1_macro = f1_score(y_t, y_p, average='macro', zero_division=0)
        f1_bin = f1_score(y_t, y_p, average='binary', zero_division=0)
        print(f'[{ds_name}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1-macro={f1_macro:.4f}  F1-binary={f1_bin:.4f}')

        results.append({
            'dataset': ds_name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_macro': f1_macro,
            'f1_binary': f1_bin,
            'reg_mode': args.reg_mode,
            'reg_lambda': args.reg_lambda,
            'label_up_beta': args.label_up_beta,
        })
        all_true.extend(y_t)
        all_pred.extend(y_p)

    if all_true:
        oa = accuracy_score(all_true, all_pred)
        op = precision_score(all_true, all_pred, zero_division=0)
        or_ = recall_score(all_true, all_pred, zero_division=0)
        of1_m = f1_score(all_true, all_pred, average='macro', zero_division=0)
        of1_b = f1_score(all_true, all_pred, average='binary', zero_division=0)
        print('\n=== Overall Metrics ===')
        print(f'Acc   = {oa:.4f}  Prec={op:.4f}  Rec={or_:.4f}  F1-Macro={of1_m:.4f}  F1-Binary={of1_b:.4f}')
        results.append({
            'dataset': 'Overall',
            'accuracy': oa,
            'precision': op,
            'recall': or_,
            'f1_macro': of1_m,
            'f1_binary': of1_b,
            'reg_mode': args.reg_mode,
            'reg_lambda': args.reg_lambda,
            'label_up_beta': args.label_up_beta,
        })

    mode = 'wavelet' if args.use_wavelet else 'rgb'
    tag = make_run_tag(args)
    out_path = f'resnet50_{mode}_{tag}_results.csv'
    csv_path = os.path.join(args.csv, out_path)
    pd.DataFrame(
        results,
        columns=['dataset', 'accuracy', 'precision', 'recall', 'f1_macro', 'f1_binary', 'reg_mode', 'reg_lambda', 'label_up_beta']
    ).to_csv(csv_path, index=False)
    print(f'\n▶ Saved metrics to {csv_path}')


if __name__ == '__main__':
    main()
