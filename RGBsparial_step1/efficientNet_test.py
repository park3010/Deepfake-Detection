#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import types
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import timm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---------------------------- Dataset (frame list per video) ----------------------------
class FrameFolderDataset(Dataset):
    """
    특정 비디오 디렉토리의 프레임들을 로드
    반환: (tensor, label, path)
    """
    def __init__(self, frame_paths: List[str], label: int, transform):
        self.frames = frame_paths
        self.label = label
        self.t = transform

    def __len__(self): return len(self.frames)

    def __getitem__(self, idx):
        p = self.frames[idx]
        try:
            img = Image.open(p).convert('RGB')
        except (UnidentifiedImageError, OSError):
            return None
        if self.t:
            img = self.t(img)
        return img, self.label, p

# ---------------------------- Attention Blocks ----------------------------
class ESFCM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8, residual_mode: str = "x_plus_xmulscale"):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1    = nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False)
        self.relu     = nn.ReLU(inplace=True)
        self.conv_mid = nn.Conv2d(mid, mid, 3, 1, 1, bias=False)
        self.conv2    = nn.Conv2d(mid, in_channels, 1, 1, 0, bias=False)
        self.sigmoid  = nn.Sigmoid()
        assert residual_mode in ("x_plus_scale","x_mul_scale","x_plus_xmulscale")
        self.residual_mode = residual_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.max_pool(x)
        m = self.conv1(m); m = self.relu(m)
        m = self.conv_mid(m); m = self.relu(m)
        m = self.conv2(m)

        a = self.avg_pool(x)
        a = self.conv1(a); a = self.relu(a)
        a = self.conv_mid(a); a = self.relu(a)
        a = self.conv2(a)

        scale = self.sigmoid(m + a)
        if self.residual_mode == "x_plus_scale":
            return x + scale
        elif self.residual_mode == "x_mul_scale":
            return x * scale
        else:
            return x + x * scale

class SE(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(self, in_channels: int, reduction: int = 16,
                 residual_mode: str = "x_mul_scale"):
        super().__init__()
        assert residual_mode in ("x_plus_scale","x_mul_scale","x_plus_xmulscale")
        self.residual_mode = residual_mode
        mid = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg_pool(x)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        scale = self.sigmoid(s)
        if self.residual_mode == "x_plus_scale":
            return x + scale
        elif self.residual_mode == "x_mul_scale":
            return x * scale
        else:
            return x + x * scale

class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    def __init__(self, in_channels: int, reduction: int = 16,
                 residual_mode: str = "x_mul_scale", kernel_size: int = 7):
        super().__init__()
        assert residual_mode in ("x_plus_scale","x_mul_scale","x_plus_xmulscale")
        self.residual_mode = residual_mode

        mid = max(1, in_channels // reduction)
        # Channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid_c = nn.Sigmoid()
        # Spatial
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                      stride=1, padding=kernel_size // 2, bias=False)
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale_c = self.sigmoid_c(avg_out + max_out)          # (B,C,1,1)
        x_c = x * scale_c
        # Spatial attention
        avg_map = torch.mean(x_c, dim=1, keepdim=True)       # (B,1,H,W)
        max_map, _ = torch.max(x_c, dim=1, keepdim=True)     # (B,1,H,W)
        s = torch.cat([avg_map, max_map], dim=1)             # (B,2,H,W)
        scale_s = self.sigmoid_s(self.conv_spatial(s))       # (B,1,H,W)

        # 최종 결합 & residual_mode 적용
        if self.residual_mode == "x_plus_scale":
            scale = scale_c * scale_s                         # broadcast
            return x + scale
        elif self.residual_mode == "x_mul_scale":
            return x * (scale_c * scale_s)
        else:  # x_plus_xmulscale
            return x + x * (scale_c * scale_s)

# ---------------------------- feature channels & attach ----------------------------
@torch.no_grad()
def _infer_feat_channels(model: nn.Module, device: Optional[torch.device] = None) -> int:
    if hasattr(model, "num_features") and isinstance(model.num_features, int) and model.num_features > 0:
        return model.num_features
    fi = getattr(model, "feature_info", None)
    if fi:
        try: return int(fi[-1]["num_chs"])
        except Exception: pass
    # probe
    input_size = (3, 224, 224)
    cfg = getattr(model, "default_cfg", {}) or {}
    if "input_size" in cfg and isinstance(cfg["input_size"], (tuple, list)) and len(cfg["input_size"]) == 3:
        input_size = tuple(cfg["input_size"])
    x = torch.zeros(1, *input_size)
    if device is not None:
        x = x.to(device)
    was_train = model.training
    model.eval()
    try:
        if hasattr(model, "forward_features"):
            y = model.forward_features(x)
            if isinstance(y, torch.Tensor) and y.ndim == 4:
                return int(y.shape[1])
    finally:
        model.train(was_train)
    return 1280

def _attach_block(model: nn.Module, block: nn.Module, attr_name: str, device: Optional[torch.device]) -> nn.Module:
    # register & wrap forward_features
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device if device is not None else torch.device('cpu')
    block = block.to(model_device)
    model.add_module(attr_name, block)

    if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
        old_ff = model.forward_features
        def wrapped_forward_features(self, x):
            f = old_ff(x)
            if isinstance(f, torch.Tensor) and f.ndim == 4:
                f = getattr(self, attr_name)(f)
            return f
        model.forward_features = types.MethodType(wrapped_forward_features, model)
    else:
        print(f"[WARN] model has no forward_features; {attr_name} not injected automatically.")
    return model

def attach_esfcm(model: nn.Module, reduction: int, residual_mode: str, device: Optional[torch.device]) -> nn.Module:
    c = _infer_feat_channels(model, device=device)
    return _attach_block(model, ESFCM(c, reduction, residual_mode), "esfcm_before_head", device)

def attach_se(model: nn.Module, reduction: int, residual_mode: str, device: Optional[torch.device]) -> nn.Module:
    c = _infer_feat_channels(model, device=device)
    return _attach_block(model, SE(c, reduction, residual_mode), "se_before_head", device)

def attach_cbam(model: nn.Module, reduction: int, residual_mode: str, kernel_size: int, device: Optional[torch.device]) -> nn.Module:
    c = _infer_feat_channels(model, device=device)
    return _attach_block(model, CBAM(c, reduction, residual_mode, kernel_size), "cbam_before_head", device)

# ---------------------------- Utils ----------------------------
def load_model_and_ckpt(model_name: str, num_classes: int, ckpt_path: str,
                        device: torch.device,
                        use_esfcm: bool = False, esfcm_reduction: int = 8, esfcm_mode: str = "x_plus_xmulscale",
                        use_se: bool = False, se_reduction: int = 16, se_mode: str = "x_mul_scale",
                        use_cbam: bool = False, cbam_reduction: int = 16, cbam_mode: str = "x_mul_scale", cbam_kernel: int = 7
                        ) -> nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(device)

    if use_esfcm:
        model = attach_esfcm(model, esfcm_reduction, esfcm_mode, device)
        print("[ESFCM] attached at classifier head input")
    elif use_se:
        model = attach_se(model, se_reduction, se_mode, device)
        print("[SE] attached at classifier head input")
    elif use_cbam:
        model = attach_cbam(model, cbam_reduction, cbam_mode, cbam_kernel, device)
        print("[CBAM] attached at classifier head input")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt)  # 호환
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    model.eval()
    return model

def compute_metrics_from_probs(y_true: List[int], p_fake: List[float], thr: float = 0.5) -> Dict[str, float]:
    preds = [int(p >= thr) for p in p_fake]
    out = {
        'acc': accuracy_score(y_true, preds) if y_true else 0.0,
        'f1':  f1_score(y_true, preds, average='binary') if y_true else 0.0,
        'prec': precision_score(y_true, preds, average='binary', zero_division=0) if y_true else 0.0,
        'recall': recall_score(y_true, preds, average='binary', zero_division=0) if y_true else 0.0,
        'auc': None
    }
    try:
        out['auc'] = roc_auc_score(y_true, p_fake)
    except Exception:
        out['auc'] = None
    return out

def list_video_dirs(root: str) -> List[str]:
    if not os.path.isdir(root): return []
    vids = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            vids.append(p)
    return sorted(vids)

def list_frames(vid_dir: str) -> List[str]:
    return sorted(
        glob.glob(os.path.join(vid_dir, "*.png")) +
        glob.glob(os.path.join(vid_dir, "*.jpg")) +
        glob.glob(os.path.join(vid_dir, "*.jpeg"))
    )

# ---------------------------- Main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='best checkpoint path (.pth)')
    ap.add_argument('--model', default='tf_efficientnet_b7')
    ap.add_argument('--num-classes', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--out', type=str, default='./results/effb7_eval.csv')

    # ── Attention options (mutually exclusive) ──
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--use-esfcm', action='store_true')
    g.add_argument('--use-se', action='store_true')
    g.add_argument('--use-cbam', action='store_true')

    # ESFCM
    ap.add_argument('--esfcm-reduction', type=int, default=8)
    ap.add_argument('--esfcm-mode', choices=['x_plus_scale','x_mul_scale','x_plus_xmulscale'],
                    default='x_plus_xmulscale')

    # SE
    ap.add_argument('--se-reduction', type=int, default=16)
    ap.add_argument('--se-mode', choices=['x_plus_scale','x_mul_scale','x_plus_xmulscale'],
                    default='x_mul_scale')

    # CBAM
    ap.add_argument('--cbam-reduction', type=int, default=16)
    ap.add_argument('--cbam-mode', choices=['x_plus_scale','x_mul_scale','x_plus_xmulscale'],
                    default='x_mul_scale')
    ap.add_argument('--cbam-kernel', type=int, default=7)

    args = ap.parse_args()

    # ─ 평가 대상 데이터셋 경로 정의 ─
    DATASETS = {
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

    # device
    if str(args.device).lower().startswith('cuda'):
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # model + ckpt (+ optional attention)
    model = load_model_and_ckpt(
        args.model, args.num_classes, args.ckpt, device,
        use_esfcm=args.use_esfcm, esfcm_reduction=args.esfcm_reduction, esfcm_mode=args.esfcm_mode,
        use_se=args.use_se, se_reduction=args.se_reduction, se_mode=args.se_mode,
        use_cbam=args.use_cbam, cbam_reduction=args.cbam_reduction, cbam_mode=args.cbam_mode, cbam_kernel=args.cbam_kernel
    )
    print(f"[INFO] Loaded model {args.model} from {args.ckpt}")

    # transform (학습과 동일)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    pinmem = (device.type == 'cuda')

    records = []        # per-video results
    y_all, p_all = [], []

    # ---------- dataset loop ----------
    for ds_name, cfg in DATASETS.items():

        # --- collect real/fake roots ---
        if ds_name == "WildDeepfake":
            real_roots, fake_roots = [], []
            for split in cfg['splits']:
                sd = os.path.join(cfg['root'], split)
                if not os.path.isdir(sd):
                    continue
                for subj in os.listdir(sd):
                    base = os.path.join(sd, subj)
                    r, f = os.path.join(base, "real"), os.path.join(base, "fake")
                    if os.path.isdir(r): real_roots.append(r)
                    if os.path.isdir(f): fake_roots.append(f)
            ds_paths = {"real": real_roots, "fake": fake_roots}

        elif ds_name == "DeepfakeTIMIT":
            fake_roots = []
            for qroot in cfg["fake"]:
                if not os.path.isdir(qroot):
                    continue
                for spk in os.listdir(qroot):
                    spk_path = os.path.join(qroot, spk)
                    if os.path.isdir(spk_path):
                        fake_roots.append(spk_path)
            ds_paths = {"real": [], "fake": fake_roots}

        else:
            ds_paths = cfg  # Celeb, DFD (real/fake list 그대로)

        # --- evaluate per label ---
        for label_name, label_val in [("real", 0), ("fake", 1)]:
            for root in ds_paths.get(label_name, []):
                if not os.path.isdir(root):
                    print(f"[Skip] {root}")
                    continue

                for vid_dir in tqdm(list_video_dirs(root), desc=f"{ds_name}-{label_name}", leave=False):
                    frames = list_frames(vid_dir)
                    if not frames:
                        continue

                    ds = FrameFolderDataset(frames, label_val, tfm)
                    def _collate(batch):
                        batch = [b for b in batch if b is not None]
                        if not batch: return None
                        imgs, labels, paths = zip(*batch)
                        return torch.stack(imgs, 0), torch.tensor(labels, dtype=torch.long), list(paths)

                    ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                                    num_workers=args.workers, pin_memory=pinmem, collate_fn=_collate)

                    probs = []
                    with torch.no_grad():
                        for b in ld:
                            if b is None:
                                continue
                            x, y, paths = b
                            x = x.to(device)
                            logits = model(x)
                            pr = torch.softmax(logits, 1)[:, 1].detach().cpu().numpy()
                            probs.append(pr)

                    if not probs:
                        continue
                    prob_fake = float(np.concatenate(probs).mean())
                    y_all.append(label_val)
                    p_all.append(prob_fake)
                    records.append({
                        "dataset": ds_name,
                        "video":   vid_dir,
                        "label":   label_val,
                        "prob_fake": prob_fake,
                        "pred": int(prob_fake >= args.thr)
                    })

    # ---------- metrics ----------
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    if y_all:
        overall = compute_metrics_from_probs(y_all, p_all, thr=args.thr)
        print("\n=== Overall (video-level mean prob) ===")
        for k,v in overall.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        metrics_rows.append({"dataset":"Overall", **overall})

    for ds_name in set([r["dataset"] for r in records]):
        ds_labels = [r["label"] for r in records if r["dataset"] == ds_name]
        ds_probs  = [r["prob_fake"] for r in records if r["dataset"] == ds_name]
        if not ds_labels:
            continue
        m = compute_metrics_from_probs(ds_labels, ds_probs, thr=args.thr)
        print(f"\n=== {ds_name} (video-level) ===")
        for k,v in m.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        metrics_rows.append({"dataset": ds_name, **m})

    # saves
    pd.DataFrame(records).to_csv(args.out, index=False)
    print(f"[SAVE] Per-video results → {args.out}")

    metr_out = Path(args.out).with_name(Path(args.out).stem + "_metrics.csv")
    pd.DataFrame(metrics_rows).to_csv(metr_out, index=False)
    print(f"[SAVE] Metrics summary → {metr_out}")

if __name__ == '__main__':
    main()
