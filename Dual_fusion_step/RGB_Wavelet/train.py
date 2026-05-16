# train_RGB+Wavelet.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-stream FF++ training script for RGB (ConvNeXt-Tiny) + Wavelet (ResNet-50)
- Fusion types:
  1) mlp            : late fusion after feature concat
  2) cross_attention: RGB tokens as query, Wavelet tokens as key/value
- RGB branch       : ConvNeXtV2-Tiny
- Wavelet branch   : ResNet-50 with wavelet-only input
- Wavelet config   : user-selected, intended default = Sym4 + Level2 + SWT + LL+Energy
- Checkpoint policy:
    best_dual_<fusion>.pth
    last_dual_<fusion>.pth
    earlystop_dual_<fusion>.pth
- Supports optional warm-start from previous single-stream checkpoints.
"""

import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pywt
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, default_collate
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from models.convnextv2 import convnextv2_tiny
except ImportError:
    from convnextv2 import convnextv2_tiny

try:
    from models.resnet_cbam import resnet50
except ImportError:
    from resnet_cbam import resnet50


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_first_conv(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None


def adapt_first_conv_in_channels(model: nn.Module, in_ch: int):
    first_conv = _find_first_conv(model)
    if first_conv is None or first_conv.in_channels == in_ch:
        return model

    with torch.no_grad():
        old_weight = first_conv.weight
        out_c, old_in_c, kH, kW = old_weight.shape
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
            new_weight = old_weight[:, :in_ch, :, :].clone()
            if new_weight.shape[1] < in_ch:
                mean_w = old_weight.mean(dim=1, keepdim=True)
                pad = mean_w.repeat(1, in_ch - new_weight.shape[1], 1, 1)
                new_weight = torch.cat([new_weight, pad], dim=1)
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
    print(f"[adapt_first_conv] 첫 Conv 입력 채널 {first_conv.in_channels} → {in_ch} 교체 완료.")
    return model


def atomic_torch_save(obj, path: Path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] improved → {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] no improve ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


class FFPPDualDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        compression: str = "raw",
        img_size: int = 224,
        wavelet: str = "sym4",
        wavelet_level: int = 2,
        wavelet_type: str = "swt",
        wavelet_gray: bool = False,
        subband: str = "ll_energy",
        robust_norm: bool = True,
    ):
        self.samples: List[Tuple[str, int]] = []
        self.img_size = img_size
        self.wavelet = wavelet
        self.wavelet_level = max(1, int(wavelet_level))
        self.wavelet_type = wavelet_type
        self.wavelet_gray = wavelet_gray
        self.subband = subband
        self.robust_norm = robust_norm
        self.resize = transforms.Resize((img_size, img_size))
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        bases = [
            os.path.join(root_dir, "original_sequences"),
            os.path.join(root_dir, "manipulated_sequences"),
        ]
        for label, base in enumerate(bases):
            if not os.path.isdir(base):
                continue
            for method in os.listdir(base):
                full = os.path.join(base, method, compression, "mtcnn")
                if not os.path.isdir(full):
                    continue
                for sub, _, files in os.walk(full):
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
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

    @staticmethod
    def _resize_to(x: np.ndarray, H: int, W: int) -> np.ndarray:
        if x.shape[:2] == (H, W):
            return x.astype(np.float32)
        return cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    def _wavelet_maps_2d(self, ch_2d: np.ndarray, H: int, W: int) -> List[np.ndarray]:
        lvl = self.wavelet_level
        maps: List[np.ndarray] = []
        if self.wavelet_type == "swt":
            coeffs = pywt.swt2(ch_2d, wavelet=self.wavelet, level=lvl, norm=True)
            cA_last = coeffs[-1][0]
            details = [c[1] for c in coeffs]
        elif self.wavelet_type == "dwt":
            coeffs = pywt.wavedec2(ch_2d, wavelet=self.wavelet, level=lvl)
            cA_last = coeffs[0]
            details = list(reversed(coeffs[1:]))
        else:
            raise ValueError("wavelet_type must be 'swt' or 'dwt'")

        if self.subband == "ll":
            maps.append(self._resize_to(cA_last, H, W))
        elif self.subband == "high":
            for (cH, cV, cD) in details:
                maps.extend([
                    self._resize_to(np.abs(cH), H, W),
                    self._resize_to(np.abs(cV), H, W),
                    self._resize_to(np.abs(cD), H, W),
                ])
        elif self.subband == "ll_energy":
            maps.append(self._resize_to(cA_last, H, W))
            for (cH, cV, cD) in details:
                energy = np.sqrt(cH.astype(np.float32) ** 2 + cV.astype(np.float32) ** 2 + cD.astype(np.float32) ** 2)
                maps.append(self._resize_to(energy, H, W))
        else:
            raise ValueError("subband must be one of: ll, high, ll_energy")

        if self.robust_norm:
            maps = [self._robust_norm01(m) for m in maps]
        else:
            maps = [np.clip(m, 0.0, 1.0) for m in maps]
        return maps

    def _wavelet_features(self, arr_bgr: np.ndarray) -> np.ndarray:
        H, W = arr_bgr.shape[:2]
        if self.wavelet_gray:
            gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            maps = self._wavelet_maps_2d(gray, H, W)
            return np.stack(maps, axis=0)
        b, g, r = cv2.split(arr_bgr.astype(np.float32))
        wb = self._wavelet_maps_2d(b, H, W)
        wg = self._wavelet_maps_2d(g, H, W)
        wr = self._wavelet_maps_2d(r, H, W)
        return np.stack(wb + wg + wr, axis=0)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None
        img = self.resize(img)
        arr_rgb = np.array(img).astype(np.float32)
        arr_bgr = arr_rgb[:, :, ::-1].copy()
        rgb = self.rgb_transform(Image.fromarray(arr_rgb.astype(np.uint8)))
        wavelet = torch.from_numpy(self._wavelet_features(arr_bgr).astype(np.float32))
        return rgb, wavelet, torch.tensor(label, dtype=torch.long)


def calc_wavelet_channels(gray: bool, subband: str, level: int) -> int:
    if subband == "ll":
        per_stream = 1
    elif subband == "high":
        per_stream = 3 * level
    elif subband == "ll_energy":
        per_stream = 1 + level
    else:
        raise ValueError("subband must be one of: ll, high, ll_energy")
    return per_stream if gray else per_stream * 3


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, in_ch: int, pretrained: bool = False):
        super().__init__()
        self.backbone = resnet50(pretrained=pretrained, num_classes=2)
        self.backbone = adapt_first_conv_in_channels(self.backbone, in_ch)
        self.feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        feat_map = m.layer4(x)
        pooled = m.avgpool(feat_map)
        pooled = torch.flatten(pooled, 1)
        return {"feat": pooled, "feat_map": feat_map}


class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = convnextv2_tiny(in_chans=3, num_classes=2, use_cbam=False)
        self.feat_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

    def forward(self, x):
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
        feat_map = x
        pooled = self.backbone.norm(x.mean([-2, -1]))
        return {"feat": pooled, "feat_map": feat_map}


class MLPLateFusionModel(nn.Module):
    def __init__(self, wavelet_in_ch: int, hidden_dim: int = 512, dropout: float = 0.2, resnet_pretrained: bool = False):
        super().__init__()
        self.rgb_branch = ConvNeXtFeatureExtractor()
        self.wavelet_branch = ResNetFeatureExtractor(in_ch=wavelet_in_ch, pretrained=resnet_pretrained)
        fusion_dim = self.rgb_branch.feat_dim + self.wavelet_branch.feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, rgb, wavelet):
        rgb_out = self.rgb_branch(rgb)
        wav_out = self.wavelet_branch(wavelet)
        fused = torch.cat([rgb_out["feat"], wav_out["feat"]], dim=1)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "rgb_feat": rgb_out["feat"],
            "wav_feat": wav_out["feat"],
            "rgb_feat_map": rgb_out["feat_map"],
            "wav_feat_map": wav_out["feat_map"],
            "fused_feat": fused,
        }


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, wavelet_in_ch: int, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1, resnet_pretrained: bool = False):
        super().__init__()
        self.rgb_branch = ConvNeXtFeatureExtractor()
        self.wavelet_branch = ResNetFeatureExtractor(in_ch=wavelet_in_ch, pretrained=resnet_pretrained)

        self.rgb_proj = nn.Conv2d(768, embed_dim, kernel_size=1)
        self.wav_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.rgb_gate = nn.Linear(self.rgb_branch.feat_dim, embed_dim)
        self.wav_gate = nn.Linear(self.wavelet_branch.feat_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim * 3)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )
        self.last_attn_weights = None

    def forward(self, rgb, wavelet):
        rgb_out = self.rgb_branch(rgb)
        wav_out = self.wavelet_branch(wavelet)

        rgb_tokens = self.rgb_proj(rgb_out["feat_map"]).flatten(2).transpose(1, 2)  # [B,Nr,D]
        wav_tokens = self.wav_proj(wav_out["feat_map"]).flatten(2).transpose(1, 2)  # [B,Nw,D]

        attn_out, attn_weights = self.attn(
            query=rgb_tokens,
            key=wav_tokens,
            value=wav_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn_weights = attn_weights

        rgb_global = self.rgb_gate(rgb_out["feat"])
        wav_global = self.wav_gate(wav_out["feat"])
        attn_global = attn_out.mean(dim=1)
        fused = self.norm(torch.cat([rgb_global, wav_global, attn_global], dim=1))
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "rgb_feat": rgb_out["feat"],
            "wav_feat": wav_out["feat"],
            "rgb_feat_map": rgb_out["feat_map"],
            "wav_feat_map": wav_out["feat_map"],
            "rgb_tokens": rgb_tokens,
            "wav_tokens": wav_tokens,
            "attn_tokens": attn_out,
            "attn_weights": attn_weights,
            "fused_feat": fused,
        }


def maybe_load_branch_weights(branch: nn.Module, ckpt_path: Optional[str], device: torch.device, prefix: str):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    own = branch.state_dict()
    loaded = {}
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            loaded[k] = v
    own.update(loaded)
    branch.load_state_dict(own, strict=False)
    print(f"[{prefix}] warm-start loaded from {ckpt_path} ({len(loaded)} tensors)")


def build_model(args, wavelet_in_ch: int, device: torch.device):
    if args.fusion == "mlp":
        model = MLPLateFusionModel(
            wavelet_in_ch=wavelet_in_ch,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            resnet_pretrained=args.wavelet_pretrained,
        )
    elif args.fusion == "cross_attention":
        model = CrossAttentionFusionModel(
            wavelet_in_ch=wavelet_in_ch,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            resnet_pretrained=args.wavelet_pretrained,
        )
    else:
        raise ValueError("fusion must be one of: mlp, cross_attention")

    model = model.to(device)
    maybe_load_branch_weights(model.rgb_branch.backbone, args.rgb_ckpt, device, "RGB")
    maybe_load_branch_weights(model.wavelet_branch.backbone, args.wavelet_ckpt, device, "WAVELET")
    return model


@torch.no_grad()
def compute_metrics(model, loader, device):
    model.eval()
    preds, probs, trues = [], [], []
    for batch in loader:
        rgb, wavelet, y = batch
        rgb = rgb.to(device)
        wavelet = wavelet.to(device)
        y = y.to(device)
        out = model(rgb, wavelet)
        logits = out["logits"]
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(1)
        preds.extend(pred.cpu().tolist())
        probs.extend(prob.cpu().tolist())
        trues.extend(y.cpu().tolist())
    metrics = {
        "acc": accuracy_score(trues, preds),
        "f1": f1_score(trues, preds, average="macro"),
        "prec": precision_score(trues, preds, average="macro", zero_division=0),
        "recall": recall_score(trues, preds, average="macro", zero_division=0),
    }
    if len(set(trues)) > 1:
        metrics["auc"] = roc_auc_score(trues, probs)
    else:
        metrics["auc"] = float("nan")
    return metrics


def _auto_find_latest_ckpt(ckpt_dir: Path, fusion: str) -> Optional[Path]:
    for name in [f"last_dual_{fusion}.pth", f"best_dual_{fusion}.pth"]:
        p = ckpt_dir / name
        if p.exists():
            return p
    return None


def _load_resume_ckpt(model, optimizer, ckpt_path: Path, device, strict: bool):
    print(f"▶ Resume from: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict:
        if missing:
            print(f"  - missing keys: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
        if unexpected:
            print(f"  - unexpected keys: {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")
    if optimizer is not None and "optim_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optim_state"])
        except Exception as e:
            print(f"  - optimizer state 로드 스킵 (사유: {e})")
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_f1 = float(ckpt.get("best_f1", 0.0))
    return start_epoch, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--compression", type=str, default="raw")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_dual")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-strict", action="store_true")
    parser.add_argument("--mode", choices=["train", "val"], default="train")
    parser.add_argument("--ckpt", type=str, default=None, help="val 모드에서 평가할 체크포인트")

    parser.add_argument("--fusion", choices=["mlp", "cross_attention"], required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--wavelet", type=str, default="sym4", choices=["haar", "sym4", "db4", "db8"])
    parser.add_argument("--wavelet-level", type=int, default=2, choices=[1, 2])
    parser.add_argument("--wavelet-type", choices=["dwt", "swt"], default="swt")
    parser.add_argument("--subband", choices=["ll", "high", "ll_energy"], default="ll_energy")
    parser.add_argument("--wavelet-gray", action="store_true")
    parser.add_argument("--no-robust-norm", action="store_true")
    parser.add_argument("--wavelet-pretrained", action="store_true", help="ResNet-50 ImageNet pretrained 사용")

    parser.add_argument("--rgb-ckpt", type=str, default=None, help="RGB-only ConvNeXt checkpoint로 warm-start")
    parser.add_argument("--wavelet-ckpt", type=str, default=None, help="Wavelet-only ResNet checkpoint로 warm-start")
    
    parser.add_argument("--freeze-rgb", action="store_true", help="RGB branch encoder freeze")
    parser.add_argument("--freeze-wavelet", action="store_true", help="Wavelet branch encoder freeze")

    args = parser.parse_args()
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}\n")

    ds = FFPPDualDataset(
        root_dir=args.data_dir,
        compression=args.compression,
        img_size=args.img_size,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        wavelet_type=args.wavelet_type,
        wavelet_gray=args.wavelet_gray,
        subband=args.subband,
        robust_norm=not args.no_robust_norm,
    )

    tr_n = int(0.8 * len(ds))
    va_n = len(ds) - tr_n
    tr_ds, va_ds = random_split(ds, [tr_n, va_n], generator=torch.Generator().manual_seed(args.seed))

    collate = lambda b: default_collate([x for x in b if x is not None])
    tr_ld = DataLoader(
        tr_ds, args.batch_size, True,
        num_workers=4, pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate,
    )
    va_ld = DataLoader(
        va_ds, args.batch_size, False,
        num_workers=2, pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate,
    )

    wavelet_in_ch = calc_wavelet_channels(gray=args.wavelet_gray, subband=args.subband, level=args.wavelet_level)
    print(
        f"▶ Dual-stream cfg | fusion={args.fusion} | RGB=ConvNeXt-Tiny | "
        f"Wavelet=ResNet50 + {args.wavelet}-{args.wavelet_level}-{args.wavelet_type}-{args.subband} | "
        f"gray={args.wavelet_gray} | wavelet_in_ch={wavelet_in_ch}\n"
    )

    model = build_model(args, wavelet_in_ch, device)
    
    if args.freeze_rgb:
        for p in model.rgb_branch.parameters():
            p.requires_grad = False
        print("▶ RGB branch frozen")

    if args.freeze_wavelet:
        for p in model.wavelet_branch.parameters():
            p.requires_grad = False
        print("▶ Wavelet branch frozen")
    
    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / f"last_dual_{args.fusion}.pth"
    best_ckpt = ckpt_dir / f"best_dual_{args.fusion}.pth"
    earlystop_ckpt = ckpt_dir / f"earlystop_dual_{args.fusion}.pth"

    if args.mode == "train":
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        best_f1 = 0.0
        start_epoch = 1

        if args.resume:
            resume_ckpt = Path(args.resume_path) if args.resume_path else _auto_find_latest_ckpt(ckpt_dir, args.fusion)
            if resume_ckpt is None:
                print("▶ resume 지정됐지만 사용할 체크포인트가 없습니다. 새로 시작합니다.")
            else:
                start_epoch, best_f1 = _load_resume_ckpt(model, optimizer, resume_ckpt, device, args.resume_strict)
                print(f"  - start_epoch={start_epoch}, best_f1={best_f1:.4f}")

        early_stop = EarlyStopping(patience=args.patience, min_delta=args.min_delta, verbose=True)
        early_stop.best_loss = 1.0 - best_f1 if best_f1 > 0 else np.inf

        for ep in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for rgb, wavelet, y in tqdm(tr_ld, desc=f"Epoch {ep}/{args.epochs}", leave=False):
                rgb, wavelet, y = rgb.to(device), wavelet.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(rgb, wavelet)
                loss = criterion(out["logits"], y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            tr_m = compute_metrics(model, tr_ld, device)
            va_m = compute_metrics(model, va_ld, device)
            avg_loss = running_loss / max(1, len(tr_ld))
            print(
                f"[{ep}] loss:{avg_loss:.4f} "
                f"Tr_acc:{tr_m['acc']:.4f} Tr_f1:{tr_m['f1']:.4f} Tr_auc:{tr_m['auc']:.4f} | "
                f"Va_acc:{va_m['acc']:.4f} Va_f1:{va_m['f1']:.4f} Va_auc:{va_m['auc']:.4f}"
            )

            state = {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "epoch": ep,
                "best_f1": best_f1,
                "args": vars(args),
            }
            if va_m["f1"] > best_f1:
                best_f1 = va_m["f1"]
                state["best_f1"] = best_f1
                atomic_torch_save(state, best_ckpt)
                print(" ▶ best ckpt 저장(덮어쓰기)")
            atomic_torch_save(state, last_ckpt)

            early_stop(1 - va_m["f1"])
            if early_stop.early_stop:
                state["best_f1"] = best_f1
                atomic_torch_save(state, earlystop_ckpt)
                print("▶ EarlyStopping 발동 (earlystop ckpt 저장)")
                break

        print(f"\n학습 완료. Best F1: {best_f1:.4f}")
        print(f" - last:      {last_ckpt}")
        print(f" - best:      {best_ckpt}")
        print(f" - earlystop: {earlystop_ckpt}")

    else:
        assert args.ckpt, "--mode val 시 --ckpt 지정 필요"
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)
        m = compute_metrics(model, va_ld, device)
        print("\n=== Validation Metrics ===")
        print(f"Accuracy : {m['acc']:.4f}")
        print(f"F1 score : {m['f1']:.4f}")
        print(f"Precision: {m['prec']:.4f}")
        print(f"Recall   : {m['recall']:.4f}")
        print(f"AUC      : {m['auc']:.4f}")


if __name__ == "__main__":
    main()
