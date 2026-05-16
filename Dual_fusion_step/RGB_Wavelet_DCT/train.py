# train_Wavelet+DCT / RGB+DCT.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional

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


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


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
            new_weight = old_weight[:, :in_ch, :, :].clone()
            if new_weight.shape[1] < in_ch:
                mean_w = old_weight.mean(dim=1, keepdim=True)
                pad = mean_w.repeat(1, in_ch - new_weight.shape[1], 1, 1)
                new_weight = torch.cat([new_weight, pad], dim=1)
        new_conv.weight.copy_(new_weight)
        if bias:
            new_conv.bias.copy_(first_conv.bias.data)

    def _replace(parent):
        for name, child in parent.named_children():
            if child is first_conv:
                setattr(parent, name, new_conv)
                return True
            if _replace(child):
                return True
        return False

    _replace(model)
    print(f"[adapt_first_conv] 첫 Conv 입력 채널 {first_conv.in_channels} -> {in_ch} 교체 완료.")
    return model


def maybe_load_branch_weights(branch_backbone: nn.Module, ckpt_path: Optional[str], device: torch.device, prefix: str):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    own = branch_backbone.state_dict()
    loaded = {}
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            loaded[k] = v
    own.update(loaded)
    branch_backbone.load_state_dict(own, strict=False)
    print(f"[{prefix}] warm-start loaded from {ckpt_path} ({len(loaded)} tensors)")


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
                print(f"[EarlyStopping] improved -> {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] no improve ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


def robust_norm01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    denom = max(p99 - p1, eps)
    y = (x - p1) / denom
    return np.clip(y, 0.0, 1.0)


def resize_to(x: np.ndarray, H: int, W: int) -> np.ndarray:
    if x.shape[:2] == (H, W):
        return x.astype(np.float32)
    return cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def wavelet_maps_2d(ch_2d: np.ndarray, H: int, W: int, wavelet: str, level: int, wavelet_type: str, subband: str, robust: bool):
    level = max(1, int(level))
    if wavelet_type == "swt":
        coeffs = pywt.swt2(ch_2d, wavelet=wavelet, level=level, norm=True)
        cA_last = coeffs[-1][0]
        details = [c[1] for c in coeffs]
    elif wavelet_type == "dwt":
        coeffs = pywt.wavedec2(ch_2d, wavelet=wavelet, level=level)
        cA_last = coeffs[0]
        details = list(reversed(coeffs[1:]))
    else:
        raise ValueError("wavelet_type must be swt/dwt")

    maps = []
    if subband == "ll":
        maps.append(resize_to(cA_last, H, W))
    elif subband == "high":
        for (cH, cV, cD) in details:
            maps.extend([resize_to(np.abs(cH), H, W), resize_to(np.abs(cV), H, W), resize_to(np.abs(cD), H, W)])
    elif subband == "ll_energy":
        maps.append(resize_to(cA_last, H, W))
        for (cH, cV, cD) in details:
            energy = np.sqrt(cH.astype(np.float32) ** 2 + cV.astype(np.float32) ** 2 + cD.astype(np.float32) ** 2)
            maps.append(resize_to(energy, H, W))
    else:
        raise ValueError("subband must be ll/high/ll_energy")

    if robust:
        maps = [robust_norm01(m) for m in maps]
    else:
        maps = [np.clip(m, 0.0, 1.0) for m in maps]
    return maps


def make_wavelet_input(arr_bgr: np.ndarray, wavelet: str, level: int, wavelet_type: str, wavelet_gray: bool, subband: str, robust: bool):
    H, W = arr_bgr.shape[:2]
    if wavelet_gray:
        gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        maps = wavelet_maps_2d(gray, H, W, wavelet, level, wavelet_type, subband, robust)
        return np.stack(maps, axis=0).astype(np.float32)

    b, g, r = cv2.split(arr_bgr.astype(np.float32))
    wb = wavelet_maps_2d(b, H, W, wavelet, level, wavelet_type, subband, robust)
    wg = wavelet_maps_2d(g, H, W, wavelet, level, wavelet_type, subband, robust)
    wr = wavelet_maps_2d(r, H, W, wavelet, level, wavelet_type, subband, robust)
    return np.stack(wb + wg + wr, axis=0).astype(np.float32)


def calc_wavelet_channels(gray: bool, subband: str, level: int) -> int:
    if subband == "ll":
        per_stream = 1
    elif subband == "high":
        per_stream = 3 * level
    elif subband == "ll_energy":
        per_stream = 1 + level
    else:
        raise ValueError("subband must be ll/high/ll_energy")
    return per_stream if gray else per_stream * 3


def make_dct_input(arr_bgr: np.ndarray, dct_mode: str):
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dct = cv2.dct(gray).astype(np.float32)
    dct = robust_norm01(dct)
    if dct_mode == "gray1":
        return dct[None, :, :].astype(np.float32)
    elif dct_mode == "gray3":
        return np.stack([dct, dct, dct], axis=0).astype(np.float32)
    else:
        raise ValueError("dct_mode must be gray1/gray3")


def calc_dct_channels(dct_mode: str) -> int:
    return 1 if dct_mode == "gray1" else 3


def make_rgb_input(arr_rgb_uint8: np.ndarray):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return t(Image.fromarray(arr_rgb_uint8))


class DualDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, compression: str, img_size: int, mode: str, args):
        self.samples: List[Tuple[str, int]] = []
        self.img_size = img_size
        self.mode = mode
        self.args = args
        self.resize = transforms.Resize((img_size, img_size))

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

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

        img = self.resize(img)
        arr_rgb = np.array(img).astype(np.float32)
        arr_bgr = arr_rgb[:, :, ::-1].copy()

        if self.mode == "wavelet_dct":
            x_main = torch.from_numpy(
                make_wavelet_input(
                    arr_bgr=arr_bgr,
                    wavelet=self.args.wavelet,
                    level=self.args.wavelet_level,
                    wavelet_type=self.args.wavelet_type,
                    wavelet_gray=self.args.wavelet_gray,
                    subband=self.args.subband,
                    robust=(not self.args.no_robust_norm),
                )
            )
        elif self.mode == "rgb_dct":
            x_main = make_rgb_input(arr_rgb.astype(np.uint8))
        else:
            raise ValueError("mode must be wavelet_dct/rgb_dct")

        x_dct = torch.from_numpy(make_dct_input(arr_bgr, self.args.dct_mode))
        return x_main, x_dct, torch.tensor(label, dtype=torch.long)


class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, in_chans: int = 3):
        super().__init__()
        self.backbone = convnextv2_tiny(in_chans=in_chans, num_classes=2, use_cbam=False)
        self.feat_dim = self.backbone.head.in_features
        self.map_dim = 768
        self.backbone.head = nn.Identity()

    def forward(self, x):
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
        feat_map = x
        pooled = self.backbone.norm(x.mean([-2, -1]))
        return {"feat": pooled, "feat_map": feat_map}


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, in_ch: int, pretrained: bool = False):
        super().__init__()
        self.backbone = resnet50(pretrained=pretrained, num_classes=2)
        self.backbone = adapt_first_conv_in_channels(self.backbone, in_ch)
        self.feat_dim = self.backbone.fc.in_features
        self.map_dim = 2048
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        m = self.backbone
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x)
        feat_map = m.layer4(x)
        pooled = m.avgpool(feat_map)
        pooled = torch.flatten(pooled, 1)
        return {"feat": pooled, "feat_map": feat_map}


def build_main_branch(mode: str, args):
    if mode == "wavelet_dct":
        in_ch = calc_wavelet_channels(args.wavelet_gray, args.subband, args.wavelet_level)
        return ResNetFeatureExtractor(in_ch=in_ch, pretrained=args.resnet_pretrained_main)
    elif mode == "rgb_dct":
        return ConvNeXtFeatureExtractor(in_chans=3)
    else:
        raise ValueError("mode must be wavelet_dct/rgb_dct")


def build_dct_branch(mode: str, args):
    dct_in = calc_dct_channels(args.dct_mode)
    if mode == "rgb_dct":
        return ConvNeXtFeatureExtractor(in_chans=dct_in)
    elif mode == "wavelet_dct":
        return ResNetFeatureExtractor(in_ch=dct_in, pretrained=args.resnet_pretrained_dct)
    else:
        raise ValueError("mode must be wavelet_dct/rgb_dct")


class MLPLateFusionModel(nn.Module):
    def __init__(self, mode: str, args):
        super().__init__()
        self.main_branch = build_main_branch(mode, args)
        self.dct_branch = build_dct_branch(mode, args)
        fusion_dim = self.main_branch.feat_dim + self.dct_branch.feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 2),
        )

    def forward(self, x_main, x_dct):
        m = self.main_branch(x_main)
        d = self.dct_branch(x_dct)
        fused = torch.cat([m["feat"], d["feat"]], dim=1)
        logits = self.classifier(fused)
        return {"logits": logits, "main_feat_map": m["feat_map"], "dct_feat_map": d["feat_map"], "fused_feat": fused}


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, mode: str, args):
        super().__init__()
        self.main_branch = build_main_branch(mode, args)
        self.dct_branch = build_dct_branch(mode, args)
        self.main_proj = nn.Conv2d(self.main_branch.map_dim, args.embed_dim, kernel_size=1)
        self.dct_proj = nn.Conv2d(self.dct_branch.map_dim, args.embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=args.embed_dim, num_heads=args.num_heads, dropout=args.dropout, batch_first=True)
        self.main_gate = nn.Linear(self.main_branch.feat_dim, args.embed_dim)
        self.dct_gate = nn.Linear(self.dct_branch.feat_dim, args.embed_dim)
        self.norm = nn.LayerNorm(args.embed_dim * 3)
        self.classifier = nn.Sequential(
            nn.Linear(args.embed_dim * 3, args.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.embed_dim, 2),
        )

    def forward(self, x_main, x_dct):
        m = self.main_branch(x_main)
        d = self.dct_branch(x_dct)
        main_tokens = self.main_proj(m["feat_map"]).flatten(2).transpose(1, 2)
        dct_tokens = self.dct_proj(d["feat_map"]).flatten(2).transpose(1, 2)
        attn_out, attn_weights = self.attn(query=main_tokens, key=dct_tokens, value=dct_tokens, need_weights=True, average_attn_weights=False)
        main_global = self.main_gate(m["feat"])
        dct_global = self.dct_gate(d["feat"])
        attn_global = attn_out.mean(dim=1)
        fused = self.norm(torch.cat([main_global, dct_global, attn_global], dim=1))
        logits = self.classifier(fused)
        return {"logits": logits, "main_feat_map": m["feat_map"], "dct_feat_map": d["feat_map"], "attn_weights": attn_weights, "fused_feat": fused}


def build_model(args, device):
    if args.fusion == "mlp":
        model = MLPLateFusionModel(args.mode, args)
    elif args.fusion == "cross_attention":
        model = CrossAttentionFusionModel(args.mode, args)
    else:
        raise ValueError("fusion must be mlp/cross_attention")

    model = model.to(device)
    maybe_load_branch_weights(model.main_branch.backbone, args.main_ckpt, device, "MAIN")

    if args.mode == "rgb_dct" and args.convnext_warmstart_dct_from_main and args.dct_ckpt is None and args.main_ckpt is not None:
        maybe_load_branch_weights(model.dct_branch.backbone, args.main_ckpt, device, "DCT_FROM_MAIN")
    else:
        maybe_load_branch_weights(model.dct_branch.backbone, args.dct_ckpt, device, "DCT")
    return model


@torch.no_grad()
def compute_metrics(model, loader, device):
    model.eval()
    preds, probs, trues = [], [], []
    for batch in loader:
        x_main, x_dct, y = batch
        x_main, x_dct, y = x_main.to(device), x_dct.to(device), y.to(device)
        out = model(x_main, x_dct)
        logits = out["logits"]
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(1)
        preds.extend(pred.cpu().tolist())
        probs.extend(prob.cpu().tolist())
        trues.extend(y.cpu().tolist())

    m = {
        "acc": accuracy_score(trues, preds),
        "f1": f1_score(trues, preds, average="macro"),
        "prec": precision_score(trues, preds, average="macro", zero_division=0),
        "recall": recall_score(trues, preds, average="macro", zero_division=0),
    }
    m["auc"] = roc_auc_score(trues, probs) if len(set(trues)) > 1 else float("nan")
    return m


def _auto_find_latest_ckpt(ckpt_dir: Path, tag: str) -> Optional[Path]:
    for name in [f"last_{tag}.pth", f"best_{tag}.pth"]:
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

    parser.add_argument("--mode", choices=["wavelet_dct", "rgb_dct"], required=True)
    parser.add_argument("--fusion", choices=["mlp", "cross_attention"], required=True)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-strict", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="val mode 일때")

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

    parser.add_argument("--dct-mode", choices=["gray1", "gray3"], default="gray3")

    parser.add_argument("--main-ckpt", type=str, default=None)
    parser.add_argument("--dct-ckpt", type=str, default=None)
    parser.add_argument("--freeze-main", action="store_true")
    parser.add_argument("--freeze-dct", action="store_true")
    parser.add_argument("--resnet-pretrained-main", action="store_true")
    parser.add_argument("--resnet-pretrained-dct", action="store_true")
    parser.add_argument("--convnext-warmstart-dct-from-main", action="store_true",
                        help="rgb_dct에서 dct-ckpt가 없을 때 main-ckpt를 DCT ConvNeXt 초기값으로 재사용")

    parser.add_argument("--eval-mode", choices=["train", "val"], default="train")
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}\n")

    ds = DualDataset(args.data_dir, args.compression, args.img_size, args.mode, args)
    tr_n = int(0.8 * len(ds))
    va_n = len(ds) - tr_n
    tr_ds, va_ds = random_split(ds, [tr_n, va_n], generator=torch.Generator().manual_seed(args.seed))

    collate = lambda b: default_collate([x for x in b if x is not None])
    tr_ld = DataLoader(tr_ds, args.batch_size, True, num_workers=4, pin_memory=(device.type == "cuda"), generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)
    va_ld = DataLoader(va_ds, args.batch_size, False, num_workers=2, pin_memory=(device.type == "cuda"), generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)

    if args.mode == "wavelet_dct":
        main_in = calc_wavelet_channels(args.wavelet_gray, args.subband, args.wavelet_level)
        main_desc = f"Wavelet(ResNet50 + {args.wavelet}-{args.wavelet_level}-{args.wavelet_type}-{args.subband})"
    else:
        main_in = 3
        main_desc = "RGB(ConvNeXt-Tiny)"
    dct_in = calc_dct_channels(args.dct_mode)

    print(f"▶ Dual cfg | mode={args.mode} | fusion={args.fusion}")
    print(f"  - main branch: {main_desc} | in_ch={main_in}")
    dct_desc = "DCT(ConvNeXt-Tiny)" if args.mode == "rgb_dct" else "DCT(ResNet50)"
    print(f"  - dct branch : {dct_desc} | in_ch={dct_in}\n")

    model = build_model(args, device)

    if args.freeze_main:
        for p in model.main_branch.parameters():
            p.requires_grad = False
        print("▶ Main branch frozen")
    if args.freeze_dct:
        for p in model.dct_branch.parameters():
            p.requires_grad = False
        print("▶ DCT branch frozen")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"▶ Trainable params: {n_trainable:,} / {n_total:,}")

    tag = f"{args.mode}_{args.fusion}"
    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / f"last_{tag}.pth"
    best_ckpt = ckpt_dir / f"best_{tag}.pth"
    earlystop_ckpt = ckpt_dir / f"earlystop_{tag}.pth"

    if args.eval_mode == "train":
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

        best_f1 = 0.0
        start_epoch = 1
        if args.resume:
            resume_ckpt = Path(args.resume_path) if args.resume_path else _auto_find_latest_ckpt(ckpt_dir, tag)
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
            for x_main, x_dct, y in tqdm(tr_ld, desc=f"Epoch {ep}/{args.epochs}", leave=False):
                x_main, x_dct, y = x_main.to(device), x_dct.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x_main, x_dct)
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
        assert args.ckpt, "--eval-mode val 시 --ckpt 지정 필요"
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
