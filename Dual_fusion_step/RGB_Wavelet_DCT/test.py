# test_Wavelet+DCT / RGB+DCT.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gc
import glob
import json
import argparse
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import pywt
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.cuda.amp import autocast

try:
    from models.convnextv2 import convnextv2_tiny
except ImportError:
    from convnextv2 import convnextv2_tiny

try:
    from models.resnet_cbam import resnet50
except ImportError:
    from resnet_cbam import resnet50

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
        "root": "/home/oem/deepfake/hdd_5TB/WildDeepfake",
        "splits": ["train", "test"],
    },
}


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
    return model


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


def make_rgb_input(arr_rgb_uint8: np.ndarray):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return t(Image.fromarray(arr_rgb_uint8))


class DualFrameDataset(Dataset):
    def __init__(self, frame_paths: List[str], img_size: int, mode: str, args):
        self.frames = frame_paths
        self.img_size = img_size
        self.mode = mode
        self.args = args
        self.resize = transforms.Resize((img_size, img_size))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path = self.frames[idx]
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
        return x_main, x_dct, path, arr_rgb.astype(np.uint8)


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


def calc_dct_channels(dct_mode: str) -> int:
    return 1 if dct_mode == "gray1" else 3


def build_main_branch(mode: str, args):
    if mode == "wavelet_dct":
        in_ch = calc_wavelet_channels(args.wavelet_gray, args.subband, args.wavelet_level)
        return ResNetFeatureExtractor(in_ch=in_ch, pretrained=False)
    elif mode == "rgb_dct":
        return ConvNeXtFeatureExtractor(in_chans=3)
    else:
        raise ValueError("mode must be wavelet_dct/rgb_dct")


def build_dct_branch(mode: str, args):
    dct_in = calc_dct_channels(args.dct_mode)
    if mode == "rgb_dct":
        return ConvNeXtFeatureExtractor(in_chans=dct_in)
    elif mode == "wavelet_dct":
        return ResNetFeatureExtractor(in_ch=dct_in, pretrained=False)
    else:
        raise ValueError("mode must be wavelet_dct/rgb_dct")


class RGBOnlyConvNeXtClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = convnextv2_tiny(in_chans=3, num_classes=2, use_cbam=False)

    def forward(self, x):
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
        feat_map = x
        feat = self.backbone.norm(x.mean([-2, -1]))
        logits = self.backbone.head(feat)
        return {"logits": logits, "feat_map": feat_map}


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


def build_dual_model(args, device):
    if args.fusion == "mlp":
        model = MLPLateFusionModel(args.mode, args)
    elif args.fusion == "cross_attention":
        model = CrossAttentionFusionModel(args.mode, args)
    else:
        raise ValueError("fusion must be mlp/cross_attention")

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=args.strict)
    if not args.strict:
        if missing:
            print(f"[dual-load] missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[dual-load] unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    return model.to(device).eval()


def build_rgb_only_model(ckpt_path: str, device):
    model = RGBOnlyConvNeXtClassifier()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def get_dataset_roots(ds_name: str, cfg: Dict):
    if ds_name == "WildDeepfake":
        real_roots, fake_roots = [], []
        for split in cfg["splits"]:
            sd = os.path.join(cfg["root"], split)
            if not os.path.isdir(sd):
                continue
            for m in os.listdir(sd):
                base = os.path.join(sd, m)
                r = os.path.join(base, "real")
                f = os.path.join(base, "fake")
                if os.path.isdir(r):
                    real_roots.append(r)
                if os.path.isdir(f):
                    fake_roots.append(f)
        return {"real": real_roots, "fake": fake_roots}

    if ds_name == "DeepfakeTIMIT":
        fake_roots = []
        for quality_root in cfg["fake"]:
            if not os.path.isdir(quality_root):
                continue
            for speaker in os.listdir(quality_root):
                sp_path = os.path.join(quality_root, speaker)
                if os.path.isdir(sp_path):
                    fake_roots.append(sp_path)
        return {"real": [], "fake": fake_roots}

    return cfg


@torch.no_grad()
def evaluate_dataset(model, device, roots: List[str], label_value: int, args):
    y_true, y_pred, y_score = [], [], []
    use_amp = (device.type == "cuda")

    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] 경로 없음: {root}")
            continue
        vids = sorted(os.listdir(root))
        for vid in tqdm(vids, desc=f"[{label_value}] {os.path.basename(root)}"):
            vid_dir = os.path.join(root, vid)
            if not os.path.isdir(vid_dir):
                continue
            frames = sorted(glob.glob(os.path.join(vid_dir, "*.png")))
            frames += sorted(glob.glob(os.path.join(vid_dir, "*.jpg")))
            frames += sorted(glob.glob(os.path.join(vid_dir, "*.jpeg")))
            if not frames:
                continue

            ds = DualFrameDataset(frames, args.img_size, args.mode, args)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=lambda b: [x for x in b if x is not None])

            sum_p, cnt = 0.0, 0
            for batch_list in tqdm(loader, desc=f" frames of {vid}", leave=False):
                if len(batch_list) == 0:
                    continue
                x_main = torch.stack([x[0] for x in batch_list], dim=0).to(device)
                x_dct = torch.stack([x[1] for x in batch_list], dim=0).to(device)
                with torch.inference_mode():
                    with autocast(enabled=use_amp):
                        out = model(x_main, x_dct)
                        p = torch.softmax(out["logits"], dim=1)[:, 1]
                sum_p += float(p.sum().item())
                cnt += int(p.numel())
                del x_main, x_dct, out, p

            if cnt == 0:
                continue
            avg_p = sum_p / cnt
            pred = 1 if avg_p >= args.threshold else 0
            y_true.append(label_value)
            y_pred.append(pred)
            y_score.append(avg_p)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    return y_true, y_pred, y_score


def colorize_cam(cam: np.ndarray, image_bgr: np.ndarray):
    cam = np.clip(cam, 0, 1)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.5, heat, 0.5, 0)


def gradcam_from_module(model, target_module, forward_fn, input_tensors, class_idx=1):
    feat = {}
    grad = {}

    def fwd_hook(_m, _i, o):
        feat["v"] = o

    def bwd_hook(_m, _gi, go):
        grad["v"] = go[0]

    handle_f = target_module.register_forward_hook(fwd_hook)
    handle_b = target_module.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    out = forward_fn(*input_tensors)
    score = out["logits"][:, class_idx].sum()
    score.backward()

    fmap = feat["v"]
    grads = grad["v"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    ref_tensor = input_tensors[0]
    cam = F.interpolate(cam, size=ref_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_f.remove()
    handle_b.remove()
    return cam


def extract_cross_attention_map(model, x_main, x_dct):
    with torch.enable_grad():
        out = model(x_main, x_dct)
    if "attn_weights" not in out or out["attn_weights"] is None:
        return None
    weights = out["attn_weights"]
    attn = weights.mean(dim=1).mean(dim=-1)
    num_q = attn.shape[-1]
    h = w = int(np.sqrt(num_q))
    if h * w != num_q:
        return None
    attn_map = attn[0].detach().cpu().numpy().reshape(h, w)
    attn_map = cv2.resize(attn_map, (x_main.shape[-1], x_main.shape[-2]), interpolation=cv2.INTER_LINEAR)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map


def save_visualizations(args, dual_model, rgb_only_model, device):
    if not args.save_vis:
        return

    vis_dir = Path(args.vis_out)
    vis_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for ds_name, cfg in TEST_DATASETS.items():
        ds_paths = get_dataset_roots(ds_name, cfg)
        for split_name, roots in ds_paths.items():
            for root in roots:
                vids = sorted(os.listdir(root)) if os.path.isdir(root) else []
                for vid in vids:
                    vid_dir = os.path.join(root, vid)
                    if not os.path.isdir(vid_dir):
                        continue
                    frames = sorted(glob.glob(os.path.join(vid_dir, "*.png")))
                    frames += sorted(glob.glob(os.path.join(vid_dir, "*.jpg")))
                    frames += sorted(glob.glob(os.path.join(vid_dir, "*.jpeg")))
                    if not frames:
                        continue

                    frame_path = frames[len(frames) // 2]
                    img = Image.open(frame_path).convert("RGB")
                    img = transforms.Resize((args.img_size, args.img_size))(img)
                    arr_rgb = np.array(img).astype(np.uint8)
                    arr_bgr = arr_rgb[:, :, ::-1].copy()

                    if args.mode == "wavelet_dct":
                        x_main = torch.from_numpy(
                            make_wavelet_input(
                                arr_bgr=arr_bgr.astype(np.float32),
                                wavelet=args.wavelet,
                                level=args.wavelet_level,
                                wavelet_type=args.wavelet_type,
                                wavelet_gray=args.wavelet_gray,
                                subband=args.subband,
                                robust=(not args.no_robust_norm),
                            )
                        ).unsqueeze(0).to(device)
                        target_main = dual_model.main_branch.backbone.layer4
                    else:
                        x_main = make_rgb_input(arr_rgb).unsqueeze(0).to(device)
                        target_main = dual_model.main_branch.backbone.stages[-1]

                    x_dct = torch.from_numpy(make_dct_input(arr_bgr.astype(np.float32), args.dct_mode)).unsqueeze(0).to(device)
                    if args.mode == "rgb_dct":
                        target_dct = dual_model.dct_branch.backbone.stages[-1]
                    else:
                        target_dct = dual_model.dct_branch.backbone.layer4

                    main_cam = gradcam_from_module(dual_model, target_main, lambda a, b: dual_model(a, b), (x_main, x_dct))
                    dct_cam = gradcam_from_module(dual_model, target_dct, lambda a, b: dual_model(a, b), (x_main, x_dct))

                    cv2.imwrite(str(vis_dir / f"{saved:03d}_{ds_name}_{split_name}_main_gradcam.jpg"), colorize_cam(main_cam, arr_bgr))
                    cv2.imwrite(str(vis_dir / f"{saved:03d}_{ds_name}_{split_name}_dct_gradcam.jpg"), colorize_cam(dct_cam, arr_bgr))

                    if args.fusion == "cross_attention":
                        attn_map = extract_cross_attention_map(dual_model, x_main, x_dct)
                        if attn_map is not None:
                            cv2.imwrite(str(vis_dir / f"{saved:03d}_{ds_name}_{split_name}_cross_attention.jpg"), colorize_cam(attn_map, arr_bgr))

                    if args.mode == "rgb_dct" and rgb_only_model is not None:
                        rgb_tensor = make_rgb_input(arr_rgb).unsqueeze(0).to(device)
                        rgb_cam = gradcam_from_module(rgb_only_model, rgb_only_model.backbone.stages[-1], lambda x: rgb_only_model(x), (rgb_tensor,))
                        cv2.imwrite(str(vis_dir / f"{saved:03d}_{ds_name}_{split_name}_rgbonly_gradcam.jpg"), colorize_cam(rgb_cam, arr_bgr))

                    meta = {"dataset": ds_name, "split": split_name, "video": vid, "frame": Path(frame_path).name}
                    with open(vis_dir / f"{saved:03d}_{ds_name}_{split_name}_meta.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                    saved += 1
                    if saved >= args.vis_limit:
                        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", choices=["wavelet_dct", "rgb_dct"], required=True)
    parser.add_argument("--fusion", choices=["mlp", "cross_attention"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strict", action="store_true")

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--wavelet", type=str, default="sym4", choices=["haar", "sym4", "db4", "db8"])
    parser.add_argument("--wavelet-level", type=int, default=2, choices=[1, 2])
    parser.add_argument("--wavelet-type", choices=["dwt", "swt"], default="swt")
    parser.add_argument("--subband", choices=["ll", "high", "ll_energy"], default="ll_energy")
    parser.add_argument("--wavelet-gray", action="store_true")
    parser.add_argument("--no-robust-norm", action="store_true")
    parser.add_argument("--dct-mode", choices=["gray1", "gray3"], default="gray3")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--csv", type=str, default="./result")

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--rgb-only-checkpoint", type=str, default=None)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-limit", type=int, default=8)
    parser.add_argument("--vis-out", type=str, default="./vis_dual")
    args = parser.parse_args()

    os.makedirs(args.csv, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")
    torch.cuda.empty_cache()

    dual_model = build_dual_model(args, device)
    rgb_only_model = build_rgb_only_model(args.rgb_only_checkpoint, device) if (args.mode == "rgb_dct" and args.rgb_only_checkpoint) else None

    results, all_true, all_pred, all_score = [], [], [], []
    for ds_name, cfg in TEST_DATASETS.items():
        ds_paths = get_dataset_roots(ds_name, cfg)
        print(f"\n>>> Evaluating {ds_name}")
        rt, rp, rs = evaluate_dataset(dual_model, device, ds_paths.get("real", []), 0, args)
        ft, fp, fs = evaluate_dataset(dual_model, device, ds_paths.get("fake", []), 1, args)

        y_t, y_p, y_s = rt + ft, rp + fp, rs + fs
        if len(y_t) == 0:
            print(f"[{ds_name}] (skip) 유효 샘플 없음")
            continue

        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1_macro = f1_score(y_t, y_p, average="macro", zero_division=0)
        f1_bin = f1_score(y_t, y_p, average="binary", zero_division=0)
        auc = roc_auc_score(y_t, y_s) if len(set(y_t)) > 1 else float("nan")

        print(f"[{ds_name}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1-macro={f1_macro:.4f}  F1-binary={f1_bin:.4f}  AUC={auc:.4f}")

        results.append({
            "dataset": ds_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_macro": f1_macro,
            "f1_binary": f1_bin,
            "auc": auc,
        })
        all_true.extend(y_t)
        all_pred.extend(y_p)
        all_score.extend(y_s)

    if all_true:
        oa = accuracy_score(all_true, all_pred)
        op = precision_score(all_true, all_pred, zero_division=0)
        or_ = recall_score(all_true, all_pred, zero_division=0)
        of1m = f1_score(all_true, all_pred, average="macro", zero_division=0)
        of1b = f1_score(all_true, all_pred, average="binary", zero_division=0)
        oauc = roc_auc_score(all_true, all_score) if len(set(all_true)) > 1 else float("nan")

        print(f"\n=== Overall Metrics ===")
        print(f"Acc={oa:.4f}  Prec={op:.4f}  Rec={or_:.4f}  F1-macro={of1m:.4f}  F1-binary={of1b:.4f}  AUC={oauc:.4f}")

        results.append({
            "dataset": "Overall",
            "accuracy": oa,
            "precision": op,
            "recall": or_,
            "f1_macro": of1m,
            "f1_binary": of1b,
            "auc": oauc,
        })

    tag = f"{args.mode}_{args.fusion}"
    csv_path = os.path.join(args.csv, f"{tag}_results.csv")
    pd.DataFrame(results, columns=["dataset", "accuracy", "precision", "recall", "f1_macro", "f1_binary", "auc"]).to_csv(csv_path, index=False)
    print(f"\n▶ Saved metrics to {csv_path}")

    save_visualizations(args, dual_model, rgb_only_model, device)
    if args.save_vis:
        print(f"▶ Visualization files saved to {args.vis_out}")


if __name__ == "__main__":
    main()
