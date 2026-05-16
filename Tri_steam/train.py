#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tri/Multi-stream FF++ training script
- Streams:
  rgb, wavelet, dct, semantic
- Fusion:
  multi_stream_cross_attention

Added:
- validation metrics saving
- eval-only mode
- resume full checkpoint loading
"""

import os
import argparse
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import pywt
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from transformers import CLIPModel, CLIPImageProcessor

from models.convnextv2 import convnextv2_tiny
from models.resnet_cbam import resnet50


# ---------------------------------------------------------
# Basic utils
# ---------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_torch_save(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


def parse_streams(streams: str) -> List[str]:
    out = [s.strip().lower() for s in streams.split(",") if s.strip()]
    valid = {"rgb", "wavelet", "dct", "semantic"}

    for s in out:
        if s not in valid:
            raise ValueError(f"Unknown stream: {s}")

    if len(out) < 2:
        raise ValueError("At least two streams are required.")

    return out


def make_tag(streams: List[str]) -> str:
    return "_".join(streams)


# ---------------------------------------------------------
# Conv channel adaptation
# ---------------------------------------------------------
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
    print(f"[adapt_first_conv] first conv input channels {old_in_c} -> {in_ch}")

    return model


# ---------------------------------------------------------
# Branch checkpoint loading
# ---------------------------------------------------------
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model_state" in ckpt_obj:
            return ckpt_obj["model_state"]
        if "model" in ckpt_obj:
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]
    return ckpt_obj


def _candidate_keys(k: str) -> List[str]:
    keys = [k]

    prefixes = [
        "module.",
        "model.",
        "backbone.",
        "rgb_branch.",
        "rgb_branch.backbone.",
        "wavelet_branch.",
        "wavelet_branch.backbone.",
        "wav_branch.",
        "wav_branch.backbone.",
        "dct_branch.",
        "dct_branch.backbone.",
        "main_branch.",
        "main_branch.backbone.",
    ]

    changed = True
    while changed:
        changed = False
        new_keys = []

        for key in keys:
            for p in prefixes:
                if key.startswith(p):
                    nk = key[len(p):]
                    if nk not in keys and nk not in new_keys:
                        new_keys.append(nk)
                        changed = True

        keys.extend(new_keys)

    unique = []
    for key in keys:
        if key not in unique:
            unique.append(key)

    return unique


def load_branch_weights(
    branch: nn.Module,
    ckpt_path: Optional[str],
    device: torch.device,
    name: str,
):
    if ckpt_path is None or ckpt_path == "":
        print(f"[{name}] no checkpoint provided. Use current initialization.")
        return

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[{name}] checkpoint not found: {ckpt_path}")

    print(f"[{name}] loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(ckpt)

    if not isinstance(state, dict):
        raise TypeError(f"[{name}] checkpoint state is not a dict: {type(state)}")

    target_state = branch.state_dict()
    loaded = {}
    skipped_shape = []
    skipped_missing = []

    for k, v in state.items():
        matched = False

        for ck in _candidate_keys(k):
            if ck in target_state:
                if target_state[ck].shape == v.shape:
                    loaded[ck] = v
                else:
                    skipped_shape.append((k, ck, tuple(v.shape), tuple(target_state[ck].shape)))
                matched = True
                break

        if not matched:
            skipped_missing.append(k)

    target_state.update(loaded)
    branch.load_state_dict(target_state, strict=False)

    print(f"[{name}] loaded tensors: {len(loaded)} / target tensors: {len(target_state)}")

    if len(loaded) == 0:
        print(f"[{name}] WARNING: no tensor was loaded. Check checkpoint key names.")

    if skipped_shape:
        print(f"[{name}] skipped by shape mismatch: {len(skipped_shape)}")
        for item in skipped_shape[:5]:
            src_k, dst_k, src_shape, dst_shape = item
            print(f"  - {src_k} -> {dst_k}: {src_shape} != {dst_shape}")

    if skipped_missing:
        print(f"[{name}] skipped missing/unmatched keys: {len(skipped_missing)}")
        print(f"  - examples: {skipped_missing[:5]}")


def load_requested_branch_checkpoints(model: nn.Module, args, streams: List[str], device: torch.device):
    if "rgb" in streams:
        load_branch_weights(
            branch=model.branches["rgb"].backbone,
            ckpt_path=args.rgb_ckpt,
            device=device,
            name="RGB",
        )

    if "wavelet" in streams:
        load_branch_weights(
            branch=model.branches["wavelet"].backbone,
            ckpt_path=args.wavelet_ckpt,
            device=device,
            name="Wavelet",
        )

    if "dct" in streams:
        load_branch_weights(
            branch=model.branches["dct"].backbone,
            ckpt_path=args.dct_ckpt,
            device=device,
            name="DCT",
        )


def load_full_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    resume_path: Optional[str],
    device: torch.device,
    strict: bool = True,
    load_optimizer: bool = True,
):
    if resume_path is None or resume_path == "":
        return 0, None

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"[resume] checkpoint not found: {resume_path}")

    print(f"[resume] loading full checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)

    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)

    if not strict:
        if missing:
            print(f"[resume] missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[resume] unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    if (
        optimizer is not None
        and load_optimizer
        and isinstance(ckpt, dict)
        and "optim_state" in ckpt
    ):
        try:
            optimizer.load_state_dict(ckpt["optim_state"])
            print("[resume] optimizer state loaded")
        except Exception as e:
            print(f"[resume] optimizer state load skipped: {e}")

    start_epoch = 0
    best_score = None

    if isinstance(ckpt, dict):
        start_epoch = int(ckpt.get("epoch", 0))
        best_score = ckpt.get("best_score", None)

    print(f"[resume] loaded epoch={start_epoch}, best_score={best_score}")
    return start_epoch, best_score


# ---------------------------------------------------------
# Wavelet feature
# ---------------------------------------------------------
def robust_norm01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    denom = max(p99 - p1, eps)
    y = (x - p1) / denom
    return np.clip(y, 0.0, 1.0)


def resize_to(x: np.ndarray, H: int, W: int) -> np.ndarray:
    if x.shape[:2] == (H, W):
        return x.astype(np.float32)
    return cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def wavelet_maps_2d(
    ch_2d: np.ndarray,
    H: int,
    W: int,
    wavelet: str,
    level: int,
    wavelet_type: str,
    subband: str,
    robust: bool,
):
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
        raise ValueError("wavelet_type must be swt or dwt")

    maps = []

    if subband == "ll":
        maps.append(resize_to(cA_last, H, W))

    elif subband == "high":
        for cH, cV, cD in details:
            maps.extend([
                resize_to(np.abs(cH), H, W),
                resize_to(np.abs(cV), H, W),
                resize_to(np.abs(cD), H, W),
            ])

    elif subband == "ll_energy":
        maps.append(resize_to(cA_last, H, W))
        for cH, cV, cD in details:
            energy = np.sqrt(
                cH.astype(np.float32) ** 2
                + cV.astype(np.float32) ** 2
                + cD.astype(np.float32) ** 2
            )
            maps.append(resize_to(energy, H, W))

    else:
        raise ValueError("subband must be ll, high, or ll_energy")

    if robust:
        maps = [robust_norm01(m) for m in maps]
    else:
        maps = [np.clip(m, 0.0, 1.0) for m in maps]

    return maps


def make_wavelet_input(
    arr_bgr: np.ndarray,
    wavelet: str,
    level: int,
    wavelet_type: str,
    wavelet_gray: bool,
    subband: str,
    robust: bool,
):
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
        raise ValueError("subband must be ll, high, or ll_energy")

    return per_stream if gray else per_stream * 3


# ---------------------------------------------------------
# DCT feature
# ---------------------------------------------------------
_DCT_MAT_CACHE = {}


def bgr_to_ycrcb01(bgr01: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor((bgr01 * 255.0).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    return ycrcb.astype(np.float32) / 255.0


def normalize_map_fast(m: np.ndarray, eps: float = 1e-6, stride: int = 4) -> np.ndarray:
    m = np.log1p(np.abs(m)).astype(np.float32)
    ms = m[::stride, ::stride].reshape(-1)

    lo, hi = np.percentile(ms, 1), np.percentile(ms, 99)
    m = np.clip(m, lo, hi)
    m = (m - lo) / (hi - lo + eps)

    return m.astype(np.float32)


def _get_dct_mat(N: int, device: str = "cpu") -> torch.Tensor:
    key = (N, device)

    if key in _DCT_MAT_CACHE:
        return _DCT_MAT_CACHE[key]

    k = torch.arange(N, dtype=torch.float32, device=device).view(N, 1)
    n = torch.arange(N, dtype=torch.float32, device=device).view(1, N)

    alpha = torch.ones((N,), dtype=torch.float32, device=device)
    alpha[0] = 1.0 / math.sqrt(2.0)

    C = math.sqrt(2.0 / N) * alpha.view(N, 1) * torch.cos(
        (math.pi * (2.0 * n + 1.0) * k) / (2.0 * N)
    )

    _DCT_MAT_CACHE[key] = C
    return C


def extract_block_dct_energy(
    bgr01: np.ndarray,
    block: int = 8,
    freq_in: str = "ycbcr",
    energy_mode: str = "ac",
) -> np.ndarray:
    ycrcb = bgr_to_ycrcb01(bgr01)

    if freq_in == "y":
        chans = [ycrcb[:, :, 0]]
    elif freq_in == "ycbcr":
        chans = [
            ycrcb[:, :, 0],
            ycrcb[:, :, 1],
            ycrcb[:, :, 2],
        ]
    else:
        raise ValueError("freq_in must be 'y' or 'ycbcr'")

    C = _get_dct_mat(block, device="cpu")
    Ct = C.t()

    outs = []

    for ch in chans:
        H, W = ch.shape

        Hp = (H + block - 1) // block * block
        Wp = (W + block - 1) // block * block

        pad = np.zeros((Hp, Wp), dtype=np.float32)
        pad[:H, :W] = ch.astype(np.float32)

        t = torch.from_numpy(pad)
        blocks = t.unfold(0, block, block).unfold(1, block, block)

        temp = torch.einsum("ij,abjk->abik", C, blocks)
        dct = torch.einsum("abik,kj->abij", temp, Ct)

        a = dct.abs()
        total = a.sum(dim=(-1, -2))

        if energy_mode == "ac":
            dc = a[:, :, 0, 0]
            energy = total - dc
        elif energy_mode == "total":
            energy = total
        else:
            raise ValueError("energy_mode must be 'ac' or 'total'")

        e_map = energy.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1).numpy()
        e_map = normalize_map_fast(e_map[:H, :W])

        outs.append(e_map[:, :, None])

    return np.concatenate(outs, axis=2).astype(np.float32)


def make_dct_input(
    arr_bgr: np.ndarray,
    freq_in: str = "ycbcr",
    block_energy: str = "ac",
):
    if arr_bgr.max() > 1.0:
        bgr01 = arr_bgr.astype(np.float32) / 255.0
    else:
        bgr01 = arr_bgr.astype(np.float32)

    feat = extract_block_dct_energy(
        bgr01=bgr01,
        block=8,
        freq_in=freq_in,
        energy_mode=block_energy,
    )

    return feat.transpose(2, 0, 1).astype(np.float32)


def calc_dct_channels(freq_in: str) -> int:
    if freq_in == "y":
        return 1
    if freq_in == "ycbcr":
        return 3
    raise ValueError("freq_in must be 'y' or 'ycbcr'")


def make_rgb_input(arr_rgb_uint8: np.ndarray):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])
    return transform(Image.fromarray(arr_rgb_uint8))


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class FFPPTriDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        compression: str,
        img_size: int,
        streams: List[str],
        args,
        clip_processor: CLIPImageProcessor = None,
    ):
        self.samples: List[Tuple[str, int]] = []
        self.img_size = img_size
        self.streams = streams
        self.args = args
        self.clip_processor = clip_processor
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

        img_resized = self.resize(img)
        arr_rgb = np.array(img_resized).astype(np.float32)
        arr_rgb_uint8 = arr_rgb.astype(np.uint8)
        arr_bgr = arr_rgb[:, :, ::-1].copy()

        item: Dict[str, torch.Tensor] = {}

        if "rgb" in self.streams:
            item["rgb"] = make_rgb_input(arr_rgb_uint8)

        if "wavelet" in self.streams:
            wav = make_wavelet_input(
                arr_bgr=arr_bgr,
                wavelet=self.args.wavelet,
                level=self.args.wavelet_level,
                wavelet_type=self.args.wavelet_type,
                wavelet_gray=self.args.wavelet_gray,
                subband=self.args.subband,
                robust=(not self.args.no_robust_norm),
            )
            wav = np.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=0.0)
            item["wavelet"] = torch.from_numpy(wav.astype(np.float32))

        if "dct" in self.streams:
            dct = make_dct_input(
                arr_bgr=arr_bgr,
                freq_in=self.args.freq_in,
                block_energy=self.args.block_energy,
            )
            dct = np.nan_to_num(dct, nan=0.0, posinf=1.0, neginf=0.0)
            item["dct"] = torch.from_numpy(dct.astype(np.float32))

        if "semantic" in self.streams:
            if self.clip_processor is None:
                raise ValueError("clip_processor is required for semantic stream")

            item["semantic"] = self.clip_processor(
                images=img,
                return_tensors="pt"
            )["pixel_values"].squeeze(0)

        item["label"] = torch.tensor(label, dtype=torch.long)
        return item


def tri_collate(batch):
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    keys = batch[0].keys()
    out = {}

    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    return out


# ---------------------------------------------------------
# Branches
# ---------------------------------------------------------
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

        return {
            "feat": pooled,
            "feat_map": feat_map,
        }


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

        return {
            "feat": pooled,
            "feat_map": feat_map,
        }


class CLIPSemanticFeatureExtractor(nn.Module):
    def __init__(self, clip_backbone: str, freeze: bool = True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_backbone)
        self.feat_dim = self.clip.config.projection_dim
        self.map_dim = self.feat_dim
        self.freeze = freeze

        if freeze:
            for p in self.clip.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        if self.freeze:
            with torch.no_grad():
                feat = self.clip.get_image_features(pixel_values=pixel_values)
        else:
            feat = self.clip.get_image_features(pixel_values=pixel_values)

        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        return {
            "feat": feat.float(),
            "token": feat.float().unsqueeze(1),
        }


# ---------------------------------------------------------
# Multi-stream cross-attention model
# ---------------------------------------------------------
class TriStreamCrossAttentionModel(nn.Module):
    def __init__(
        self,
        streams: List[str],
        args,
        wavelet_in_ch: int,
        dct_in_ch: int,
    ):
        super().__init__()

        self.streams = streams
        self.embed_dim = args.embed_dim

        self.main_streams = [s for s in ["rgb", "wavelet"] if s in streams]
        self.aux_streams = [s for s in ["dct", "semantic"] if s in streams]

        if len(self.main_streams) == 0:
            raise ValueError(
                "At least one main stream is required. "
                "Expected one of: rgb, wavelet"
            )

        if len(self.aux_streams) == 0:
            raise ValueError(
                "At least one auxiliary stream is required. "
                "Expected one of: dct, semantic"
            )

        print(f"▶ Main streams: {self.main_streams}")
        print(f"▶ Aux streams : {self.aux_streams}")

        self.branches = nn.ModuleDict()
        self.token_proj = nn.ModuleDict()
        self.global_proj = nn.ModuleDict()

        if "rgb" in streams:
            self.branches["rgb"] = ConvNeXtFeatureExtractor(in_chans=3)

        if "wavelet" in streams:
            self.branches["wavelet"] = ResNetFeatureExtractor(
                in_ch=wavelet_in_ch,
                pretrained=args.resnet_pretrained_wavelet,
            )

        if "dct" in streams:
            self.branches["dct"] = ConvNeXtFeatureExtractor(in_chans=dct_in_ch)

        if "semantic" in streams:
            self.branches["semantic"] = CLIPSemanticFeatureExtractor(
                clip_backbone=args.clip_backbone,
                freeze=(not args.finetune_clip),
            )

        for s in streams:
            branch = self.branches[s]

            if s == "semantic":
                self.token_proj[s] = nn.Linear(branch.feat_dim, args.embed_dim)
            else:
                self.token_proj[s] = nn.Conv2d(
                    branch.map_dim,
                    args.embed_dim,
                    kernel_size=1,
                )

            self.global_proj[s] = nn.Linear(branch.feat_dim, args.embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            batch_first=True,
        )

        fusion_dim = args.embed_dim * (len(streams) + 1)
        self.norm = nn.LayerNorm(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 2),
        )

        self.last_attn_weights = None

    def _make_tokens(self, stream_name: str, out: Dict[str, torch.Tensor]):
        if stream_name == "semantic":
            return self.token_proj[stream_name](out["token"])

        tok = self.token_proj[stream_name](out["feat_map"])
        tok = tok.flatten(2).transpose(1, 2)
        return tok

    def forward(self, batch: Dict[str, torch.Tensor]):
        outs = {}
        tokens = {}
        globals_ = {}

        for s in self.streams:
            outs[s] = self.branches[s](batch[s])
            tokens[s] = self._make_tokens(s, outs[s])
            globals_[s] = self.global_proj[s](outs[s]["feat"])

        query_tokens = torch.cat([tokens[s] for s in self.main_streams], dim=1)
        kv_tokens = torch.cat([tokens[s] for s in self.aux_streams], dim=1)

        attn_out, attn_weights = self.attn(
            query=query_tokens,
            key=kv_tokens,
            value=kv_tokens,
            need_weights=True,
            average_attn_weights=False,
        )

        self.last_attn_weights = attn_weights

        attn_global = attn_out.mean(dim=1)

        fused_list = [globals_[s] for s in self.streams]
        fused_list.append(attn_global)

        fused = torch.cat(fused_list, dim=1)
        fused = self.norm(fused)

        logits = self.classifier(fused)

        return {
            "logits": logits,
            "fused_feat": fused,
            "attn_weights": attn_weights,
            "main_streams": self.main_streams,
            "aux_streams": self.aux_streams,
            "query_tokens": query_tokens,
            "kv_tokens": kv_tokens,
            "attn_out": attn_out,
        }


def build_model(args, streams: List[str]):
    wavelet_in_ch = calc_wavelet_channels(
        gray=args.wavelet_gray,
        subband=args.subband,
        level=args.wavelet_level,
    )

    dct_in_ch = calc_dct_channels(args.freq_in)

    model = TriStreamCrossAttentionModel(
        streams=streams,
        args=args,
        wavelet_in_ch=wavelet_in_ch,
        dct_in_ch=dct_in_ch,
    )

    print(
        f"▶ Multi-stream cfg | streams={streams} | "
        f"wavelet_ch={wavelet_in_ch} | dct_ch={dct_in_ch}"
    )

    return model


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
@torch.no_grad()
def compute_metrics(model, loader, device, desc="Validation"):
    model.eval()

    y_true, y_pred, y_score = [], [], []

    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        if batch is None:
            continue

        label = batch.pop("label").to(device)

        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        out = model(batch)
        logits = out["logits"]

        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)

        y_true.extend(label.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        y_score.extend(prob.cpu().tolist())

        pbar.set_postfix(samples=len(y_true))

    if len(y_true) == 0:
        return {
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1_macro": float("nan"),
            "f1_binary": float("nan"),
            "auc": float("nan"),
            "confusion_matrix": [[0, 0], [0, 0]],
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1_macro,
        "f1_binary": f1_binary,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_eval_outputs(val_m, metrics_csv: Path, metrics_json: Path, epoch_label="eval_only"):
    row = {
        "epoch": epoch_label,
        "val_accuracy": val_m["acc"],
        "val_precision": val_m["precision"],
        "val_recall": val_m["recall"],
        "val_f1_binary": val_m["f1_binary"],
        "val_f1_macro": val_m["f1_macro"],
        "val_auc": val_m["auc"],
        "val_tn": val_m["tn"],
        "val_fp": val_m["fp"],
        "val_fn": val_m["fn"],
        "val_tp": val_m["tp"],
    }
    pd.DataFrame([row]).to_csv(metrics_csv, index=False)

    pd.Series({
        "epoch": epoch_label,
        "accuracy": val_m["acc"],
        "precision": val_m["precision"],
        "recall": val_m["recall"],
        "f1_binary": val_m["f1_binary"],
        "f1_macro": val_m["f1_macro"],
        "auc": val_m["auc"],
        "tn": val_m["tn"],
        "fp": val_m["fp"],
        "fn": val_m["fn"],
        "tp": val_m["tp"],
        "confusion_matrix": str(val_m["confusion_matrix"]),
    }).to_json(metrics_json, force_ascii=False, indent=2)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--compression", type=str, default="raw")
    parser.add_argument("--img-size", type=int, default=224)

    parser.add_argument(
        "--streams",
        type=str,
        required=True,
        help="e.g. rgb,wavelet,dct / rgb,wavelet,semantic"
    )
    parser.add_argument(
        "--query-stream",
        type=str,
        default="rgb",
        choices=["rgb", "wavelet", "dct", "semantic"]
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--monitor", choices=["f1_binary", "f1_macro", "auc"], default="f1_binary")
    parser.add_argument("--checkpoint", type=str, default="./ckpt_tri")

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--wavelet", type=str, default="sym4", choices=["haar", "sym4", "db4", "db8"])
    parser.add_argument("--wavelet-level", type=int, default=2, choices=[1, 2])
    parser.add_argument("--wavelet-type", type=str, default="swt", choices=["dwt", "swt"])
    parser.add_argument("--subband", type=str, default="ll_energy", choices=["ll", "high", "ll_energy"])
    parser.add_argument("--wavelet-gray", action="store_true")
    parser.add_argument("--no-robust-norm", action="store_true")

    parser.add_argument("--dct-mode", type=str, default="block", choices=["block"])
    parser.add_argument("--freq-in", type=str, default="ycbcr", choices=["y", "ycbcr"])
    parser.add_argument("--block-energy", type=str, default="ac", choices=["ac", "total"])

    parser.add_argument("--clip-backbone", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--finetune-clip", action="store_true")

    parser.add_argument("--resnet-pretrained-wavelet", action="store_true")

    parser.add_argument("--rgb-ckpt", type=str, default=None)
    parser.add_argument("--wavelet-ckpt", type=str, default=None)
    parser.add_argument("--dct-ckpt", type=str, default=None)

    parser.add_argument("--freeze-rgb", action="store_true")
    parser.add_argument("--freeze-wavelet", action="store_true")
    parser.add_argument("--freeze-dct", action="store_true")
    parser.add_argument("--freeze-semantic", action="store_true")

    # Added
    parser.add_argument("--resume", type=str, default=None,
                        help="Full training checkpoint path to resume/evaluate")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load checkpoint and run validation only")

    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streams = parse_streams(args.streams)
    tag = make_tag(streams)

    print(f"▶ Device: {device}")
    print(f"▶ Streams: {streams}")
    print("▶ Main streams are automatically set to RGB/Wavelet if included.")
    print("▶ Aux streams are automatically set to DCT/Semantic if included.")

    clip_processor = None
    if "semantic" in streams:
        clip_processor = CLIPImageProcessor.from_pretrained(args.clip_backbone)

    ds = FFPPTriDataset(
        root_dir=args.data_dir,
        compression=args.compression,
        img_size=args.img_size,
        streams=streams,
        args=args,
        clip_processor=clip_processor,
    )

    tr_n = int(0.8 * len(ds))
    va_n = len(ds) - tr_n

    tr_ds, va_ds = random_split(
        ds,
        [tr_n, va_n],
        generator=torch.Generator().manual_seed(args.seed),
    )

    tr_ld = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=tri_collate,
        generator=torch.Generator().manual_seed(args.seed),
    )

    va_ld = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=tri_collate,
    )

    model = build_model(args, streams).to(device)

    # Optional branch-only checkpoint loading
    load_requested_branch_checkpoints(
        model=model,
        args=args,
        streams=streams,
        device=device,
    )

    if args.freeze_rgb and "rgb" in streams:
        for p in model.branches["rgb"].parameters():
            p.requires_grad = False
        print("▶ RGB branch frozen")

    if args.freeze_wavelet and "wavelet" in streams:
        for p in model.branches["wavelet"].parameters():
            p.requires_grad = False
        print("▶ Wavelet branch frozen")

    if args.freeze_dct and "dct" in streams:
        for p in model.branches["dct"].parameters():
            p.requires_grad = False
        print("▶ DCT branch frozen")

    if args.freeze_semantic and "semantic" in streams:
        for p in model.branches["semantic"].parameters():
            p.requires_grad = False
        print("▶ Semantic branch frozen")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"▶ Trainable params: {n_trainable:,} / {n_total:,}")

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / f"best_tri_{tag}.pth"
    last_ckpt = ckpt_dir / f"last_tri_{tag}.pth"
    earlystop_ckpt = ckpt_dir / f"earlystop_tri_{tag}.pth"

    metrics_csv = ckpt_dir / f"val_metrics_tri_{tag}.csv"
    best_metrics_json = ckpt_dir / f"best_val_metrics_tri_{tag}.json"
    last_metrics_json = ckpt_dir / f"last_val_metrics_tri_{tag}.json"
    eval_only_json = ckpt_dir / f"eval_only_val_metrics_tri_{tag}.json"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_score = -1.0

    if args.resume:
        loaded_epoch, loaded_best = load_full_checkpoint(
            model=model,
            optimizer=optimizer if not args.eval_only else None,
            resume_path=args.resume,
            device=device,
            strict=False,
            load_optimizer=(not args.eval_only),
        )
        start_epoch = loaded_epoch + 1
        if loaded_best is not None:
            best_score = float(loaded_best)

    if args.eval_only:
        val_m = compute_metrics(model, va_ld, device, desc="Eval-only Validation")

        print("\n=== Eval-only Validation Metrics ===")
        print(
            f"Acc={val_m['acc']:.4f} "
            f"Prec={val_m['precision']:.4f} "
            f"Rec={val_m['recall']:.4f} "
            f"F1-macro={val_m['f1_macro']:.4f} "
            f"F1-binary={val_m['f1_binary']:.4f} "
            f"AUC={val_m['auc']:.4f} "
            f"| CM=[[{val_m['tn']}, {val_m['fp']}], [{val_m['fn']}, {val_m['tp']}]]"
        )

        save_eval_outputs(
            val_m=val_m,
            metrics_csv=metrics_csv,
            metrics_json=eval_only_json,
            epoch_label="eval_only",
        )

        print(f"Eval-only validation CSV: {metrics_csv}")
        print(f"Eval-only validation JSON: {eval_only_json}")
        return

    patience_counter = 0
    history_rows = []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        run_loss = 0.0
        n_batches = 0

        pbar = tqdm(tr_ld, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            if batch is None:
                continue

            label = batch.pop("label").to(device)

            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out = model(batch)
            loss = criterion(out["logits"], label)

            loss.backward()
            optimizer.step()

            run_loss += float(loss.item())
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_m = compute_metrics(model, va_ld, device, desc=f"Validation {epoch}/{args.epochs}")
        score = val_m[args.monitor]

        history_rows.append({
            "epoch": epoch,
            "val_accuracy": val_m["acc"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1_binary": val_m["f1_binary"],
            "val_f1_macro": val_m["f1_macro"],
            "val_auc": val_m["auc"],
            "val_tn": val_m["tn"],
            "val_fp": val_m["fp"],
            "val_fn": val_m["fn"],
            "val_tp": val_m["tp"],
        })

        pd.DataFrame(history_rows).to_csv(metrics_csv, index=False)

        avg_loss = run_loss / max(1, n_batches)

        print(
            f"[Epoch {epoch}] "
            f"loss={avg_loss:.4f} "
            f"Acc={val_m['acc']:.4f} "
            f"Prec={val_m['precision']:.4f} "
            f"Rec={val_m['recall']:.4f} "
            f"F1-macro={val_m['f1_macro']:.4f} "
            f"F1-binary={val_m['f1_binary']:.4f} "
            f"AUC={val_m['auc']:.4f} "
            f"| CM=[[{val_m['tn']}, {val_m['fp']}], [{val_m['fn']}, {val_m['tp']}]]"
        )

        save_obj = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_score": best_score,
            "streams": streams,
            "query_stream": args.query_stream,
            "args": vars(args),
            "val_metrics": {
                "accuracy": val_m["acc"],
                "precision": val_m["precision"],
                "recall": val_m["recall"],
                "f1_binary": val_m["f1_binary"],
                "f1_macro": val_m["f1_macro"],
                "auc": val_m["auc"],
                "confusion_matrix": val_m["confusion_matrix"],
                "tn": val_m["tn"],
                "fp": val_m["fp"],
                "fn": val_m["fn"],
                "tp": val_m["tp"],
            },
        }

        atomic_torch_save(save_obj, last_ckpt)

        pd.Series({
            "epoch": epoch,
            "accuracy": val_m["acc"],
            "precision": val_m["precision"],
            "recall": val_m["recall"],
            "f1_binary": val_m["f1_binary"],
            "f1_macro": val_m["f1_macro"],
            "auc": val_m["auc"],
            "tn": val_m["tn"],
            "fp": val_m["fp"],
            "fn": val_m["fn"],
            "tp": val_m["tp"],
            "confusion_matrix": str(val_m["confusion_matrix"]),
        }).to_json(last_metrics_json, force_ascii=False, indent=2)

        if score > best_score + args.min_delta:
            best_score = score
            patience_counter = 0

            save_obj["best_score"] = best_score
            atomic_torch_save(save_obj, best_ckpt)

            pd.Series({
                "epoch": epoch,
                "accuracy": val_m["acc"],
                "precision": val_m["precision"],
                "recall": val_m["recall"],
                "f1_binary": val_m["f1_binary"],
                "f1_macro": val_m["f1_macro"],
                "auc": val_m["auc"],
                "tn": val_m["tn"],
                "fp": val_m["fp"],
                "fn": val_m["fn"],
                "tp": val_m["tp"],
                "confusion_matrix": str(val_m["confusion_matrix"]),
            }).to_json(best_metrics_json, force_ascii=False, indent=2)

            print(f"▶ Best saved: {best_ckpt} | {args.monitor}={best_score:.4f}")

        else:
            patience_counter += 1
            print(f"▶ No improvement: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                save_obj["best_score"] = best_score
                atomic_torch_save(save_obj, earlystop_ckpt)

                print(f"▶ EarlyStopping triggered: {earlystop_ckpt}")
                break

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Validation metrics CSV: {metrics_csv}")
    print(f"Last validation metrics JSON: {last_metrics_json}")
    print(f"Best validation metrics JSON: {best_metrics_json}")


if __name__ == "__main__":
    main()