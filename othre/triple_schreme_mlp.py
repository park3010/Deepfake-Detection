#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, math, argparse, types, warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ===================== Utils =====================
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_device(dev_str: str) -> torch.device:
    s = str(dev_str).strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s.isdigit():
        return torch.device(f"cuda:{s}") if torch.cuda.is_available() else torch.device("cpu")
    if s.startswith("cuda"):
        return torch.device(s if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

def _load_state_any(path, device):
    obj = torch.load(path, map_location=device)
    return obj.get("model", obj) if isinstance(obj, dict) else obj

def _strip_prefix(sd: dict, prefix: str) -> dict:
    pref = prefix if prefix.endswith(".") else prefix + "."
    out = {}
    for k, v in sd.items():
        if k.startswith(pref):
            out[k[len(pref):]] = v
    return out

# ===================== Dataset (CSV) =====================
class ImageCSV(Dataset):
    """
    CSV: path,label
    path는 RGB 이미지 경로, label은 0/1
    """
    def __init__(self, csv_path: str, rgb_tfm, clip_tfm):
        self.items = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                p, y = row[0], int(row[1])
                self.items.append((p, y))
        self.rgb_tfm = rgb_tfm
        self.clip_tfm = clip_tfm

    def __len__(self): return len(self.items)

    def _pil_open(self, p: str):
        try:
            return Image.open(p).convert('RGB')
        except (UnidentifiedImageError, OSError):
            return None

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = self._pil_open(p)
        if img is None:
            # 손상 → caller에서 collate_fn으로 제거
            return None
        rgb = self.rgb_tfm(img) if self.rgb_tfm else transforms.ToTensor()(img)
        clip_img = self.clip_tfm(img) if self.clip_tfm else transforms.ToTensor()(img)
        return rgb, clip_img, y, p

def collate_drop_corrupt(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    a, b, c, d = zip(*batch)
    return torch.stack(a, 0), torch.stack(b, 0), torch.tensor(c, dtype=torch.long), list(d)

# ===================== Attention blocks (ESFCM / SE / CBAM) =====================
class ESFCM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8, residual_mode: str = "x_plus_xmulscale"):
        super().__init__()
        assert residual_mode in ("x_plus_scale","x_mul_scale","x_plus_xmulscale")
        mid = max(1, in_channels // reduction)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1    = nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False)
        self.relu     = nn.ReLU(inplace=True)
        self.conv_mid = nn.Conv2d(mid, mid, 3, 1, 1, bias=False)
        self.conv2    = nn.Conv2d(mid, in_channels, 1, 1, 0, bias=False)
        self.sigmoid  = nn.Sigmoid()
        self.residual_mode = residual_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.max_pool(x); m = self.conv1(m); m = self.relu(m); m = self.conv_mid(m); m = self.relu(m); m = self.conv2(m)
        a = self.avg_pool(x); a = self.conv1(a); a = self.relu(a); a = self.conv_mid(a); a = self.relu(a); a = self.conv2(a)
        scale = self.sigmoid(m + a)
        if self.residual_mode == "x_plus_scale":
            return x + scale
        elif self.residual_mode == "x_mul_scale":
            return x * scale
        else:
            return x + x * scale

class SE(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, residual_mode: str = "x_mul_scale"):
        super().__init__()
        assert residual_mode in ("x_plus_scale","x_mul_scale","x_plus_xmulscale")
        self.residual_mode = residual_mode
        mid = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.avg_pool(x); s = self.fc1(s); s = self.relu(s); s = self.fc2(s)
        scale = self.sigmoid(s)
        if self.residual_mode == "x_plus_scale":
            return x + scale
        elif self.residual_mode == "x_mul_scale":
            return x * scale
        else:
            return x + x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, residual_mode: str = "x_mul_scale", kernel_size: int = 7):
        super().__init__()
        assert residual_mode in ("x_plus_scale","x_mul_scale","x_plus_xmulscale")
        self.residual_mode = residual_mode
        mid = max(1, in_channels // reduction)
        # channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False)
        )
        self.sigmoid_c = nn.Sigmoid()
        # spatial
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x):
        # channel
        avg_c = self.mlp(self.avg_pool(x))
        max_c = self.mlp(self.max_pool(x))
        scale_c = self.sigmoid_c(avg_c + max_c)            # (B,C,1,1)
        x_c = x * scale_c
        # spatial
        avg_s = torch.mean(x_c, 1, keepdim=True)
        max_s, _ = torch.max(x_c, 1, keepdim=True)
        s = torch.cat([avg_s, max_s], 1)                   # (B,2,H,W)
        scale_s = self.sigmoid_s(self.conv_spatial(s))     # (B,1,H,W)

        if self.residual_mode == "x_plus_scale":
            scale = scale_c * scale_s
            return x + scale
        elif self.residual_mode == "x_mul_scale":
            return x * (scale_c * scale_s)
        else:
            return x + x * (scale_c * scale_s)

# ===================== Hook injector =====================
@torch.no_grad()
def _infer_feat_channels(model: nn.Module, device: Optional[torch.device] = None) -> int:
    if hasattr(model, "num_features") and isinstance(model.num_features, int) and model.num_features > 0:
        return int(model.num_features)
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
    x = x.to(device) if device is not None else x
    was = model.training; model.eval()
    try:
        if hasattr(model, "forward_features"):
            y = model.forward_features(x)
            if isinstance(y, torch.Tensor) and y.ndim == 4:
                return int(y.shape[1])
    finally:
        model.train(was)
    return 1280

def attach_block_before_head(model: nn.Module, block: nn.Module, name: str, device: Optional[torch.device]):
    # register module + wrap forward_features
    model.add_module(name, block)
    if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
        old_ff = model.forward_features
        def wrapped_forward_features(self, x):
            f = old_ff(x)
            if isinstance(f, torch.Tensor) and f.ndim == 4:
                f = getattr(self, name)(f)
            return f
        model.forward_features = types.MethodType(wrapped_forward_features, model)
    else:
        warnings.warn("model has no forward_features; attention not injected.")
    return model

def maybe_inject_rgb_attention(model: nn.Module, device: torch.device, args):
    c = _infer_feat_channels(model, device=device)
    if args.use_esfcm:
        blk = ESFCM(c, reduction=args.esfcm_reduction, residual_mode=args.esfcm_mode).to(device)
        attach_block_before_head(model, blk, "esfcm_before_head", device)
        print("[Hook] ESFCM injected")
    elif args.use_se:
        blk = SE(c, reduction=args.se_reduction, residual_mode=args.se_mode).to(device)
        attach_block_before_head(model, blk, "se_before_head", device)
        print("[Hook] SE injected")
    elif args.use_cbam:
        blk = CBAM(c, reduction=args.cbam_reduction, residual_mode=args.cbam_mode, kernel_size=args.cbam_kernel).to(device)
        attach_block_before_head(model, blk, "cbam_before_head", device)
        print("[Hook] CBAM injected")
    return model

# ===================== RGB encoder (timm) =====================
class RGBEncoder(nn.Module):
    """
    timm backbone → GAP → Linear(→256)
    """
    def __init__(self, backbone: str = "tf_efficientnet_b7", out_dim: int = 256, num_classes: int = 2):
        super().__init__()
        import timm
        self.net = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
        # replace classifier with identity, keep features
        if hasattr(self.net, "classifier"):
            in_ch = self.net.classifier.in_features
            self.net.classifier = nn.Identity()
        elif hasattr(self.net, "fc"):
            in_ch = self.net.fc.in_features
            self.net.fc = nn.Identity()
        else:
            # fallback: infer channels with dummy
            in_ch = _infer_feat_channels(self.net)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, out_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # try to get spatial feats
        if hasattr(self.net, "forward_features"):
            f = self.net.forward_features(x)
            if f.ndim == 4:
                f = self.pool(f).flatten(1)
            else:
                # some models return already pooled
                pass
        else:
            f = self.net.forward(x)  # may be pooled
        if f.ndim > 2:
            f = f.flatten(1)
        z = self.proj(f)  # (B,256)
        return z

    def forward(self, x):
        return self.forward_features(x)

# ===================== Frequency encoder (Tri-branch) =====================
# ---- low-level extractors using numpy/opencv/pywt
import cv2
import pywt

def extract_fft(bgr: np.ndarray) -> np.ndarray:
    chans = []
    for ch in cv2.split(bgr):
        dft = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        shift = np.fft.fftshift(dft)
        mag = cv2.magnitude(shift[:,:,0], shift[:,:,1])
        chans.append((20*np.log(mag+1)).astype(np.float32))
    return np.stack(chans, 2)  # HxWx3

def extract_dct(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    dctm = cv2.dct(gray)
    return dctm.astype(np.float32)  # HxW

def extract_wavelet(bgr: np.ndarray, wave='db2', level=2) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)/255.0
    coeffs = pywt.wavedec2(gray, wavelet=wave, level=level)
    # 사용: 최상위 approximation만
    A, *details = coeffs
    A = cv2.resize(A, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return A.astype(np.float32)

class FreqBranchConv(nn.Module):
    """
    입력: 4x224x224 (RGB[0..1] 3ch + freq 1ch)
    출력: 256
    """
    def __init__(self, in_ch=4, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        f = self.net(x).flatten(1)
        return self.proj(f)

class FreqTriEncoder(nn.Module):
    """
    RGB 텐서(0..1 정규화 전제 또는 [-1,1]이어도 스케일만 복원)로부터
    DCT/FFT/Wavelet map을 만들어 각 브랜치(4ch 입력)에 통과 → 256,256,256
    combine='mean'이면 평균으로 256, 'concat'이면 concat 후 proj로 256.
    """
    def __init__(self, out_dim=256, combine: str = "mean"):
        super().__init__()
        assert combine in ("mean","concat")
        self.combine = combine
        self.dct_enc = FreqBranchConv(in_ch=4, out_dim=out_dim)
        self.fft_enc = FreqBranchConv(in_ch=4, out_dim=out_dim)
        self.wav_enc = FreqBranchConv(in_ch=4, out_dim=out_dim)
        if combine == "concat":
            self.fuse = nn.Linear(out_dim*3, out_dim)

    @torch.no_grad()
    def _build_freq4(self, rgb_bchw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        rgb_bchw: (B,3,224,224) in [-1,1] or [0,1]
        반환: dct_in, fft_in, wav_in  (각각 4ch)
        """
        B, C, H, W = rgb_bchw.shape
        # to uint8 BGR
        rgb = rgb_bchw
        # unnormalize to [0,1]
        rgb01 = (rgb + 1)/2 if rgb.min() < 0 else rgb.clamp(0,1)
        rgb_np = (rgb01.permute(0,2,3,1).cpu().numpy()*255.0).astype(np.uint8)

        dct_list, fft_list, wav_list = [], [], []
        for i in range(B):
            bgr = rgb_np[i][:,:,::-1]  # RGB->BGR
            # DCT
            dctm = extract_dct(bgr)[..., None]              # HxWx1
            # FFT
            fft3 = extract_fft(bgr).astype(np.float32)      # HxWx3
            fftm = np.mean(fft3, axis=2, keepdims=True)     # HxWx1
            # Wav
            wavm = extract_wavelet(bgr)[..., None]          # HxWx1
            # stack with BGR(0..1)
            bgr_f = (bgr.astype(np.float32)/255.0)
            dct_in = np.concatenate([bgr_f, dctm], 2)
            fft_in = np.concatenate([bgr_f, fftm], 2)
            wav_in = np.concatenate([bgr_f, wavm], 2)
            dct_list.append(dct_in.transpose(2,0,1))
            fft_list.append(fft_in.transpose(2,0,1))
            wav_list.append(wav_in.transpose(2,0,1))

        dct = torch.from_numpy(np.stack(dct_list,0)).float().to(rgb_bchw.device)
        fft = torch.from_numpy(np.stack(fft_list,0)).float().to(rgb_bchw.device)
        wav = torch.from_numpy(np.stack(wav_list,0)).float().to(rgb_bchw.device)
        return dct, fft, wav

    def forward(self, rgb_bchw: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dct_in, fft_in, wav_in = self._build_freq4(rgb_bchw)
        zd = self.dct_enc(dct_in)
        zf = self.fft_enc(fft_in)
        zw = self.wav_enc(wav_in)
        if self.combine == "mean":
            return (zd + zf + zw) / 3.0
        else:
            z = torch.cat([zd, zf, zw], dim=1)
            return self.fuse(z)

# ===================== Semantic (CLIP ViT-B/16) =====================
class SemanticCLIP(nn.Module):
    """
    CLIP ViT-B/16 image encoder → projector(256)
    freeze_ratio: [0..1], 비율만큼 앞쪽 블록 freeze
    """
    def __init__(self, out_dim=256, freeze_ratio: float = 0.5):
        super().__init__()
        self.out_dim = out_dim
        self.freeze_ratio = float(max(0.0, min(1.0, freeze_ratio)))
        self.encoder, self.preprocess, self.feat_dim = self._build_clip_encoder()
        self.proj = nn.Linear(self.feat_dim, out_dim)

        self._partial_freeze()

    def _build_clip_encoder(self):
        # 우선 open_clip 시도 → 실패 시 timm ViT-B/16
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained='openai'
            )
            feat_dim = model.visual.output_dim if hasattr(model.visual,"output_dim") else 768
            class ImgEnc(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x): return self.m.encode_image(x)
            enc = ImgEnc(model).eval()
            return enc, preprocess, feat_dim
        except Exception:
            import timm
            vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            # timm vit forward → (B,768)
            def _ppil(img):
                tfm = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3),
                ])
                return tfm(img)
            class ImgEnc(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x): return self.m(x)
            return ImgEnc(vit).eval(), _ppil, 768

    def _partial_freeze(self):
        # encoder 내부 블록 수를 추정해서 앞쪽 일부 freeze
        # open_clip이든 timm이든 parameter list를 블록 단위로 가정하지 못하므로
        # 간단히 "앞쪽 비율"만큼의 파라미터들을 freeze 한다.
        params = list(self.encoder.parameters())
        k = int(len(params) * self.freeze_ratio)
        for p in params[:k]:
            p.requires_grad = False

    def forward(self, clip_img_bchw: torch.Tensor) -> torch.Tensor:
        f = self.encoder(clip_img_bchw)   # (B,feat_dim)
        if f.ndim > 2: f = f.flatten(1)
        return self.proj(f)               # (B,256)

# ===================== Fusion (Cross-Attention 1-layer) =====================
class CrossAttnFusion(nn.Module):
    """
    z_rgb, z_freq, z_sem (각 256) → rgb를 query로, freq|sem을 key/value로 1층 MHA
    """
    def __init__(self, dim=256, heads=4, num_classes=2, p=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(dim, num_classes)
        )

    def forward(self, z_rgb, z_freq, z_sem):
        # tokens: [rgb,q]=1, [k/v]=2
        Q = self.q_proj(z_rgb).unsqueeze(0)                 # (1,B,256)
        KV = torch.stack([self.k_proj(z_freq), self.k_proj(z_sem)], dim=0)  # (2,B,256)
        VV = torch.stack([self.v_proj(z_freq), self.v_proj(z_sem)], dim=0)  # (2,B,256)
        out, _ = self.mha(Q, KV, VV)                        # (1,B,256)
        z = self.norm(out.squeeze(0))                       # (B,256)
        logits = self.head(z)                               # (B,2)
        return logits, z

# ===================== Whole model =====================
class MultiBranchFusion(nn.Module):
    def __init__(self,
                 rgb_backbone: str = "tf_efficientnet_b7",
                 attn_rgb: dict = None,
                 freq_combine: str = "mean",
                 clip_freeze_ratio: float = 0.5,
                 dim: int = 256,
                 num_classes: int = 2,
                 heads: int = 4):
        super().__init__()
        self.rgb = RGBEncoder(rgb_backbone, out_dim=dim, num_classes=num_classes)
        self.freq = FreqTriEncoder(out_dim=dim, combine=freq_combine)
        self.sem  = SemanticCLIP(out_dim=dim, freeze_ratio=clip_freeze_ratio)
        self.fuse = CrossAttnFusion(dim=dim, heads=heads, num_classes=num_classes)

        # 저장용: RGB timm 모델 핸들 (hook 주입용)
        self.rgb_timm = self.rgb.net

        # attn_rgb 설정(외부에서 실제 주입)
        self.attn_rgb_cfg = attn_rgb or {}

    def forward(self, rgb_img, clip_img):
        z_rgb = self.rgb(rgb_img)               # (B,256)
        z_freq = self.freq(rgb_img)             # (B,256)  (freq 생성은 내부에서)
        z_sem = self.sem(clip_img)              # (B,256)
        logits, z = self.fuse(z_rgb, z_freq, z_sem)
        return logits, (z_rgb, z_freq, z_sem, z)

# ===================== CKPT Loaders (backbone-level & branch-level) =====================
def load_branch_ckpts(model: MultiBranchFusion,
                      ckpt_rgb: str = "", ckpt_freq: str = "", ckpt_sem: str = "",
                      strict: bool = False):
    dev = next(model.parameters()).device
    if ckpt_rgb:
        sd = _load_state_any(ckpt_rgb, dev)
        # 예상 prefix 후보
        tried = False
        missing, unexpected = model.rgb.load_state_dict(sd, strict=False)
        tried = True
        if not tried or (missing and len(missing) > 10):
            part = _strip_prefix(sd, "rgb")
            if part:
                missing, unexpected = model.rgb.load_state_dict(part, strict=strict)
        print(f"[LOAD] rgb <- {ckpt_rgb}  missing={len(missing)} unexpected={len(unexpected)}")

    if ckpt_freq:
        sd = _load_state_any(ckpt_freq, dev)
        missing, unexpected = model.freq.load_state_dict(sd, strict=False)
        if missing:
            part = _strip_prefix(sd, "freq")
            if part:
                missing, unexpected = model.freq.load_state_dict(part, strict=strict)
        print(f"[LOAD] freq <- {ckpt_freq}  missing={len(missing)} unexpected={len(unexpected)}")

    if ckpt_sem:
        sd = _load_state_any(ckpt_sem, dev)
        missing, unexpected = model.sem.load_state_dict(sd, strict=False)
        if missing:
            part = _strip_prefix(sd, "sem")
            if part:
                missing, unexpected = model.sem.load_state_dict(part, strict=strict)
        print(f"[LOAD] sem  <- {ckpt_sem}  missing={len(missing)} unexpected={len(unexpected)}")

def load_freq_branch_ckpts(model: MultiBranchFusion,
                           ckpt_dct: str = "", ckpt_fft: str = "", ckpt_wav: str = "",
                           strict: bool = False):
    dev = next(model.parameters()).device
    enc = model.freq

    def _try(name: str, path: str, module: nn.Module):
        if not path: return
        sd = _load_state_any(path, dev)
        missing, unexpected = module.load_state_dict(sd, strict=False)
        ok = len(missing) < len(list(module.state_dict().keys())) or not missing
        if not ok:
            for pref in (f"freq.{name}.", f"{name}."):
                part = _strip_prefix(sd, pref)
                if part:
                    missing, unexpected = module.load_state_dict(part, strict=strict)
                    ok = True
                    break
        print(f"[FREQ-LOAD] {name} <- {path}  missing={len(missing)} unexpected={len(unexpected)}")

    if ckpt_dct: _try("dct_enc", ckpt_dct, enc.dct_enc)
    if ckpt_fft: _try("fft_enc", ckpt_fft, enc.fft_enc)
    if ckpt_wav: _try("wav_enc", ckpt_wav, enc.wav_enc)

# ===================== Metrics =====================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def metrics_from_logits(y_true: List[int], logits_all: List[np.ndarray], average: str = "binary") -> Dict[str, float]:
    probs = torch.softmax(torch.from_numpy(np.concatenate(logits_all,0)), dim=1).numpy()
    p_fake = probs[:,1]
    preds = (p_fake >= 0.5).astype(np.int32)
    out = {
        "acc":  accuracy_score(y_true, preds) if y_true else 0.0,
        "f1":   f1_score(y_true, preds, average=average) if y_true else 0.0,
        "prec": precision_score(y_true, preds, average=average, zero_division=0) if y_true else 0.0,
        "recall": recall_score(y_true, preds, average=average, zero_division=0) if y_true else 0.0,
        "auc":  None
    }
    try:
        out["auc"] = roc_auc_score(y_true, p_fake)
    except Exception:
        out["auc"] = None
    return out

# ===================== EarlyStopping =====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False, path='checkpoint_es.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best = float('inf')
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            torch.save({"model": model.state_dict()}, self.path)
            if self.verbose: print(f"[ES] improved -> {val_loss:.4f} (saved {self.path})")
        else:
            self.counter += 1
            if self.verbose: print(f"[ES] no improve ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True

# ===================== Train / Eval =====================
def train_one_epoch(model, opt, crit, ld, device):
    model.train()
    total = 0.0
    for b in tqdm(ld, desc="train", leave=False):
        if b is None: continue
        rgb, clip_img, y, _ = b
        rgb, clip_img, y = rgb.to(device), clip_img.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits, _ = model(rgb, clip_img)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(1, len(ld))

@torch.no_grad()
def evaluate(model, ld, device):
    model.eval()
    ys, logits_all = [], []
    for b in tqdm(ld, desc="eval", leave=False):
        if b is None: continue
        rgb, clip_img, y, _ = b
        rgb, clip_img = rgb.to(device), clip_img.to(device)
        logits, _ = model(rgb, clip_img)
        ys.extend(y.numpy().tolist())
        logits_all.append(logits.detach().cpu().numpy())
    m = metrics_from_logits(ys, logits_all, average="binary")
    return m

# ===================== Main =====================
def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data-csv-train", type=str, required=True, help="CSV: path,label")
    p.add_argument("--data-csv-val",   type=str, required=True, help="CSV: path,label")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # device / ckpt io
    p.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda[:N] or N")
    p.add_argument("--outdir", type=str, default="./checkpoints/fusion")
    p.add_argument("--tag", type=str, default="effb7_freqtri_clip_vitb16_ca")
    p.add_argument("--resume", type=str, default="", help="resume from last_epoch ckpt (saved here)")

    # RGB backbone & hook
    p.add_argument("--rgb-backbone", type=str, default="tf_efficientnet_b7")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-esfcm", action="store_true")
    g.add_argument("--use-se", action="store_true")
    g.add_argument("--use-cbam", action="store_true")
    p.add_argument("--esfcm-reduction", type=int, default=8)
    p.add_argument("--esfcm-mode", choices=["x_plus_scale","x_mul_scale","x_plus_xmulscale"], default="x_plus_xmulscale")
    p.add_argument("--se-reduction", type=int, default=16)
    p.add_argument("--se-mode", choices=["x_plus_scale","x_mul_scale","x_plus_xmulscale"], default="x_plus_xmulscale")
    p.add_argument("--cbam-reduction", type=int, default=16)
    p.add_argument("--cbam-mode", choices=["x_plus_scale","x_mul_scale","x_plus_xmulscale"], default="x_plus_xmulscale")
    p.add_argument("--cbam-kernel", type=int, default=7)

    # Freq
    p.add_argument("--freq-combine", choices=["mean","concat"], default="mean")

    # Semantic
    p.add_argument("--clip-freeze-ratio", type=float, default=0.5)

    # Fusion
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=2)

    # Unified ckpts
    p.add_argument("--ckpt-rgb", type=str, default="")
    p.add_argument("--ckpt-freq", type=str, default="")
    p.add_argument("--ckpt-sem", type=str, default="")

    # Branch ckpts (DCT/FFT/Wavelet)
    p.add_argument("--ckpt-dct", type=str, default="", help="DCT branch checkpoint")
    p.add_argument("--ckpt-fft", type=str, default="", help="FFT branch checkpoint")
    p.add_argument("--ckpt-wav", type=str, default="", help="Wavelet branch checkpoint")

    # ES
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=0.0)

    args = p.parse_args()
    seed_everything(args.seed)
    device = normalize_device(args.device)

    # transforms
    rgb_tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    # CLIP 전처리는 SemanticCLIP 내부에서 open_clip을 만들 때 가져오지만,
    # 학습 파이프라인 통일을 위해 여기서는 동일한 리사이즈/정규화 사용
    clip_tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # data
    train_ds = ImageCSV(args.data_csv_train, rgb_tfm, clip_tfm)
    val_ds   = ImageCSV(args.data_csv_val,   rgb_tfm, clip_tfm)
    pin = (device.type == "cuda")
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=pin, collate_fn=collate_drop_corrupt)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=max(1,args.workers//2),
                          pin_memory=pin, collate_fn=collate_drop_corrupt)

    # model
    model = MultiBranchFusion(
        rgb_backbone=args.rgb_backbone,
        attn_rgb=None,
        freq_combine=args.freq_combine,
        clip_freeze_ratio=args.clip_freeze_ratio,
        dim=args.dim,
        num_classes=args.num_classes,
        heads=args.heads
    ).to(device)

    # inject RGB attention block if requested
    maybe_inject_rgb_attention(model.rgb_timm, device, args)

    # ckpt load (unified)
    if args.ckpt_rgb or args.ckpt_freq or args.ckpt_sem:
        load_branch_ckpts(model, args.ckpt_rgb, args.ckpt_freq, args.ckpt_sem, strict=False)
    # ckpt load (per-branch) — unified 위에 덮어쓰기
    if args.ckpt_dct or args.ckpt_fft or args.ckpt_wav:
        load_freq_branch_ckpts(model, args.ckpt_dct, args.ckpt_fft, args.ckpt_wav, strict=False)

    # opt/crit
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()

    # out dir
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    ckpt_best = outdir / f"best_{tag}.pth"
    ckpt_es   = outdir / f"es_{tag}.pth"

    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta, verbose=True, path=str(ckpt_es))

    # resume
    start_ep = 1
    last_path = outdir / f"last_{tag}.pth"
    if args.resume and Path(args.resume).exists():
        st = torch.load(args.resume, map_location=device)
        model.load_state_dict(st["model"], strict=False)
        if "opt" in st: opt.load_state_dict(st["opt"])
        start_ep = int(st.get("epoch", 1)) + 1
        print(f"[RESUME] from {args.resume} (start_ep={start_ep})")
    elif last_path.exists():
        st = torch.load(last_path, map_location=device)
        model.load_state_dict(st["model"], strict=False)
        if "opt" in st: opt.load_state_dict(st["opt"])
        start_ep = int(st.get("epoch", 1)) + 1
        print(f"[RESUME] from {last_path} (start_ep={start_ep})")

    best_f1 = -1.0
    for ep in range(start_ep, args.epochs + 1):
        tr_loss = train_one_epoch(model, opt, crit, train_ld, device)
        m = evaluate(model, val_ld, device)
        print(f"[{ep}] loss:{tr_loss:.4f} | acc:{m['acc']:.4f} f1:{m['f1']:.4f} prec:{m['prec']:.4f} recall:{m['recall']:.4f} auc:{m['auc'] if m['auc'] is not None else 'N/A'}")

        # save last
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep}, last_path)

        # save epoch
        torch.save({"model": model.state_dict(), "epoch": ep}, outdir / f"epoch_{tag}_{ep:03d}.pth")

        # best on f1
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best_f1": best_f1}, ckpt_best)
            print(f"  ↑ best updated: F1={best_f1:.4f} -> {ckpt_best.name}")

        # ES on (1 - f1)
        early(1.0 - m["f1"], model)
        if early.early_stop:
            print(f"[EarlyStopping] stop. saved {ckpt_es.name}")
            break

    print(f"Done. Best F1={best_f1:.4f}. best_ckpt={ckpt_best}")

if __name__ == "__main__":
    main()
