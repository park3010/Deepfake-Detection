#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_freq.py
- FF++ mtcnn frame 기반 단일 스트림 학습 스크립트
  * stream: rgb | dct | fft
  * dct-mode:
      - block  : YCbCr 변환 후 block DCT(8x8) "밴드 분리 없이" 에너지 맵
                 (권장: AC 에너지 = 전체 에너지 - DC)
      - global : global DCT(전체 이미지 DCT) 후 저/중/고 주파수 밴드 분리
  * fft-mode:
      - global : global FFT(전체 이미지 FFT) 후 저/중/고 주파수 밴드 분리 (radial)
  * freq_in: y | ycbcr
      - y     : block=1ch, global=3ch(low/mid/high on Y)
      - ycbcr : block=3ch(Y,Cr,Cb), global=9ch(low/mid/high on Y,Cr,Cb)

- band option:
  * --band all|low|mid|high
    - global DCT/FFT에서만 적용됨
    - all : low/mid/high 3채널 동시 사용
    - low/mid/high : 해당 밴드만 남기고 나머지 0으로 (입력 채널 수는 동일 유지)

- backbone:
  * convnext_tiny (없으면 convnextv2_large fallback)
  * replknet_b    (없으면 RepLKNet31B fallback)

Speed optimizations (keep original structure):
1) Block DCT: python double-loop 제거 -> torch unfold + einsum (C @ block @ C^T)
2) Global FFT: radial mask(rr) 계산 캐시
3) Global DCT: DCT mask(rr) 계산 캐시 + torch matmul 기반( C @ img @ C^T )
4) (Optional) feature disk cache(.npy): --cache-dir

Checkpoint saving policy (disk-friendly):
- 매 epoch마다 "last"는 덮어쓰기 저장:  <exp_prefix>_last.pth
- val 기준 best 갱신 시에만 "best" 저장: <exp_prefix>_best.pth
- 학습 종료 후 "earlystop" 저장:
    - best가 있으면 best를 복사해서 <exp_prefix>_earlystop.pth 생성
    - best가 없으면(last만 존재) last를 복사해서 earlystop 생성
=> 최종적으로 각 실험 폴더에 (best/last/earlystop/meta.json)만 남도록 함.

Note:
- train mode에서도 (val_ratio>0이면) val split을 떼서 best/earlystop 판단에 사용
- val mode는 단독 평가(원하면 유지). 다만 본 실험 파이프라인은 train mode에서 best/earlystop을 만드는 것을 권장.
"""

import os, csv, re, glob, argparse, math, hashlib, json, random, shutil
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# ───────────────────────────────────────────────────────────
# Reproducibility / atomic save utils
# ───────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_torch_atomic(obj: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)  # atomic replace
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def copy_atomic(src: str, dst: str):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + f".tmp{os.getpid()}"
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ───────────────────────────────────────────────────────────
# 모델 import (프로젝트 상황에 맞춰 try/fallback)
# ───────────────────────────────────────────────────────────
try:
    from models.convnextv2 import convnextv2_tiny
    _HAS_CONVNEXT_TINY = True
except Exception:
    convnextv2_tiny = None
    _HAS_CONVNEXT_TINY = False

from models.convnextv2 import convnextv2_large  # fallback

try:
    from models.replknet import create_RepLKNetB
    _HAS_REPLK_B = True
except Exception:
    create_RepLKNetB = None
    _HAS_REPLK_B = False

from models.replknet import create_RepLKNet31B  # fallback


# ───────────────────────────────────────────────────────────
# 유틸: 색공간/정규화
# ───────────────────────────────────────────────────────────
def bgr_to_ycrcb01(bgr01: np.ndarray) -> np.ndarray:
    """bgr01: float32 [0,1] -> ycrcb float32 [0,1]"""
    ycrcb = cv2.cvtColor((bgr01 * 255.0).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    return ycrcb.astype(np.float32) / 255.0


def _normalize_map_fast(m: np.ndarray, eps: float = 1e-6, stride: int = 4) -> np.ndarray:
    """
    log + percentile minmax (robust) with subsampled percentile for speed.
    output in [0,1]
    """
    m = np.log1p(np.abs(m)).astype(np.float32)
    ms = m[::stride, ::stride].reshape(-1)
    lo, hi = np.percentile(ms, 1), np.percentile(ms, 99)
    m = np.clip(m, lo, hi)
    m = (m - lo) / (hi - lo + eps)
    return m.astype(np.float32)


def _apply_band_select(feat_hwc: np.ndarray, band: str, enabled: bool) -> np.ndarray:
    """
    feat_hwc: (H,W,C)
    band: all|low|mid|high
    enabled=False이면 그대로 반환 (예: block DCT는 밴드 분리 안 함)
    enabled=True (global DCT/FFT)에서만:
      - all: return as-is
      - else: keep only selected band channels, zero out others (shape 유지)
    Channel order per color channel: [low, mid, high]
    """
    if not enabled or band == "all":
        return feat_hwc
    assert band in ("low", "mid", "high")
    band_idx = {"low": 0, "mid": 1, "high": 2}[band]
    out = np.zeros_like(feat_hwc, dtype=np.float32)
    for base in range(0, feat_hwc.shape[2], 3):
        out[:, :, base + band_idx] = feat_hwc[:, :, base + band_idx]
    return out


# ───────────────────────────────────────────────────────────
# DCT basis (cache)
# ───────────────────────────────────────────────────────────
_DCT_MAT_CACHE = {}


def _get_dct_mat(N: int, device: str = "cpu") -> torch.Tensor:
    """
    Orthonormal DCT-II basis matrix (NxN). Cache per (N, device).
    """
    key = (N, device)
    if key in _DCT_MAT_CACHE:
        return _DCT_MAT_CACHE[key]

    k = torch.arange(N, dtype=torch.float32, device=device).view(N, 1)
    n = torch.arange(N, dtype=torch.float32, device=device).view(1, N)

    alpha = torch.ones((N,), dtype=torch.float32, device=device)
    alpha[0] = 1.0 / math.sqrt(2.0)

    C = math.sqrt(2.0 / N) * alpha.view(N, 1) * torch.cos((math.pi * (2.0 * n + 1.0) * k) / (2.0 * N))
    _DCT_MAT_CACHE[key] = C
    return C


# ───────────────────────────────────────────────────────────
# (A) Block DCT (NO band split)
# ───────────────────────────────────────────────────────────
def extract_block_dct_energy(
    bgr01: np.ndarray,
    block: int = 8,
    freq_in: str = "y",
    energy_mode: str = "ac",  # "ac" | "total"
) -> np.ndarray:
    ycrcb = bgr_to_ycrcb01(bgr01)
    chans = [ycrcb[:, :, 0]] if freq_in == "y" else [ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]]

    C = _get_dct_mat(block, device="cpu")
    Ct = C.t()

    outs = []
    for ch in chans:
        H, W = ch.shape
        Hp = (H + block - 1) // block * block
        Wp = (W + block - 1) // block * block

        pad = np.zeros((Hp, Wp), dtype=np.float32)
        pad[:H, :W] = ch.astype(np.float32)

        t = torch.from_numpy(pad)  # (Hp,Wp)
        blocks = t.unfold(0, block, block).unfold(1, block, block)  # (Hb,Wb,block,block)

        temp = torch.einsum("ij,abjk->abik", C, blocks)
        dct = torch.einsum("abik,kj->abij", temp, Ct)
        a = dct.abs()  # (Hb,Wb,block,block)

        total = a.sum(dim=(-1, -2))  # (Hb,Wb)
        if energy_mode == "ac":
            dc = a[:, :, 0, 0]
            energy = total - dc
        elif energy_mode == "total":
            energy = total
        else:
            raise ValueError(f"Unknown energy_mode: {energy_mode}")

        e_map = energy.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1).numpy()
        e_map = _normalize_map_fast(e_map[:H, :W])

        outs.append(e_map[:, :, None])

    return np.concatenate(outs, axis=2).astype(np.float32)


# ───────────────────────────────────────────────────────────
# (B) Global FFT bands: radial low/mid/high (mask cache)
# ───────────────────────────────────────────────────────────
_FFT_MASK_CACHE = {}


def _get_radial_masks(H: int, W: int, r1: float, r2: float):
    key = (H, W, float(r1), float(r2))
    if key in _FFT_MASK_CACHE:
        return _FFT_MASK_CACHE[key]

    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (rr.max() + 1e-6)

    m_low = (rr <= r1)
    m_mid = ((rr > r1) & (rr <= r2))
    m_high = (rr > r2)

    _FFT_MASK_CACHE[key] = (m_low, m_mid, m_high)
    return m_low, m_mid, m_high


def extract_fft_bands(
    bgr01: np.ndarray,
    freq_in: str = "y",
    r1: float = 0.12,
    r2: float = 0.35,
) -> np.ndarray:
    """
    Return:
      freq_in='y'    : (H,W,3)
      freq_in='ycbcr': (H,W,9)
    """
    ycrcb = bgr_to_ycrcb01(bgr01)
    chans = [ycrcb[:, :, 0]] if freq_in == "y" else [ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]]

    outs = []
    for ch in chans:
        H, W = ch.shape
        m_low, m_mid, m_high = _get_radial_masks(H, W, r1, r2)

        f = np.fft.fft2(ch.astype(np.float32))
        f = np.fft.fftshift(f)
        mag = np.log1p(np.abs(f)).astype(np.float32)

        low = _normalize_map_fast(mag * m_low)
        mid = _normalize_map_fast(mag * m_mid)
        high = _normalize_map_fast(mag * m_high)

        outs.append(np.stack([low, mid, high], axis=2))

    return np.concatenate(outs, axis=2).astype(np.float32)


# ───────────────────────────────────────────────────────────
# (C) Global DCT bands: full-image DCT -> radial masks -> band maps
# ───────────────────────────────────────────────────────────
_DCT_MASK_CACHE = {}


def _get_dct_radial_masks(H: int, W: int, r1: float, r2: float):
    key = (H, W, float(r1), float(r2))
    if key in _DCT_MASK_CACHE:
        return _DCT_MASK_CACHE[key]

    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - 0.0) ** 2 + (xx - 0.0) ** 2)
    rr = rr / (rr.max() + 1e-6)

    m_low = (rr <= r1)
    m_mid = ((rr > r1) & (rr <= r2))
    m_high = (rr > r2)

    _DCT_MASK_CACHE[key] = (m_low, m_mid, m_high)
    return m_low, m_mid, m_high


def extract_global_dct_bands(
    bgr01: np.ndarray,
    freq_in: str = "y",
    r1: float = 0.12,
    r2: float = 0.35,
) -> np.ndarray:
    """
    Return:
      freq_in='y'    : (H,W,3)
      freq_in='ycbcr': (H,W,9)
    """
    ycrcb = bgr_to_ycrcb01(bgr01)
    chans = [ycrcb[:, :, 0]] if freq_in == "y" else [ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]]

    outs = []
    for ch in chans:
        H, W = ch.shape

        C_H = _get_dct_mat(H, device="cpu")  # (H,H)
        C_W = _get_dct_mat(W, device="cpu")  # (W,W)

        x = torch.from_numpy(ch.astype(np.float32))  # (H,W)
        D = (C_H @ x) @ C_W.t()
        A = D.abs().numpy()

        m_low, m_mid, m_high = _get_dct_radial_masks(H, W, r1, r2)

        low = _normalize_map_fast(A * m_low)
        mid = _normalize_map_fast(A * m_mid)
        high = _normalize_map_fast(A * m_high)

        outs.append(np.stack([low, mid, high], axis=2))

    return np.concatenate(outs, axis=2).astype(np.float32)


# ───────────────────────────────────────────────────────────
# CSV 기록 유틸 (val mode에서만 사용)
# ───────────────────────────────────────────────────────────
def append_metrics_csv(csv_path: str, row_dict: dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.isfile(csv_path)

    fieldnames = [
        "epoch",
        "backbone",
        "stream",
        "freq_in",
        "dct_mode",
        "fft_mode",
        "band",
        "acc",
        "f1_macro",
        "f1_binary",
        "prec",
        "recall",
        "ckpt_path",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in fieldnames})


# ───────────────────────────────────────────────────────────
# 데이터셋 (FF++ mtcnn 폴더를 video별 frame 이미지로 스캔)
# labels:
#   0 = original_sequences (actors/youtube)
#   1 = manipulated_sequences (Deepfakes, FaceSwap, ...)
# ───────────────────────────────────────────────────────────
class FFPPFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        compression: str = "raw",
        stream: str = "rgb",        # rgb|dct|fft
        dct_mode: str = "block",    # block|global
        fft_mode: str = "global",   # global
        freq_in: str = "y",         # y|ycbcr
        band: str = "all",          # all|low|mid|high  (global DCT/FFT만 의미 있음)
        transform=None,
        cache_dir: Optional[str] = None,
        # fixed params
        dct_block: int = 8,
        band_r1: float = 0.12,
        band_r2: float = 0.35,
        block_energy_mode: str = "ac",  # "ac" | "total"
    ):
        print("Dataset 초기화 중…")
        self.stream = stream
        self.dct_mode = dct_mode
        self.fft_mode = fft_mode
        self.freq_in = freq_in
        self.band = band
        self.transform = transform
        self.samples = []

        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.dct_block = dct_block
        self.band_r1 = float(band_r1)
        self.band_r2 = float(band_r2)
        self.block_energy_mode = block_energy_mode

        # FF++ directory layout:
        # root_dir/original_sequences/<method>/raw/mtcnn/<video_id>/<frame>.png
        # root_dir/manipulated_sequences/<method>/raw/mtcnn/<video_id>/<frame>.png
        orig_base = os.path.join(root_dir, "original_sequences")
        manip_base = os.path.join(root_dir, "manipulated_sequences")

        def scan(base_dir: str, label: int):
            if not os.path.isdir(base_dir):
                print(f"[WARN] missing dir: {base_dir}")
                return
            for method in sorted(os.listdir(base_dir)):
                full_dir = os.path.join(base_dir, method, compression, "mtcnn")
                if not os.path.isdir(full_dir):
                    continue
                for subdir, _, files in os.walk(full_dir):
                    for fname in files:
                        if fname.lower().endswith(("png", "jpg", "jpeg")):
                            self.samples.append((os.path.join(subdir, fname), label))

        scan(orig_base, 0)
        scan(manip_base, 1)

        print(f"총 샘플 수: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, img_path: str) -> str:
        key = (
            f"{img_path}|{self.stream}|{self.freq_in}"
            f"|dct{self.dct_mode}|fft{self.fft_mode}"
            f"|band{self.band}|r{self.band_r1}_{self.band_r2}"
            f"|blk{self.dct_block}|blkE{self.block_energy_mode}"
        )
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npy")

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

        if self.transform:
            img = self.transform(img)

        arr_rgb = np.array(img).astype(np.float32) / 255.0
        arr_bgr = arr_rgb[:, :, ::-1].copy()

        if self.stream == "rgb":
            x = arr_rgb.transpose(2, 0, 1).astype(np.float32)
            return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)

        if self.cache_dir:
            cpath = self._cache_path(path)
            if os.path.isfile(cpath):
                feat = np.load(cpath)
            else:
                feat = self._compute_freq_feat(arr_bgr)
                tmp = cpath + f".tmp{os.getpid()}"
                try:
                    with open(tmp, "wb") as f:
                        np.save(f, feat, allow_pickle=False)
                    os.replace(tmp, cpath)
                finally:
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
        else:
            feat = self._compute_freq_feat(arr_bgr)

        band_enabled = (self.dct_mode == "global") or (self.stream == "fft")
        feat = _apply_band_select(feat, self.band, enabled=band_enabled)

        x = feat.transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)

    def _compute_freq_feat(self, arr_bgr: np.ndarray) -> np.ndarray:
        if self.stream == "dct":
            if self.dct_mode == "block":
                return extract_block_dct_energy(
                    arr_bgr,
                    block=self.dct_block,
                    freq_in=self.freq_in,
                    energy_mode=self.block_energy_mode,
                )
            elif self.dct_mode == "global":
                return extract_global_dct_bands(
                    arr_bgr,
                    freq_in=self.freq_in,
                    r1=self.band_r1,
                    r2=self.band_r2,
                )
            else:
                raise ValueError(f"Unknown dct_mode: {self.dct_mode}")

        elif self.stream == "fft":
            return extract_fft_bands(
                arr_bgr,
                freq_in=self.freq_in,
                r1=self.band_r1,
                r2=self.band_r2,
            )
        else:
            raise ValueError(f"Unknown stream: {self.stream}")


# ───────────────────────────────────────────────────────────
# 평가 메트릭
# ───────────────────────────────────────────────────────────
def compute_metrics(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            x = x.to(device, non_blocking=True)
            out = model(x)
            p = out.argmax(dim=1).detach().cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(y.numpy().tolist())

    if len(trues) == 0:
        return {"f1_binary": 0.0, "f1_macro": 0.0, "prec": 0.0, "recall": 0.0, "acc": 0.0}

    return {
        "f1_binary": f1_score(trues, preds, average="binary"),
        "f1_macro": f1_score(trues, preds, average="macro"),
        "prec": precision_score(trues, preds, average="macro", zero_division=0),
        "recall": recall_score(trues, preds, average="macro", zero_division=0),
        "acc": accuracy_score(trues, preds),
    }


# ───────────────────────────────────────────────────────────
# 메인
# ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--compression", type=str, default="raw")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help='CUDA_VISIBLE_DEVICES에 넣을 값. 예: "0", "1", "0,1"'
    )

    # reproducibility
    parser.add_argument("--seed", type=int, default=2026)

    # stream config
    parser.add_argument("--stream", choices=["rgb", "dct", "fft"], required=True)
    parser.add_argument("--freq-in", choices=["y", "ycbcr"], default="y")

    # dct/fft mode
    parser.add_argument("--dct-mode", choices=["block", "global"], default="block")
    parser.add_argument("--fft-mode", choices=["global"], default="global")

    # band selection (global DCT/FFT only)
    parser.add_argument(
        "--band",
        choices=["all", "low", "mid", "high"],
        default="all",
        help="(global DCT/FFT only) all=use low/mid/high; else keep only selected band and zero others",
    )

    # band split params (global DCT/FFT에서 사용)
    parser.add_argument("--r1", type=float, default=0.12, help="low band radius threshold (0~1)")
    parser.add_argument("--r2", type=float, default=0.35, help="mid band radius threshold (0~1)")

    # block DCT energy mode (NO band split)
    parser.add_argument(
        "--block-energy",
        choices=["ac", "total"],
        default="ac",
        help="block DCT output map: ac=sum(|DCT|)-|DC| (recommended), total=sum(|DCT|)",
    )

    # cache
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional .npy cache for DCT/FFT features")

    # backbone
    parser.add_argument("--backbone", choices=["convnext_tiny", "replknet_b"], default="convnext_tiny")
    parser.add_argument("--use-cbam", action="store_true")

    # train
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    # early stopping (train mode)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="train mode에서 val로 떼어낼 비율 (0이면 val/earlystop/best 저장 안 함)",
    )
    parser.add_argument(
        "--monitor",
        choices=["f1_binary", "acc", "f1_macro"],
        default="f1_binary",
        help="best/earlystop을 판단할 metric (val 기준)",
    )

    parser.add_argument("--mode", choices=["train", "val"], default="train")

    # ckpt / resume
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_freq")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, help="Checkpoint path for val mode (manual eval)")

    # logging (val mode)
    parser.add_argument(
        "--val-metrics-csv",
        type=str,
        default="./freq_metrics.csv",
        help="CSV file to append validation metrics (val mode only)",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n▶ CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    print(f"▶ Using device: {device}\n")

    # 입력 채널 수 결정
    if args.stream == "rgb":
        in_ch = 3
    elif args.stream == "dct" and args.dct_mode == "block":
        in_ch = 1 if args.freq_in == "y" else 3
    else:
        in_ch = 3 if args.freq_in == "y" else 9

    # Dataset
    tfm = transforms.Resize((224, 224))
    dataset = FFPPFrameDataset(
        root_dir=args.data_dir,
        compression=args.compression,
        stream=args.stream,
        dct_mode=args.dct_mode,
        fft_mode=args.fft_mode,
        freq_in=args.freq_in,
        band=args.band,
        transform=tfm,
        cache_dir=args.cache_dir,
        dct_block=8,
        band_r1=args.r1,
        band_r2=args.r2,
        block_energy_mode=args.block_energy,
    )

    total_len = len(dataset)

    # split (train mode에서도 val_ratio>0이면 val 사용)
    val_ratio = float(args.val_ratio) if args.mode == "train" else 0.2
    val_ratio = max(0.0, min(0.5, val_ratio))

    if args.mode == "train":
        if val_ratio > 0.0:
            val_size = int(val_ratio * total_len)
            train_size = total_len - val_size
            train_ds, val_ds = random_split(dataset, [train_size, val_size])
            print(f"Train/Val split (train mode): {train_size}/{val_size}")
        else:
            train_ds, val_ds = dataset, None
            print(f"Train only (train mode): {total_len}/{total_len}")
    else:
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        print(f"Train/Val split: {train_size}/{val_size}")

    def _collate(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        return default_collate(batch)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=_collate,
    )

    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, args.batch_size // 4),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=_collate,
        )
    else:
        val_loader = None

    # Model
    if args.backbone == "convnext_tiny":
        if _HAS_CONVNEXT_TINY:
            model = convnextv2_tiny(in_chans=in_ch, num_classes=2, use_cbam=args.use_cbam)
        else:
            print("⚠️ convnextv2_tiny() not found. Fallback to convnextv2_large().")
            model = convnextv2_large(in_chans=in_ch, num_classes=2, use_cbam=args.use_cbam)
    else:
        if _HAS_REPLK_B:
            model = create_RepLKNetB(num_classes=2, in_channels=in_ch, use_cbam=args.use_cbam)
        else:
            print("⚠️ create_RepLKNetB() not found. Fallback to create_RepLKNet31B().")
            model = create_RepLKNet31B(num_classes=2, in_channels=in_ch, use_cbam=args.use_cbam)

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # exp prefix (in_ch 포함 + block_energy 포함 + seed 포함)
    exp_prefix = (
        f"{args.backbone}_{args.stream}_{args.freq_in}"
        f"_dct{args.dct_mode}_fft{args.fft_mode}"
        f"_band{args.band}_r{args.r1}-{args.r2}"
        f"_blkE{args.block_energy}_inch{in_ch}"
        f"_seed{args.seed}"
    )
    os.makedirs(args.ckpt_dir, exist_ok=True)

    best_path = os.path.join(args.ckpt_dir, f"{exp_prefix}_best.pth")
    last_path = os.path.join(args.ckpt_dir, f"{exp_prefix}_last.pth")
    earlystop_path = os.path.join(args.ckpt_dir, f"{exp_prefix}_earlystop.pth")
    meta_path = os.path.join(args.ckpt_dir, f"{exp_prefix}_meta.json")

    # Resume: last에서만
    start_ep = 0
    if args.resume and os.path.isfile(last_path):
        print(f"▶ Resume from {last_path}")
        ckpt = torch.load(last_path, map_location=device)
        start_ep = int(ckpt.get("epoch", 0))
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])

    # ────────────────────────────────────────────────────
    # TRAIN MODE (disk-friendly: best/last/earlystop만 유지)
    # ────────────────────────────────────────────────────
    if args.mode == "train":
        best_score = -1e18
        best_epoch = 0
        bad_epochs = 0
        stopped_early = False
        last_epoch_ran = start_ep

        def get_score(metrics: dict) -> float:
            return float(metrics.get(args.monitor, 0.0))

        for epoch in range(start_ep + 1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                if batch is None:
                    continue
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                seen_batches += 1

            avg_loss = running_loss / max(1, seen_batches)
            last_epoch_ran = epoch
            print(f"[Train] Epoch {epoch}  Loss {avg_loss:.4f}")

            # 1) last: 매 epoch 덮어쓰기
            save_torch_atomic(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "train_loss": avg_loss,
                    "config": vars(args),
                    "in_ch": in_ch,
                },
                last_path,
            )

            # 2) val로 best/earlystop 판단 (val_loader 있을 때만)
            if val_loader is not None:
                metrics = compute_metrics(model, val_loader, device)
                score = get_score(metrics)
                print(
                    f"[Val] Epoch {epoch}  {args.monitor}={score:.4f}  "
                    f"acc={metrics['acc']:.4f}  f1b={metrics['f1_binary']:.4f}"
                )

                improved = score > best_score + 1e-9
                if improved:
                    best_score = score
                    best_epoch = epoch
                    bad_epochs = 0

                    save_torch_atomic(
                        {
                            "epoch": epoch,
                            "best_score": best_score,
                            "model": model.state_dict(),
                            "optim": optimizer.state_dict(),
                            "train_loss": avg_loss,
                            "val_metrics": metrics,
                            "config": vars(args),
                            "in_ch": in_ch,
                        },
                        best_path,
                    )
                else:
                    bad_epochs += 1

                # early stopping
                if args.patience > 0 and bad_epochs >= args.patience:
                    print(
                        f"⏹ EarlyStop triggered at epoch={epoch} "
                        f"(best_epoch={best_epoch}, best_{args.monitor}={best_score:.4f})"
                    )
                    stopped_early = True
                    break

        # 3) earlystop checkpoint 생성 (best가 있으면 best 복사, 없으면 last 복사)
        if os.path.isfile(best_path):
            copy_atomic(best_path, earlystop_path)
        else:
            copy_atomic(last_path, earlystop_path)

        # meta.json 저장
        meta = {
            "exp_prefix": exp_prefix,
            "best_epoch": best_epoch if os.path.isfile(best_path) else None,
            "best_score": best_score if os.path.isfile(best_path) else None,
            "monitor": args.monitor,
            "stopped_early": stopped_early,
            "last_epoch": last_epoch_ran,
            "paths": {"best": best_path, "last": last_path, "earlystop": earlystop_path},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print("\n✅ Final saved checkpoints:")
        if os.path.isfile(best_path):
            print(f"  - BEST     : {best_path}")
        else:
            print("  - BEST     : (not created; val_ratio=0 or no val)")
        print(f"  - LAST     : {last_path}")
        print(f"  - EARLYSTOP: {earlystop_path}")
        print(f"  - META     : {meta_path}")

    # ────────────────────────────────────────────────────
    # VAL MODE (manual evaluation of a given ckpt)
    # ────────────────────────────────────────────────────
    else:
        assert args.ckpt, "--ckpt 경로가 필요합니다."
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)

        assert val_loader is not None, "val_loader가 None입니다. (dataset split 문제)"
        metrics = compute_metrics(model, val_loader, device)
        print("\n=== Validation Metrics ===")
        for k, v in metrics.items():
            print(f"{k:>8}: {v:.4f}")

        append_metrics_csv(
            args.val_metrics_csv,
            {
                "epoch": int(state.get("epoch", -1)) if isinstance(state, dict) else -1,
                "backbone": args.backbone,
                "stream": args.stream,
                "freq_in": args.freq_in,
                "dct_mode": args.dct_mode,
                "fft_mode": args.fft_mode,
                "band": args.band,
                "acc": metrics["acc"],
                "f1_macro": metrics["f1_macro"],
                "f1_binary": metrics["f1_binary"],
                "prec": metrics["prec"],
                "recall": metrics["recall"],
                "ckpt_path": args.ckpt,
            },
        )


if __name__ == "__main__":
    main()