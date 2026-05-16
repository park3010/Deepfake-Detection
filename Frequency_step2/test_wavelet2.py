# test_wavelet.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wavelet(FF++ script options-aligned) evaluation script
- Stream: rgb / wavelet / rgb+wavelet
- Wavelet: DWT or SWT, level 1~2, family {haar, sym4, db4, db8}
- Subband: ll / high / ll_energy
- Backbone: resnet50 / convnextv2_tiny (학습 코드와 동일 계열)
- Video-level eval: frame softmax(prob_fake) mean -> threshold

Usage example:
  python eval_wavelet_stream.py \
    --gpu 0 --backbone convnextv2_tiny --checkpoint /path/to/best.pth \
    --img-size 224 \
    --stream wavelet \
    --wavelet-type swt --wavelet sym4 --wavelet-level 2 --subband ll_energy --wavelet-gray \
    --batch-size 32 --threshold 0.5 --csv ./result
"""

import os, gc
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.cuda.amp import autocast
import pywt


# -------------------------
# (학습 코드와 동일한) first conv 채널 적응 유틸
# -------------------------
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
        print(f"[adapt_first_conv] 첫 Conv 입력 채널 {first_conv.in_channels} → {in_ch} 교체 완료.")
    else:
        print("[adapt_first_conv] 교체 실패(스킵).")
    return model


# -------------------------
# Wavelet channel calculator (학습 코드와 동일 규칙)
# -------------------------
def calc_wavelet_channels(gray: bool, subband: str, level: int) -> int:
    """
    subband:
      - ll: 1 map (cA_last)
      - high: 3 maps per level -> 3*L
      - ll_energy: 1 (LL) + 1 per level (energy) -> 1 + L
    """
    level = max(1, int(level))
    if subband == "ll":
        per_stream = 1
    elif subband == "high":
        per_stream = 3 * level
    elif subband == "ll_energy":
        per_stream = 1 + level
    else:
        raise ValueError("subband must be one of: ll, high, ll_energy")
    return per_stream if gray else per_stream * 3


# -------------------------
# Wavelet feature utils (학습 코드와 동일 로직)
# -------------------------
def _robust_norm01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    denom = max(p99 - p1, eps)
    y = (x - p1) / denom
    return np.clip(y, 0.0, 1.0)

def _resize_to(x: np.ndarray, H: int, W: int) -> np.ndarray:
    if x.shape[0] == H and x.shape[1] == W:
        return x.astype(np.float32)
    return cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

def wavelet_maps_2d(
    ch_2d: np.ndarray,
    H: int, W: int,
    wavelet: str,
    level: int,
    wavelet_type: str,   # "swt" or "dwt"
    subband: str,        # "ll" / "high" / "ll_energy"
    robust_norm: bool,
) -> List[np.ndarray]:
    lvl = max(1, int(level))
    maps: List[np.ndarray] = []

    if wavelet_type == "swt":
        coeffs = pywt.swt2(ch_2d, wavelet=wavelet, level=lvl, norm=True)
        cA_last = coeffs[-1][0]
        details = [c[1] for c in coeffs]  # (cH,cV,cD) per level

        if subband == "ll":
            maps.append(_resize_to(cA_last, H, W))
        elif subband == "high":
            for (cH, cV, cD) in details:
                maps.extend([
                    _resize_to(np.abs(cH), H, W),
                    _resize_to(np.abs(cV), H, W),
                    _resize_to(np.abs(cD), H, W),
                ])
        elif subband == "ll_energy":
            maps.append(_resize_to(cA_last, H, W))
            for (cH, cV, cD) in details:
                energy = np.sqrt(cH.astype(np.float32)**2 + cV.astype(np.float32)**2 + cD.astype(np.float32)**2)
                maps.append(_resize_to(energy, H, W))
        else:
            raise ValueError("subband must be one of: ll, high, ll_energy")

    elif wavelet_type == "dwt":
        coeffs = pywt.wavedec2(ch_2d, wavelet=wavelet, level=lvl)
        cA_last = coeffs[0]
        details = list(reversed(coeffs[1:]))  # level 1..L

        if subband == "ll":
            maps.append(_resize_to(cA_last, H, W))
        elif subband == "high":
            for (cH, cV, cD) in details:
                maps.extend([
                    _resize_to(np.abs(cH), H, W),
                    _resize_to(np.abs(cV), H, W),
                    _resize_to(np.abs(cD), H, W),
                ])
        elif subband == "ll_energy":
            maps.append(_resize_to(cA_last, H, W))
            for (cH, cV, cD) in details:
                energy = np.sqrt(cH.astype(np.float32)**2 + cV.astype(np.float32)**2 + cD.astype(np.float32)**2)
                maps.append(_resize_to(energy, H, W))
        else:
            raise ValueError("subband must be one of: ll, high, ll_energy")
    else:
        raise ValueError("wavelet_type must be 'swt' or 'dwt'")

    if robust_norm:
        maps = [_robust_norm01(m) for m in maps]
    else:
        maps = [np.clip(m, 0.0, 1.0) for m in maps]
    return maps

def wavelet_features(
    arr_bgr: np.ndarray,
    wavelet: str,
    level: int,
    wavelet_type: str,
    wavelet_gray: bool,
    subband: str,
    robust_norm: bool,
) -> np.ndarray:
    H, W = arr_bgr.shape[:2]

    if wavelet_gray:
        gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        maps = wavelet_maps_2d(gray, H, W, wavelet, level, wavelet_type, subband, robust_norm)
        return np.stack(maps, axis=0)  # [Cw,H,W]
    else:
        b, g, r = cv2.split(arr_bgr.astype(np.float32))
        wb = wavelet_maps_2d(b, H, W, wavelet, level, wavelet_type, subband, robust_norm)
        wg = wavelet_maps_2d(g, H, W, wavelet, level, wavelet_type, subband, robust_norm)
        wr = wavelet_maps_2d(r, H, W, wavelet, level, wavelet_type, subband, robust_norm)
        return np.stack(wb + wg + wr, axis=0)  # [Cw,H,W]


# -------------------------
# VideoFrameDataset (학습 코드의 FFPPFrameDataset 입력 규칙과 동일)
# -------------------------
class VideoFrameDataset(Dataset):
    def __init__(
        self,
        frame_paths: List[str],
        img_size: int,
        stream: str,
        wavelet: str,
        wavelet_level: int,
        wavelet_type: str,
        wavelet_gray: bool,
        subband: str,
        robust_norm: bool,
    ):
        self.frames = frame_paths
        self.stream = stream
        self.wavelet = wavelet
        self.wavelet_level = max(1, int(wavelet_level))
        self.wavelet_type = wavelet_type
        self.wavelet_gray = wavelet_gray
        self.subband = subband
        self.robust_norm = robust_norm

        self.transform = transforms.Resize((img_size, img_size))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path = self.frames[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

        img = self.transform(img)
        arr_rgb = np.array(img).astype(np.float32)        # [H,W,3] RGB (0..255)
        arr_bgr = arr_rgb[:, :, ::-1].copy()              # [H,W,3] BGR

        chans = []

        # rgb or rgb+wavelet
        if self.stream in ("rgb", "rgb+wavelet"):
            base = (arr_rgb.transpose(2, 0, 1) / 255.0).astype(np.float32)  # [3,H,W], 0..1
            chans.append(base)

        # wavelet or rgb+wavelet
        if self.stream in ("wavelet", "rgb+wavelet"):
            w = wavelet_features(
                arr_bgr=arr_bgr,
                wavelet=self.wavelet,
                level=self.wavelet_level,
                wavelet_type=self.wavelet_type,
                wavelet_gray=self.wavelet_gray,
                subband=self.subband,
                robust_norm=self.robust_norm,
            ).astype(np.float32)
            chans.append(w)

        x = np.concatenate(chans, axis=0)  # [C,H,W]
        return torch.from_numpy(x)


# -------------------------
# TEST_DATASETS 정의 (기존 그대로 사용)
# -------------------------
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

# -------------------------
# evaluate_dataset: video-level (frame prob mean) 평가
# -------------------------
def evaluate_dataset(
    model,
    device,
    roots: List[str],
    label_value: int,
    img_size: int,
    batch_size: int,
    threshold: float,
    stream: str,
    wavelet: str,
    wavelet_level: int,
    wavelet_type: str,
    wavelet_gray: bool,
    subband: str,
    robust_norm: bool,
):
    y_true, y_pred = [], []
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

            ds = VideoFrameDataset(
                frame_paths=frames,
                img_size=img_size,
                stream=stream,
                wavelet=wavelet,
                wavelet_level=wavelet_level,
                wavelet_type=wavelet_type,
                wavelet_gray=wavelet_gray,
                subband=subband,
                robust_norm=robust_norm,
            )

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=lambda b: [x for x in b if x is not None],  # None 제거
            )

            sum_p, cnt = 0.0, 0

            for batch_list in tqdm(loader, desc=f" frames of {vid}", leave=False):
                if len(batch_list) == 0:
                    continue
                batch = torch.stack(batch_list, dim=0).to(device)  # [B,C,H,W]

                with torch.inference_mode():
                    with autocast(enabled=use_amp):
                        logits = model(batch)
                        p = torch.softmax(logits, dim=1)[:, 1]  # fake prob, (B,)

                sum_p += float(p.sum().item())
                cnt += int(p.numel())

                del batch, logits, p

            if cnt == 0:
                continue

            avg_p = sum_p / cnt
            pred = 1 if avg_p >= threshold else 0

            y_true.append(label_value)
            y_pred.append(pred)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    return y_true, y_pred


# -------------------------
# Model builder (학습 코드 계열)
# -------------------------
def build_model(backbone: str, in_ch: int, num_classes: int, pretrained_resnet: bool, convnext_cbam: bool):
    if backbone == "resnet50":
        from models.resnet_cbam import resnet50  # 너 학습 코드와 동일 import
        model = resnet50(pretrained=pretrained_resnet, num_classes=num_classes)
        model = adapt_first_conv_in_channels(model, in_ch)
        return model

    if backbone == "convnextv2_tiny":
        from models.convnextv2 import convnextv2_tiny
        model = convnextv2_tiny(in_chans=in_ch, num_classes=num_classes, use_cbam=convnext_cbam)
        if pretrained_resnet:
            print("[WARN] convnextv2_* 커스텀 구현에는 ImageNet pretrained 로더가 없습니다(pretrained 무시).")
        return model

    raise ValueError("backbone must be one of: resnet50, convnextv2_tiny")


def main():
    parser = argparse.ArgumentParser()

    # device / model
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backbone", choices=["resnet50", "convnextv2_tiny"], required=True)
    parser.add_argument("--checkpoint", required=True, help="학습된 .pth (epoch_*.pth/best_*.pth/es_*.pth 등)")
    parser.add_argument("--pretrained", action="store_true", help="(ResNet만) ImageNet pretrained weights 사용")
    parser.add_argument("--convnext-cbam", action="store_true", help="(ConvNeXt만) use_cbam=True 적용")
    parser.add_argument("--strict", action="store_true", help="checkpoint load strict=True")

    # input / wavelet (학습 옵션과 동일)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--stream", choices=["rgb", "wavelet", "rgb+wavelet"], default="wavelet")
    parser.add_argument("--wavelet", type=str, default="sym4", choices=["haar", "sym4", "db4", "db8"])
    parser.add_argument("--wavelet-level", type=int, default=2, choices=[1, 2])
    parser.add_argument("--wavelet-type", choices=["dwt", "swt"], default="swt")
    parser.add_argument("--subband", choices=["ll", "high", "ll_energy"], default="ll_energy")
    parser.add_argument("--wavelet-gray", action="store_true")
    parser.add_argument("--no-robust-norm", action="store_true")

    # eval
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)

    # output
    parser.add_argument("--csv", type=str, default="./result", help="결과 CSV 저장 디렉터리")
    args = parser.parse_args()

    os.makedirs(args.csv, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")
    torch.cuda.empty_cache()

    robust_norm = (not args.no_robust_norm)

    # in_ch 계산 (학습 코드 규칙과 동일)
    in_ch = 0
    if args.stream in ("rgb", "rgb+wavelet"):
        in_ch += 3
    if args.stream in ("wavelet", "rgb+wavelet"):
        wch = calc_wavelet_channels(gray=args.wavelet_gray, subband=args.subband, level=int(args.wavelet_level))
        in_ch += wch
        per = wch if args.wavelet_gray else (wch // 3)
        print(
            f"▶ Wavelet cfg | type={args.wavelet_type} | wavelet={args.wavelet} | level={args.wavelet_level} | "
            f"subband={args.subband} | gray={args.wavelet_gray} | robust_norm={robust_norm}\n"
            f"  - per={per} | Wch={wch} | 최종 in_ch={in_ch}\n"
        )
    else:
        print(f"▶ RGB-only | 최종 in_ch={in_ch}\n")

    # 모델 생성
    model = build_model(
        backbone=args.backbone,
        in_ch=in_ch,
        num_classes=2,
        pretrained_resnet=args.pretrained,
        convnext_cbam=args.convnext_cbam,
    ).to(device).eval()

    # ckpt 로드
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=args.strict)
    if not args.strict:
        if missing:
            print(f"[load] missing keys: {missing[:8]}{' ...' if len(missing)>8 else ''}")
        if unexpected:
            print(f"[load] unexpected keys: {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

    print(f"▶ Loaded checkpoint: {args.checkpoint}")
    print(f"▶ Backbone={args.backbone} | stream={args.stream} | in_ch={in_ch}")
    torch.cuda.empty_cache()

    results, all_true, all_pred = [], [], []

    # 데이터셋별 root 정리 (기존 테스트 코드 흐름 유지)
    for ds_name, cfg in TEST_DATASETS.items():
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
            ds_paths = {"real": real_roots, "fake": fake_roots}

        elif ds_name == "DeepfakeTIMIT":
            fake_roots = []
            for quality_root in cfg["fake"]:
                if not os.path.isdir(quality_root):
                    continue
                for speaker in os.listdir(quality_root):
                    sp_path = os.path.join(quality_root, speaker)
                    if os.path.isdir(sp_path):
                        fake_roots.append(sp_path)
            ds_paths = {"real": [], "fake": fake_roots}

        else:
            ds_paths = cfg

        print(f"\n>>> Evaluating {ds_name}")

        rt, rp = evaluate_dataset(
            model=model,
            device=device,
            roots=ds_paths.get("real", []),
            label_value=0,
            img_size=args.img_size,
            batch_size=args.batch_size,
            threshold=args.threshold,
            stream=args.stream,
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
            wavelet_type=args.wavelet_type,
            wavelet_gray=args.wavelet_gray,
            subband=args.subband,
            robust_norm=robust_norm,
        )

        ft, fp = evaluate_dataset(
            model=model,
            device=device,
            roots=ds_paths.get("fake", []),
            label_value=1,
            img_size=args.img_size,
            batch_size=args.batch_size,
            threshold=args.threshold,
            stream=args.stream,
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
            wavelet_type=args.wavelet_type,
            wavelet_gray=args.wavelet_gray,
            subband=args.subband,
            robust_norm=robust_norm,
        )

        y_t, y_p = rt + ft, rp + fp

        if len(y_t) == 0:
            print(f"[{ds_name}] (skip) 유효 샘플 없음")
            continue

        acc = accuracy_score(y_t, y_p)
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1_macro = f1_score(y_t, y_p, average="macro", zero_division=0)
        f1_bin = f1_score(y_t, y_p, average="binary", zero_division=0)

        print(f"[{ds_name}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1-macro={f1_macro:.4f}  F1-binary={f1_bin:.4f}")

        results.append({
            "dataset": ds_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_macro": f1_macro,
            "f1_binary": f1_bin
        })

        all_true.extend(y_t)
        all_pred.extend(y_p)

    # Overall
    if len(all_true) > 0:
        oa = accuracy_score(all_true, all_pred)
        op = precision_score(all_true, all_pred, zero_division=0)
        or_ = recall_score(all_true, all_pred, zero_division=0)
        of1_m = f1_score(all_true, all_pred, average="macro", zero_division=0)
        of1_b = f1_score(all_true, all_pred, average="binary", zero_division=0)

        print("\n=== Overall Metrics ===")
        print(f"Acc   = {oa:.4f}  Prec={op:.4f}  Rec={or_:.4f}  F1-Macro={of1_m:.4f}  F1-Binary={of1_b:.4f}")

        results.append({
            "dataset": "Overall",
            "accuracy": oa,
            "precision": op,
            "recall": or_,
            "f1_macro": of1_m,
            "f1_binary": of1_b
        })

    # CSV 저장
    tag = (
        f"{args.backbone}"
        f"__{args.stream}"
        f"__{args.wavelet_type}-{args.wavelet}-L{args.wavelet_level}-{args.subband}"
        f"{'__gray' if args.wavelet_gray else ''}"
        f"{'__nonrobust' if args.no_robust_norm else ''}"
    )
    csv_path = os.path.join(args.csv, f"{tag}_results.csv")
    pd.DataFrame(results, columns=["dataset", "accuracy", "precision", "recall", "f1_macro", "f1_binary"]).to_csv(csv_path, index=False)
    print(f"\n▶ Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
