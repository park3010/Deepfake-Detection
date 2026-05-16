#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion(3-stream: RGB + DCT(RepLK) + Wavelet(ConvNeXt)) 모델 평가 스크립트 (v2)
- 학습 스크립트(trip_schreame_mlp_v2.py)와 동일한 구성으로 재현
- 입력: 각 테스트 데이터셋의 real/fake 디렉토리(비디오 단위 폴더 → 프레임 이미지)
- 출력: 데이터셋별 및 전체 통합 Acc/Prec/Rec/F1, CSV 저장
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------
# sys.path 설정 (안전)
# -----------------------
BASE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE, "Frequency_step2"))
sys.path.append(os.path.join(BASE, "RGBsparial_step1"))
sys.path.append(os.path.join(BASE, "MLP_step3"))

# RepLKNet 로컬 구현 위치 후보 (학습 스크립트도 동일 경로를 추가함)
_FREQ_MODELS = os.path.join(BASE, "Frequency_step2", "models")
_ALT_MODELS = os.path.join(BASE, "models")
for _p in (_FREQ_MODELS, _ALT_MODELS):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ====== 학습 코드의 구성요소 가져오기 ======
from MLP_step3.trip_schreame_mlp_v2 import (
    ESFCM, SE, CBAM,
    FreqConvNeXtWavEncoder, DCTRepLKEncoder,  # 브랜치 구현
    MultiBranchFusion,
    maybe_inject_rgb_attention,
    load_branch_ckpt_any
)

# =========================
# 테스트 데이터셋 정의
# =========================
TEST_DATASETS: Dict[str, Dict] = {
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


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# =========================
# Dataset: 단일 이미지 → RGB 텐서 (학습과 동일 전처리)
# =========================
class PairImageDataset(Dataset):
    def __init__(self, frame_paths: List[str],
                 rgb_tfm: transforms.Compose):
        self.frames = frame_paths
        self.rgb_tfm = rgb_tfm

    def __len__(self): return len(self.frames)

    def __getitem__(self, idx):
        p = self.frames[idx]
        img = Image.open(p).convert("RGB")
        rgb  = self.rgb_tfm(img)   # [-1,1] 스케일(학습과 동일: mean=0.5,std=0.5)
        return rgb

def collect_video_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

def collect_frame_paths(vid_dir: str) -> List[str]:
    frames = []
    for e in IMG_EXTS:
        frames += sorted(glob.glob(os.path.join(vid_dir, f"*{e}")))
    return frames

# =========================
# 평가 루틴 (비디오 단위 평균 확률)
# =========================
@torch.no_grad()
def evaluate_roots_fusion(model: nn.Module, device: torch.device,
                          roots: List[str], label_value: int,
                          rgb_tfm: transforms.Compose,
                          batch_size: int = 8, threshold: float = 0.5) -> Tuple[List[int], List[int]]:
    y_true, y_pred = [], []
    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] 경로 없음: {root}")
            continue

        vids = sorted(os.listdir(root))
        for vid in tqdm(vids, desc=f"[{label_value}] {os.path.basename(root)}"):
            vid_dir = os.path.join(root, vid)
            if not os.path.isdir(vid_dir):
                continue
            frames = collect_frame_paths(vid_dir)
            if not frames:
                continue

            ds = PairImageDataset(frames, rgb_tfm)
            ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

            probs = []
            for rgb in tqdm(ld, desc=f" frames of {vid}", leave=False):
                rgb  = rgb.to(device, non_blocking=True)
                with autocast():
                    # 학습 forward 시그니처: forward(self, rgb_img, _clip_img_unused=None)
                    logits, _ = model(rgb, None)          # (B,2)
                p = torch.softmax(logits, dim=1)[:, 1]     # fake 확률
                probs.append(p.detach().cpu().numpy())

            avg_p = float(np.concatenate(probs).mean())
            pred  = 1 if avg_p >= threshold else 0

            y_true.append(label_value)
            y_pred.append(pred)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return y_true, y_pred

# =========================
# 메인
# =========================
def main():
    ap = argparse.ArgumentParser()

    # 디바이스/출력
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--csv', type=str, default="/home/sujin/psj2003/deepfake/code/result")
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--threshold', type=float, default=0.5)

    # ===== 통합 ckpt(선택) & 브랜치별 ckpt(필수 권장) =====
    ap.add_argument("--ckpt", type=str, default="", help="훈련 시 저장된 통합 ckpt(best_*.pth 등). 없으면 생략 가능")
    ap.add_argument("--ckpt-rgb", type=str, default="", help="RGB 브랜치 ckpt")
    ap.add_argument("--ckpt-wav", type=str, default="", help="Wavelet(ConvNeXt) 브랜치 ckpt")
    ap.add_argument("--ckpt-dct", type=str, default="", help="DCT(RepLKNet) 브랜치 ckpt")

    # RGB 옵션(평가 시 재현)
    ap.add_argument("--use-rgb", action="store_true", default=True)
    ap.add_argument("--use-cbam", action="store_true", default=False)
    ap.add_argument("--cbam-reduction", type=int, default=14)
    ap.add_argument("--cbam-mode", choices=["x_plus_scale","x_mul_scale","x_plus_xmulscale"], default="x_plus_xmulscale")
    ap.add_argument("--cbam-kernel", type=int, default=7)

    # Wavelet 옵션(평가 시 재현)
    ap.add_argument("--wav-backbone", type=str, default="convnextv2_large")
    ap.add_argument("--wavelet", type=str, default="sym4")
    ap.add_argument("--wavelet-level", type=int, default=1)
    ap.add_argument("--wavelet-gray", action="store_true")
    ap.add_argument("--wavelet-details", choices=["separate", "energy"], default="energy")
    ap.add_argument("--wavelet-include-approx", action="store_true", default=True)
    ap.add_argument("--wavelet-backend", choices=["swt","dwt"], default="swt")

    # DCT(RepLK) 옵션(평가 시 재현)
    ap.add_argument("--dct-in-chans", type=int, default=1, choices=[1,4])
    ap.add_argument("--dct-input-mode", choices=["dct","rgb_dct"], default="dct")
    ap.add_argument("--dct-use-cbam", action="store_true", default=False)

    args = ap.parse_args()

    # 디바이스
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    # 전처리(학습과 동일)
    rgb_tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # ===== 모델 구성: 학습 옵션 그대로 재현 =====
    model = MultiBranchFusion(
        rgb_backbone="tf_efficientnet_b7",
        use_rgb=args.use_rgb,                 # 학습 때 --use-rgb
        use_dct_replk=True,                   # 학습 때 --use-dct-replk
        use_wav_convnext=True,                # 학습 때 --use-wav-convnext
        dim=256,
        num_classes=2,
        heads=4,
        dct_in_chans=args.dct_in_chans,
        dct_input_mode=args.dct_input_mode,
        dct_use_cbam=args.dct_use_cbam,
        wav_backbone=args.wav_backbone,
        wavelet_cfg=dict(
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
            wavelet_gray=args.wavelet_gray,
            wavelet_details=args.wavelet_details,
            wavelet_include_approx=args.wavelet_include_approx,
            wavelet_backend=args.wavelet_backend
        ),
        use_branch_adapter=True,              # 학습 때 --use-branch-adapter
        adapter_dropout=0.0
    ).to(device)

    # ===== RGB Attention Hook (학습 시 --use-cbam을 켰다면 동일 주입) =====
    class _ArgShim:
        use_esfcm = False
        use_se = False
        def __init__(self, use_cbam, r, mode, k):
            self.use_cbam = use_cbam
            self.cbam_reduction = r
            self.cbam_mode = mode
            self.cbam_kernel = k

    if args.use_rgb and getattr(model, "rgb_timm", None) is not None:
        shim = _ArgShim(args.use_cbam, args.cbam_reduction, args.cbam_mode, args.cbam_kernel)
        maybe_inject_rgb_attention(model.rgb_timm, device, shim)

    # ===== 체크포인트 로드 =====
    # 통합 ckpt(있으면 먼저 로드)
    if args.ckpt and Path(args.ckpt).exists():
        st = torch.load(args.ckpt, map_location=device)
        sd = st.get("model", st.get("state_dict", st))
        model.load_state_dict(sd, strict=False)
        print(f"[LOAD] unified ckpt <- {args.ckpt}")

    # 브랜치별 ckpt (통합 ckpt 이후에 덮어쓰기)
    if args.ckpt_rgb and args.use_rgb:
        load_branch_ckpt_any(
            module=model.rgb,
            ckpt_path=args.ckpt_rgb,
            device=device,
            prefixes=["rgb.", "model.rgb.", "net.", "module.", ""],
            strict=False,
            tag="rgb"
        )
    if args.ckpt_wav:
        load_branch_ckpt_any(
            module=model.wav,
            ckpt_path=args.ckpt_wav,
            device=device,
            prefixes=["wav.", "model.wav.", "net.", "module.", ""],
            strict=False,
            tag="wav"
        )
    if args.ckpt_dct:
        load_branch_ckpt_any(
            module=model.dct,
            ckpt_path=args.ckpt_dct,
            device=device,
            prefixes=["dct.", "model.dct.", "net.", "module.", ""],
            strict=False,
            tag="dct"
        )

    model.eval()
    torch.cuda.empty_cache()

    # ===== WildDeepfake의 real/fake 수집(폴더 구조 특성상) =====
    datasets = {}
    for ds_name, cfg in TEST_DATASETS.items():
        if ds_name == "WildDeepfake":
            real_roots, fake_roots = [], []
            for split in cfg['splits']:
                sd = os.path.join(cfg['root'], split)
                if not os.path.isdir(sd):
                    continue
                for m in os.listdir(sd):
                    base = os.path.join(sd, m)
                    r, f = os.path.join(base, "real"), os.path.join(base, "fake")
                    if os.path.isdir(r): real_roots.append(r)
                    if os.path.isdir(f): fake_roots.append(f)
            datasets[ds_name] = {"real": real_roots, "fake": fake_roots}
        elif ds_name == "DeepfakeTIMIT":
            # speaker 단위 하위 폴더를 fake 루트로 간주
            fake_roots = []
            for quality_root in cfg['fake']:
                if not os.path.isdir(quality_root):
                    continue
                for speaker in os.listdir(quality_root):
                    sp_path = os.path.join(quality_root, speaker)
                    if os.path.isdir(sp_path):
                        fake_roots.append(sp_path)
            datasets[ds_name] = {"real": [], "fake": fake_roots}
        else:
            datasets[ds_name] = cfg

    os.makedirs(args.csv, exist_ok=True)
    results, all_true, all_pred = [], [], []

    for ds_name, paths in datasets.items():
        print(f"\n>>> Evaluating {ds_name}")
        rt, rp = evaluate_roots_fusion(
            model, device, paths.get("real", []), 0,
            rgb_tfm, batch_size=args.batch_size, threshold=args.threshold
        )
        ft, fp = evaluate_roots_fusion(
            model, device, paths.get("fake", []), 1,
            rgb_tfm, batch_size=args.batch_size, threshold=args.threshold
        )

        y_t, y_p = rt + ft, rp + fp
        if not y_t:
            print(f"[WARN] {ds_name}: 평가할 샘플이 없습니다.")
            continue

        acc   = accuracy_score(y_t, y_p)
        prec  = precision_score(y_t, y_p, zero_division=0)
        rec   = recall_score(y_t, y_p, zero_division=0)
        f1_m  = f1_score(y_t, y_p, average='macro',  zero_division=0)
        f1_b  = f1_score(y_t, y_p, average='binary', zero_division=0)
        print(f"[{ds_name}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1-macro={f1_m:.4f}  F1-binary={f1_b:.4f}")

        results.append({
            "dataset": ds_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_macro": f1_m,
            "f1_binary": f1_b
        })
        all_true.extend(y_t)
        all_pred.extend(y_p)

    # 전체 통합
    if all_true:
        oa   = accuracy_score(all_true, all_pred)
        op   = precision_score(all_true, all_pred, zero_division=0)
        or_  = recall_score(all_true, all_pred, zero_division=0)
        of1m = f1_score(all_true, all_pred, average='macro',  zero_division=0)
        of1b = f1_score(all_true, all_pred, average='binary', zero_division=0)
        print("\n=== Overall Metrics ===")
        print(f"Acc={oa:.4f}  Prec={op:.4f}  Rec={or_:.4f}  F1-Macro={of1m:.4f}  F1-Binary={of1b:.4f}")
        results.append({
            "dataset": "Overall",
            "accuracy": oa,
            "precision": op,
            "recall": or_,
            "f1_macro": of1m,
            "f1_binary": of1b
        })

    # 저장
    out_csv = os.path.join(args.csv, "fusion_stream3_v2_results.csv")
    pd.DataFrame(results, columns=["dataset","accuracy","precision","recall","f1_macro","f1_binary"]).to_csv(out_csv, index=False)
    print(f"\n▶ Saved metrics to {out_csv}")

if __name__ == "__main__":
    main()
