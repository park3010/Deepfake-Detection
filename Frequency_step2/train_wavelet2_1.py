# train_wavelet.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FF++ frame classification (DeepFake) training script
- Stream: rgb / wavelet / rgb+wavelet
- Wavelet: DWT or SWT, level 1~2, family {haar, sym4, db4, db8}
- Subband: ll / high / ll_energy
- Backbone (models/): ResNet50, ResNet50+CBAM, ConvNeXtV2-Tiny/Nano (+ optional CBAM)
- EarlyStopping, Resume
- ✅ Checkpoint policy (3 files only):
    1) best_{backbone}_{stream}.pth        (val F1 최고 갱신 시 덮어쓰기)
    2) earlystop_{backbone}_{stream}.pth  (EarlyStopping "발동 시점" 덮어쓰기)
    3) last_{backbone}_{stream}.pth        (매 epoch 마지막 가중치 덮어쓰기)
- ✅ Atomic save (tmp -> os.replace) 적용

Example (Wavelet-only, ConvNeXtV2-Tiny):
  python train_ffpp_wavelet_stream.py \
    --mode train --data-dir /path/to/ffpp --compression raw \
    --stream wavelet --backbone convnextv2_tiny --convnext-cbam \
    --wavelet-type swt --wavelet sym4 --wavelet-level 2 --subband ll_energy --wavelet-gray \
    --epochs 20 --batch-size 32 --lr 1e-4 --gpu 0
"""

import os
import argparse
import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, random_split, default_collate
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

import pywt  # pip install PyWavelets

# ✅ your custom backbones in models/
from models.convnextv2 import convnextv2_tiny
from models.resnet_cbam import resnet50

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 재현성 우선
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- Utils -------------------------
def _find_first_conv(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None


def adapt_first_conv_in_channels(model: torch.nn.Module, in_ch: int):
    """
    모델 내 첫 nn.Conv2d의 입력 채널 수가 in_ch와 다르면
    동일 하이퍼파라미터로 새 Conv를 만들고 가중치를 평균/반복 복사하여 교체.
    """
    first_conv = _find_first_conv(model)
    if first_conv is None:
        print("[adapt_first_conv] Conv2d를 찾지 못했어요(스킵).")
        return model

    if first_conv.in_channels == in_ch:
        return model

    with torch.no_grad():
        old_weight = first_conv.weight  # [out_c, in_c, k, k]
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
            mean_w = old_weight.mean(dim=1, keepdim=True)  # [out_c,1,k,k]
            rep = mean_w.repeat(1, in_ch, 1, 1)           # [out_c,in_ch,k,k]
            new_weight = rep.clone()
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
        print(f"[adapt_first_conv] 첫 Conv 입력 채널 {old_in_c} → {in_ch} 교체 완료.")
    else:
        print("[adapt_first_conv] 교체 실패(스킵).")
    return model


def atomic_torch_save(obj, path: Path):
    """
    안전한 저장: tmp에 저장 후 os.replace로 원자적 교체(덮어쓰기).
    저장 도중 실패해도 기존 파일이 깨질 확률을 줄임.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp_path))
    os.replace(str(tmp_path), str(path))


def _auto_find_latest_ckpt(ckpt_dir: Path, backbone: str, stream: str) -> Optional[Path]:
    """
    새 정책: last -> best -> (구버전 호환) epoch_*.pth
    """
    if not ckpt_dir.exists():
        return None

    last = ckpt_dir / f"last_{backbone}_{stream}.pth"
    if last.exists():
        return last

    best = ckpt_dir / f"best_{backbone}_{stream}.pth"
    if best.exists():
        return best

    cands = sorted(ckpt_dir.glob("epoch_*.pth"))
    return cands[-1] if cands else None


def _load_resume_ckpt(model, optimizer, ckpt_path: Path, device, strict: bool):
    print(f"▶ Resume from: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)

    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict:
        if missing:
            print(f"  - missing keys: {missing[:6]}{' ...' if len(missing)>6 else ''}")
        if unexpected:
            print(f"  - unexpected keys: {unexpected[:6]}{' ...' if len(unexpected)>6 else ''}")

    if optimizer is not None and "optim_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optim_state"])
        except Exception as e:
            print(f"  - optimizer state 로드 스킵 (사유: {e})")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_f1 = float(ckpt.get("best_f1", 0.0))
    return start_epoch, best_f1


# --------------------- EarlyStopping ---------------------
class EarlyStopping:
    """val_loss가 patience epoch간 개선되지 않으면 학습 조기 종료(판단만)"""

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


# --------------------- Dataset ---------------------
class FFPPFrameDataset(torch.utils.data.Dataset):
    """
    FF++ (mtcnn crop) frame dataset
    - stream: rgb / wavelet / rgb+wavelet
    - wavelet: dwt/swt, subband: ll/high/ll_energy
    """

    def __init__(
        self,
        root_dir: str,
        compression: str = "raw",
        transform=None,
        stream: str = "wavelet",  # rgb / wavelet / rgb+wavelet
        wavelet: str = "sym4",
        wavelet_level: int = 2,
        wavelet_type: str = "swt",  # swt / dwt
        wavelet_gray: bool = False,
        subband: str = "ll_energy",  # ll / high / ll_energy
        robust_norm: bool = True,
    ):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        self.stream = stream
        self.wavelet = wavelet
        self.wavelet_level = max(1, int(wavelet_level))
        self.wavelet_type = wavelet_type
        self.wavelet_gray = wavelet_gray
        self.subband = subband
        self.robust_norm = robust_norm

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
        if x.shape[0] == H and x.shape[1] == W:
            return x.astype(np.float32)
        return cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    def _wavelet_maps_2d(self, ch_2d: np.ndarray, H: int, W: int) -> List[np.ndarray]:
        """
        return list of feature maps (each HxW)
        subband:
          - ll: cA_last only
          - high: (cH,cV,cD) for each level
          - ll_energy: cA_last + energy(sqrt(cH^2+cV^2+cD^2)) per level
        type:
          - swt: same size maps
          - dwt: smaller maps -> resize back to HxW
        """
        lvl = self.wavelet_level
        maps: List[np.ndarray] = []

        if self.wavelet_type == "swt":
            coeffs = pywt.swt2(ch_2d, wavelet=self.wavelet, level=lvl, norm=True)
            cA_last = coeffs[-1][0]
            details = [c[1] for c in coeffs]  # list of (cH,cV,cD) per level (1..lvl)

            if self.subband == "ll":
                maps.append(self._resize_to(cA_last, H, W))
            elif self.subband == "high":
                for (cH, cV, cD) in details:
                    maps.extend(
                        [
                            self._resize_to(np.abs(cH), H, W),
                            self._resize_to(np.abs(cV), H, W),
                            self._resize_to(np.abs(cD), H, W),
                        ]
                    )
            elif self.subband == "ll_energy":
                maps.append(self._resize_to(cA_last, H, W))
                for (cH, cV, cD) in details:
                    energy = np.sqrt(
                        cH.astype(np.float32) ** 2
                        + cV.astype(np.float32) ** 2
                        + cD.astype(np.float32) ** 2
                    )
                    maps.append(self._resize_to(energy, H, W))
            else:
                raise ValueError("subband must be one of: ll, high, ll_energy")

        elif self.wavelet_type == "dwt":
            coeffs = pywt.wavedec2(ch_2d, wavelet=self.wavelet, level=lvl)
            cA_last = coeffs[0]
            details = list(reversed(coeffs[1:]))  # now level 1..L

            if self.subband == "ll":
                maps.append(self._resize_to(cA_last, H, W))
            elif self.subband == "high":
                for (cH, cV, cD) in details:
                    maps.extend(
                        [
                            self._resize_to(np.abs(cH), H, W),
                            self._resize_to(np.abs(cV), H, W),
                            self._resize_to(np.abs(cD), H, W),
                        ]
                    )
            elif self.subband == "ll_energy":
                maps.append(self._resize_to(cA_last, H, W))
                for (cH, cV, cD) in details:
                    energy = np.sqrt(
                        cH.astype(np.float32) ** 2
                        + cV.astype(np.float32) ** 2
                        + cD.astype(np.float32) ** 2
                    )
                    maps.append(self._resize_to(energy, H, W))
            else:
                raise ValueError("subband must be one of: ll, high, ll_energy")
        else:
            raise ValueError("wavelet_type must be 'swt' or 'dwt'")

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
            return np.stack(maps, axis=0)  # [Cw, H, W]
        else:
            b, g, r = cv2.split(arr_bgr.astype(np.float32))
            wb = self._wavelet_maps_2d(b, H, W)
            wg = self._wavelet_maps_2d(g, H, W)
            wr = self._wavelet_maps_2d(r, H, W)
            return np.stack(wb + wg + wr, axis=0)  # [Cw, H, W]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

        if self.transform:
            img = self.transform(img)

        arr_rgb = np.array(img).astype(np.float32)
        arr_bgr = arr_rgb[:, :, ::-1].copy()

        chans: List[np.ndarray] = []

        if self.stream in ("rgb", "rgb+wavelet"):
            base = (arr_rgb.transpose(2, 0, 1) / 255.0).astype(np.float32)  # [3,H,W]
            chans.append(base)

        if self.stream in ("wavelet", "rgb+wavelet"):
            w = self._wavelet_features(arr_bgr).astype(np.float32)
            chans.append(w)

        x = np.concatenate(chans, axis=0)  # [C,H,W]
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


# --------------------- Metrics ---------------------
def compute_metrics(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            out = model(x)
            p = out.argmax(1).cpu().tolist()
            preds.extend(p)
            trues.extend(y.tolist())
    return {
        "acc": accuracy_score(trues, preds),
        "f1": f1_score(trues, preds, average="macro"),
        "prec": precision_score(trues, preds, average="macro", zero_division=0),
        "recall": recall_score(trues, preds, average="macro", zero_division=0),
    }


# --------------------- Wavelet channel calculator ---------------------
def calc_wavelet_channels(gray: bool, subband: str, level: int) -> int:
    """
    subband:
      - ll: 1 map (cA_last)
      - high: 3 maps per level (cH,cV,cD) -> 3*L
      - ll_energy: 1 (LL) + 1 per level (energy) -> 1 + L
    """
    if subband == "ll":
        per_stream = 1
    elif subband == "high":
        per_stream = 3 * level
    elif subband == "ll_energy":
        per_stream = 1 + level
    else:
        raise ValueError("subband must be one of: ll, high, ll_energy")

    return per_stream if gray else per_stream * 3


# --------------------- Model factory ---------------------
def build_model(backbone: str, num_classes: int, pretrained: bool, in_ch: int, convnext_cbam: bool) -> nn.Module:
    # ResNet50 (plain)
    if backbone == "resnet50":
        model = resnet50(pretrained=pretrained, num_classes=num_classes)
        model = adapt_first_conv_in_channels(model, in_ch)
        return model

    # ConvNeXtV2 Tiny (your custom implementation supports in_chans/num_classes/use_cbam)
    if backbone == "convnextv2_tiny":
        model = convnextv2_tiny(in_chans=in_ch, num_classes=num_classes, use_cbam=convnext_cbam)
        if pretrained:
            print("[WARN] convnextv2_* 커스텀 구현에는 ImageNet pretrained 로더가 없습니다. (pretrained 무시)")
        return model

    raise ValueError("backbone must be one of: resnet50, convnextv2_tiny")


# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "convnextv2_tiny"],
        default="resnet50",
    )
    parser.add_argument("--pretrained", action="store_true", help="(ResNet만) ImageNet pretrained weights 사용")
    parser.add_argument("--convnext-cbam", action="store_true", help="(ConvNeXt만) use_cbam=True 적용")

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--compression", type=str, default="raw")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42, help="random seed for split/shuffle/init")

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--mode", choices=["train", "val"], default="train")
    parser.add_argument("--ckpt", type=str, help="val 모드에서 불러올 체크포인트(.pth) 경로")

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints")

    parser.add_argument("--resume", action="store_true", help="이전 체크포인트에서 이어서 학습")
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="이어 학습할 체크포인트(.pth) 경로 (미지정 시 last->best->epoch 최신 자동 선택)",
    )
    parser.add_argument("--resume-strict", action="store_true", help="state_dict 로드 strict=True")

    # stream
    parser.add_argument(
        "--stream",
        choices=["rgb", "wavelet", "rgb+wavelet"],
        default="rgb+wavelet",
        help="실험 스트림: rgb / wavelet / rgb+wavelet",
    )

    # wavelet
    parser.add_argument("--wavelet", type=str, default="sym4", choices=["haar", "sym4", "db4", "db8"])
    parser.add_argument("--wavelet-level", type=int, default=2, choices=[1, 2])
    parser.add_argument("--wavelet-type", choices=["dwt", "swt"], default="swt")
    parser.add_argument("--subband", choices=["ll", "high", "ll_energy"], default="ll_energy")
    parser.add_argument("--wavelet-gray", action="store_true", help="웨이블릿을 gray로 계산(채널 수 감소)")
    parser.add_argument("--no-robust-norm", action="store_true", help="robust percentile norm 비활성화")

    args = parser.parse_args()
    
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}\n")

    tfm = transforms.Resize((args.img_size, args.img_size))

    ds = FFPPFrameDataset(
        root_dir=args.data_dir,
        compression=args.compression,
        transform=tfm,
        stream=args.stream,
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

    tr_ld = DataLoader(
        tr_ds,
        args.batch_size,
        True,
        num_workers=4,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda b: default_collate([x for x in b if x]),
    )
    va_ld = DataLoader(
        va_ds,
        args.batch_size,
        False,
        num_workers=2,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda b: default_collate([x for x in b if x]),
    )

    # in_ch 계산
    in_ch = 0
    if args.stream in ("rgb", "rgb+wavelet"):
        in_ch += 3
    if args.stream in ("wavelet", "rgb+wavelet"):
        wch = calc_wavelet_channels(gray=args.wavelet_gray, subband=args.subband, level=int(args.wavelet_level))
        in_ch += wch
        per = wch if args.wavelet_gray else (wch // 3)
        print(
            f"▶ Wavelet cfg | type={args.wavelet_type} | wavelet={args.wavelet} | level={args.wavelet_level} | "
            f"subband={args.subband} | gray={args.wavelet_gray}\n"
            f"  - per={per} | Wch={wch} | 최종 in_ch={in_ch}\n"
        )
    else:
        print(f"▶ RGB-only | 최종 in_ch={in_ch}\n")

    model = build_model(
        backbone=args.backbone,
        num_classes=2,
        pretrained=args.pretrained,
        in_ch=in_ch,
        convnext_cbam=args.convnext_cbam,
    ).to(device)
    print(f"▶ Backbone: {args.backbone} | pretrained={args.pretrained} | stream={args.stream} | in_ch={in_ch}\n")

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 3-file checkpoint policy
    last_ckpt = ckpt_dir / f"last_{args.backbone}_{args.stream}.pth"
    best_ckpt = ckpt_dir / f"best_{args.backbone}_{args.stream}.pth"
    earlystop_ckpt = ckpt_dir / f"earlystop_{args.backbone}_{args.stream}.pth"

    early_stop = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True,
    )

    if args.mode == "train":
        crit = nn.CrossEntropyLoss()
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        best_f1 = 0.0

        # === Resume ===
        start_epoch = 1
        if args.resume:
            resume_ckpt_path = Path(args.resume_path) if args.resume_path else _auto_find_latest_ckpt(
                ckpt_dir, args.backbone, args.stream
            )
            if resume_ckpt_path is None:
                print("▶ resume 지정됐지만 사용할 체크포인트가 없습니다. 새로 시작합니다.")
            else:
                start_epoch, best_f1 = _load_resume_ckpt(
                    model=model,
                    optimizer=opt,
                    ckpt_path=resume_ckpt_path,
                    device=device,
                    strict=args.resume_strict,
                )
                print(f"  - start_epoch={start_epoch}, best_f1={best_f1:.4f}")

        # EarlyStopping은 val_loss로 판단: 여기서는 (1 - F1)을 loss처럼 사용
        early_stop.best_loss = 1.0 - best_f1 if best_f1 > 0 else np.inf

        for ep in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0

            for x, y in tqdm(tr_ld, desc=f"Epoch {ep}/{args.epochs}", leave=False):
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                running_loss += loss.item()

            tr_m = compute_metrics(model, tr_ld, device)
            va_m = compute_metrics(model, va_ld, device)
            avg_loss = running_loss / max(1, len(tr_ld))

            print(
                f"[{ep}] loss:{avg_loss:.4f} "
                f"Tr_acc:{tr_m['acc']:.4f} Tr_f1:{tr_m['f1']:.4f} | "
                f"Va_acc:{va_m['acc']:.4f} Va_f1:{va_m['f1']:.4f}"
            )

            # ✅ best ckpt (덮어쓰기)
            if va_m["f1"] > best_f1:
                best_f1 = va_m["f1"]
                atomic_torch_save(
                    {
                        "model_state": model.state_dict(),
                        "optim_state": opt.state_dict(),
                        "epoch": ep,
                        "best_f1": best_f1,
                        "args": vars(args),
                    },
                    best_ckpt,
                )
                print(" ▶ best ckpt 저장(덮어쓰기)")

            # ✅ last ckpt (매 epoch 끝나면 항상 덮어쓰기)
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "optim_state": opt.state_dict(),
                    "epoch": ep,
                    "best_f1": best_f1,
                    "args": vars(args),
                },
                last_ckpt,
            )

            # ✅ EarlyStopping 체크: (1 - val_f1)
            early_stop(1 - va_m["f1"])
            if early_stop.early_stop:
                # ✅ earlystop ckpt: "발동한 그 시점" 저장(덮어쓰기)
                atomic_torch_save(
                    {
                        "model_state": model.state_dict(),
                        "optim_state": opt.state_dict(),
                        "epoch": ep,
                        "best_f1": best_f1,
                        "args": vars(args),
                    },
                    earlystop_ckpt,
                )
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


if __name__ == "__main__":
    main()
