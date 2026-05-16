#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image, UnidentifiedImageError
import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------
DATASETS = {
    'original':                     'original_sequences/youtube',
    'DeepFakeDetection_original':   'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face':      'manipulated_sequences/Face2Face',
    'FaceShifter':    'manipulated_sequences/FaceShifter',
    'FaceSwap':       'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures',
}

EXCLUDE_DATASETS = {}
# ---------------------------------------------------------------------

class FFPP_RGB(Dataset):
    """
    FF++ (mtcnn crop) 프레임 데이터셋
    root_dir/
      original_sequences/<method>/<compression>/mtcnn/**/*.jpg|png
      manipulated_sequences/<method>/<compression>/mtcnn/**/*.jpg|png
    """
    def __init__(self, root_dir: str, compression: str = 'c23', transform=None, exclude=EXCLUDE_DATASETS):
        self.root_dir = root_dir
        self.compression = compression
        self.t = transform
        self.samples = []

        roots = [
            os.path.join(root_dir, 'original_sequences'),
            os.path.join(root_dir, 'manipulated_sequences'),
        ]
        for label, base in enumerate(roots):
            if not os.path.isdir(base):
                continue
            for method in os.listdir(base):
                if base.endswith('manipulated_sequences') and method in (exclude or set()):
                    continue
                d = os.path.join(base, method, compression, 'mtcnn')
                if not os.path.isdir(d):
                    continue
                for sub, _, fs in os.walk(d):
                    for f in fs:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(sub, f), label))

        if len(self.samples) == 0:
            print(f"[WARN] No images found under {root_dir} with compression={compression}/mtcnn")
        else:
            print(f"총 프레임 수: {len(self.samples):,}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, l = self.samples[i]
        try:
            img = Image.open(p).convert('RGB')
        except (UnidentifiedImageError, OSError):
            return None  # 손상 이미지 → collate_fn에서 걸러냄
        if self.t: img = self.t(img)
        return img, l
# ---------------------------------------------------------------------


def compute_metrics(model: nn.Module, loader, device: torch.device):
    model.eval()
    preds, trues, scores = [], [], []
    num_classes = None

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)                      # (B, C)
            if num_classes is None:
                num_classes = logits.shape[1]
            prob = torch.softmax(logits, dim=1)   # (B, C)

            pred = prob.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            trues.extend(y.cpu().numpy().tolist())

            if prob.shape[1] == 2:
                scores.extend(prob[:, 1].cpu().numpy().tolist())

    metrics = {
        'acc':    accuracy_score(trues, preds) if len(trues) else 0.0,
        'f1':     f1_score(trues, preds, average='macro') if len(trues) else 0.0,
        'prec':   precision_score(trues, preds, average='macro', zero_division=0) if len(trues) else 0.0,
        'recall': recall_score(trues, preds, average='macro', zero_division=0) if len(trues) else 0.0,
        'auc':    None,
    }
    if num_classes == 2 and len(scores) == len(trues) and len(trues) > 0:
        try:
            metrics['auc'] = roc_auc_score(trues, scores)
        except Exception:
            metrics['auc'] = None
    return metrics


# ===================== ESCFM =====================
class ESFCM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8, residual_mode: str = "x_plus_xmulscale"):
        super().__init__()
        assert reduction >= 1
        mid = max(1, in_channels // reduction)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1    = nn.Conv2d(in_channels, mid, 1, 1, 0, bias=False)
        self.relu     = nn.ReLU(inplace=True)
        self.conv_mid = nn.Conv2d(mid, mid, 3, 1, 1, bias=False)
        self.conv2    = nn.Conv2d(mid, in_channels, 1, 1, 0, bias=False)
        self.sigmoid  = nn.Sigmoid()

        assert residual_mode in ("x_plus_scale", "x_mul_scale", "x_plus_xmulscale")
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

        scale = self.sigmoid(m + a)  # (B, C, 1, 1)

        if self.residual_mode == "x_plus_scale":
            return x + scale
        elif self.residual_mode == "x_mul_scale":
            return x * scale
        else:
            return x + x * scale

# ===================== SE =====================
class SE(nn.Module):
    """
    Squeeze-and-Excitation (SE) block
    - ESFCM와 동일한 인터페이스(residual_mode)로 작성
    - Global Average Pooling 기반 채널 어텐션
    """
    def __init__(self, in_channels: int, reduction: int = 16,
                 residual_mode: str = "x_mul_scale"):
        super().__init__()
        assert reduction >= 1
        assert residual_mode in ("x_plus_scale", "x_mul_scale", "x_plus_xmulscale")

        self.residual_mode = residual_mode
        mid = max(1, in_channels // reduction)

        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation (두 개의 1x1 Conv는 FC와 동일한 역할)
        self.fc1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: (B, C, H, W) -> (B, C, 1, 1)
        s = self.avg_pool(x)
        # Excitation: (B, C, 1, 1) -> (B, C, 1, 1)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        scale = self.sigmoid(s)

        if self.residual_mode == "x_plus_scale":
            return x + scale
        elif self.residual_mode == "x_mul_scale":
            return x * scale
        else:  # "x_plus_xmulscale"
            return x + x * scale

# ===================== CBAM =====================
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    - Channel Attention + Spatial Attention
    - ESFCM / SE 와 동일한 residual_mode 인터페이스 제공
    """
    def __init__(self, in_channels: int, reduction: int = 16,
                 residual_mode: str = "x_mul_scale", kernel_size: int = 7):
        super().__init__()
        assert reduction >= 1
        assert residual_mode in ("x_plus_scale", "x_mul_scale", "x_plus_xmulscale")

        self.residual_mode = residual_mode

        # ----- Channel Attention -----
        mid = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 공유 MLP (1x1 Conv로 구현)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid_c = nn.Sigmoid()

        # ----- Spatial Attention -----
        # AvgPool과 MaxPool을 채널 차원에서 concat → Conv
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                      stride=1, padding=kernel_size // 2, bias=False)
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- Channel Attention -----
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale_c = self.sigmoid_c(avg_out + max_out)  # (B, C, 1, 1)

        x_c = x * scale_c  # 채널 어텐션 적용

        # ----- Spatial Attention -----
        avg_out = torch.mean(x_c, dim=1, keepdim=True)            # (B, 1, H, W)
        max_out, _ = torch.max(x_c, dim=1, keepdim=True)          # (B, 1, H, W)
        s = torch.cat([avg_out, max_out], dim=1)                  # (B, 2, H, W)
        scale_s = self.sigmoid_s(self.conv_spatial(s))            # (B, 1, H, W)

        # ----- Residual Mode (scale_c, scale_s를 결합해 적용) -----
        if self.residual_mode == "x_plus_scale":
            scale = scale_c * scale_s                              # 브로드캐스트로 (B,C,H,W) 정합
            return x + scale
        elif self.residual_mode == "x_mul_scale":
            return x * (scale_c * scale_s)
        else:  # "x_plus_xmulscale"
            return x + x * (scale_c * scale_s)


# ===================== Feature map 채널 추정 =====================
@torch.no_grad()
def _infer_feat_channels(model: nn.Module, device: Optional[torch.device] = None) -> int:
    if hasattr(model, "num_features") and isinstance(model.num_features, int) and model.num_features > 0:
        return model.num_features

    fi = getattr(model, "feature_info", None)
    if fi:
        try:
            return int(fi[-1]["num_chs"])
        except Exception:
            pass

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
        if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
            y = model.forward_features(x)
            if isinstance(y, torch.Tensor) and y.ndim == 4:
                return int(y.shape[1])
    finally:
        model.train(was_train)

    return 1280


def attach_esfcm_module(model: nn.Module,
                        reduction: int = 8,
                        residual_mode: str = "x_plus_xmulscale",
                        device: Optional[torch.device] = None) -> nn.Module:
    # 모델의 실제 디바이스
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device if device is not None else torch.device('cpu')

    c = _infer_feat_channels(model, device=model_device)
    esfcm = ESFCM(c, reduction=reduction, residual_mode=residual_mode).to(model_device)
    model.add_module("esfcm_before_head", esfcm)

    if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
        old_ff = model.forward_features

        def wrapped_forward_features(self, x):
            f = old_ff(x)
            if isinstance(f, torch.Tensor) and f.ndim == 4:
                f = self.esfcm_before_head(f)
            return f

        model.forward_features = types.MethodType(wrapped_forward_features, model)
    else:
        print("[WARN] model has no forward_features; ESFCM not injected automatically.")
    return model


def attach_se_module(model: nn.Module,
                     reduction: int = 16,
                     residual_mode: str = "x_mul_scale",
                     device: Optional[torch.device] = None) -> nn.Module:
    # 모델의 실제 디바이스
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device if device is not None else torch.device('cpu')

    c = _infer_feat_channels(model, device=model_device)
    se = SE(c, reduction=reduction, residual_mode=residual_mode).to(model_device)
    model.add_module("se_before_head", se)

    if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
        old_ff = model.forward_features

        def wrapped_forward_features(self, x):
            f = old_ff(x)
            if isinstance(f, torch.Tensor) and f.ndim == 4:
                f = self.se_before_head(f)
            return f

        model.forward_features = types.MethodType(wrapped_forward_features, model)
    else:
        print("[WARN] model has no forward_features; SE not injected automatically.")
    return model

def attach_cbam_module(model: nn.Module,
                       reduction: int = 16,
                       residual_mode: str = "x_mul_scale",
                       kernel_size: int = 7,
                       device: Optional[torch.device] = None) -> nn.Module:
    """
    timm 스타일 모델( EfficientNet 등 )의 classifier head 직전 feature map에 CBAM을 삽입합니다.
    - forward_features가 있는 모델에서 동작
    - 모듈명: cbam_before_head
    """
    # 모델 디바이스 파악
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device if device is not None else torch.device('cpu')

    # 채널 수 추정 및 CBAM 생성
    c = _infer_feat_channels(model, device=model_device)
    cbam = CBAM(in_channels=c,
                reduction=reduction,
                residual_mode=residual_mode,
                kernel_size=kernel_size).to(model_device)

    # 모듈 등록
    model.add_module("cbam_before_head", cbam)

    # forward_features 래핑하여 헤드 직전에 주입
    if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
        old_ff = model.forward_features

        def wrapped_forward_features(self, x):
            f = old_ff(x)
            if isinstance(f, torch.Tensor) and f.ndim == 4:
                f = self.cbam_before_head(f)
            return f

        model.forward_features = types.MethodType(wrapped_forward_features, model)
    else:
        print("[WARN] model has no forward_features; CBAM not injected automatically.")
    return model

# ===================== EarlyStopping =====================
class EarlyStopping:
    """val_loss(작을수록 좋음)이 patience 동안 개선되지 않으면 조기 종료"""
    def __init__(self, patience=5, min_delta=0.0, verbose=False, path='checkpoint_es.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.checkpoint_path = path

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(f"[EarlyStopping] val_loss improved → {val_loss:.4f} (ckpt saved)")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] no improve ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


# ===================== Example / Train Loop =====================
if __name__ == "__main__":
    import argparse
    import timm
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="FF++ root directory (contains original_sequences/ and manipulated_sequences/)")
    parser.add_argument("--compression", type=str, default="raw",
                        help="FF++ compression level: raw / c23 / c40")
    parser.add_argument("--model", type=str, default="tf_efficientnet_b7")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda[:N] or just N")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/efficientnet_escfm")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--tag", type=str, default=None, help="파일명 태그(기본: 모델명)")
    parser.add_argument("--eval-after-train", action="store_true", help="학습 종료 후 최종 검증 실행")

    # === ESFCM / SE / CBAM 옵션 ===
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-esfcm", action="store_true", help="Attach ESFCM before classifier head")
    group.add_argument("--use-se", action="store_true", help="Attach SE before classifier head")
    group.add_argument("--use-cbam", action="store_true", help="Attach CBAM before classifier head")

    parser.add_argument("--esfcm-reduction", type=int, default=8)
    parser.add_argument("--esfcm-mode",
                        choices=["x_plus_scale", "x_mul_scale", "x_plus_xmulscale"],
                        default="x_plus_xmulscale")
    
    parser.add_argument("--se-reduction", type=int, default=16,
                        help="Reduction ratio for SE block (default: 16)")
    parser.add_argument("--se-mode",
                        choices=["x_plus_scale", "x_mul_scale", "x_plus_xmulscale"],
                        default="x_mul_scale",
                        help="Residual connection mode for SE block")
    
    parser.add_argument("--cbam-reduction", type=int, default=16,
                        help="Reduction ratio for CBAM channel attention (default: 16)")
    parser.add_argument("--cbam-mode",
                        choices=["x_plus_scale", "x_mul_scale", "x_plus_xmulscale"],
                        default="x_mul_scale",
                        help="Residual connection mode for CBAM block")
    parser.add_argument("--cbam-kernel", type=int, default=7,
                        help="Kernel size for CBAM spatial attention (default: 7)")

    # ==== Resume ====
    parser.add_argument("--resume", type=str, default="",
                        help="체크포인트(.pth) 경로 또는 체크포인트 디렉토리(가장 최근 epoch_* 자동 선택)")
    parser.add_argument("--resume-strict", action="store_true",
                        help="state_dict 로드 시 strict=True")

    # parse
    args = parser.parse_args()

    def normalize_device(dev_str: str) -> torch.device:
        s = str(dev_str).strip().lower()
        if s == "cpu":
            return torch.device("cpu")
        if s.isdigit():
            return torch.device(f"cuda:{s}") if torch.cuda.is_available() else torch.device("cpu")
        if s.startswith("cuda"):
            return torch.device(s if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")
    
    device = normalize_device(args.device)

    # model
    model = timm.create_model(args.model, pretrained=True, num_classes=args.num_classes).to(device)
    if args.use_esfcm:
        model = attach_esfcm_module(
            model,
            reduction=args.esfcm_reduction,
            residual_mode=args.esfcm_mode,
            device=device
        )
        print("[ESFCM] attached at classifier head input")
    elif args.use_se:
        model = attach_se_module(
            model,
            reduction=args.se_reduction,
            residual_mode=args.se_mode,
            device=device
        )
        print("[SE] attached at classifier head input")
    elif args.use_cbam:
        model = attach_cbam_module(
            model,
            reduction=args.cbam_reduction,
            residual_mode=args.cbam_mode,
            kernel_size=args.cbam_kernel,
            device=device
        )
        print("[CBAM] attached at classifier head input")

    # transform
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    ds = FFPP_RGB(args.data_dir, compression=args.compression, transform=tfm)
    # 손상 프레임(None) 제거를 위한 collate
    collate = lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x is not None])

    n_tr = int(0.8 * len(ds))
    tr_ds, va_ds = torch.utils.data.random_split(ds, [n_tr, len(ds) - n_tr],
                                                 generator=torch.Generator().manual_seed(42))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True,  collate_fn=collate)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True,  collate_fn=collate)

    print(f"Train iters/epoch: {len(tr_ld)} | Val iters/epoch: {len(va_ld)}")

    # optim/loss
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # checkpoints
    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag_parts = [args.model]
    if args.use_esfcm:
        tag_parts.append(f"esfcm_r{args.esfcm_reduction}_{args.esfcm_mode}")
    if args.use_se:
        tag_parts.append(f"se_r{args.se_reduction}_{args.se_mode}")
    if args.use_cbam:
        tag_parts.append(f"cbam_r{args.cbam_reduction}_{args.cbam_mode}_k{args.cbam_kernel}")
    tag_parts.append(args.compression)
    tag = "_".join(tag_parts)

    best_ckpt = ckpt_dir / f"best_{tag}.pth"
    es_ckpt   = ckpt_dir / f"es_{tag}.pth"

    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta, verbose=True, path=str(es_ckpt))

    # ===== Resume helpers =====
    def pick_latest_epoch_ckpt(dir_path: Path, prefix: str = "epoch_") -> Optional[Path]:
        if not dir_path.exists() or not dir_path.is_dir():
            return None
        cands = sorted(dir_path.glob(f"{prefix}*.pth"))
        if not cands:
            return None
        def _key(p: Path):
            stem = p.stem  # epoch_xxx
            nums = [int(s) for s in stem.split("_") if s.isdigit()]
            return nums[-1] if nums else -1
        cands_sorted = sorted(cands, key=lambda p: (_key(p), p.stat().st_mtime))
        return cands_sorted[-1]

    def load_resume(ckpt_path: Path, strict: bool = False):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"[Resume][WARN] missing keys({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
        if unexpected:
            print(f"[Resume][WARN] unexpected keys({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

        if 'optimizer' in ckpt:
            try:
                opt.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                print(f"[Resume][WARN] optimizer state load failed: {e}")

        start_epoch = int(ckpt.get('epoch', 0))
        best_f1_resume = float(ckpt.get('best_f1', 0.0))

        # RNG 복구(선택)
        rng = ckpt.get('rng', None)
        if rng:
            try:
                torch.set_rng_state(rng['torch'])
                if torch.cuda.is_available() and rng.get('cuda') is not None:
                    torch.cuda.set_rng_state_all(rng['cuda'])
                np.random.set_state(rng['numpy'])
            except Exception as e:
                print(f"[Resume][WARN] RNG restore failed: {e}")

        print(f"[Resume] Loaded: {ckpt_path.name} | epoch={start_epoch} | best_f1={best_f1_resume:.4f}")
        return start_epoch, best_f1_resume

    # ===== Try resume =====
    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            latest = pick_latest_epoch_ckpt(resume_path)
            if latest is None:
                print(f"[Resume][WARN] no epoch_*.pth under {resume_path}, skip resume.")
            else:
                start_epoch, best_f1 = load_resume(latest, strict=args.resume_strict)
        elif resume_path.is_file():
            start_epoch, best_f1 = load_resume(resume_path, strict=args.resume_strict)
        else:
            print(f"[Resume][WARN] invalid path: {resume_path}, skip resume.")

    # ===== Train loop =====
    for ep in range(start_epoch + 1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tqdm(tr_ld, desc=f"Epoch {ep}/{args.epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()

        tr_m = compute_metrics(model, tr_ld, device)
        va_m = compute_metrics(model, va_ld, device)
        avg_loss = running / max(1, len(tr_ld))
        print(f"[{ep}] loss:{avg_loss:.4f} | "
              f"Tr acc:{tr_m['acc']:.4f} f1:{tr_m['f1']:.4f} auc:{(tr_m['auc'] or 0):.4f} || "
              f"Va acc:{va_m['acc']:.4f} f1:{va_m['f1']:.4f} auc:{(va_m['auc'] or 0):.4f}")

        # (1) 에포크별 체크포인트 (옵티마/베스트/RNG 포함)
        save_obj = {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'epoch': ep,
            'best_f1': best_f1,
            'rng': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
            },
        }
        torch.save(save_obj, ckpt_dir / f"epoch_{tag}_{ep:03d}.pth")

        # (2) 베스트 체크포인트 (F1 기준)
        if va_m['f1'] > best_f1:
            best_f1 = va_m['f1']
            save_obj.update({'best_f1': best_f1, 'epoch': ep})
            torch.save(save_obj, best_ckpt)
            print(f"  ↑ best updated: F1={best_f1:.4f} -> {best_ckpt}")

        # (3) EarlyStopping 체크포인트 (val_loss = 1 - f1)
        early(1 - va_m['f1'], model)
        if early.early_stop:
            print(f"[EarlyStopping] Stopped. ES checkpoint: {early.checkpoint_path}")
            break

    print(f"Training done. Best F1: {best_f1:.4f}. Best ckpt: {best_ckpt}")

    # (옵션) 학습 종료 후 최종 검증
    if args.eval_after_train:
        final_ckpt = best_ckpt if best_ckpt.exists() else Path(es_ckpt)
        if final_ckpt.exists():
            state = torch.load(final_ckpt, map_location=device)
            model.load_state_dict(state.get('model', state), strict=False)
            print(f"[Final Eval] Loaded {final_ckpt.name}")
        va_m_final = compute_metrics(model, va_ld, device)
        print("\n=== Final Validation ===")
        print(f"Accuracy : {va_m_final['acc']:.4f}")
        print(f"F1 score : {va_m_final['f1']:.4f}")
        print(f"Precision: {va_m_final['prec']:.4f}")
        print(f"Recall   : {va_m_final['recall']:.4f}")
        print(f"AUC      : {va_m_final['auc'] if va_m_final['auc'] is not None else 'N/A'}")
