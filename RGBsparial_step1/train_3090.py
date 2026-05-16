# train_rgb.py
#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import sys

from pathlib import Path
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ───────── 모델 정의 ────────────────────────────
from Xception.xception import xception
from maxvit.maxvit import MaxViT
from hornet.hornet import hornet_large_gf
from coatnet.coatnet import coatnet_0
from resnet.resnet_cbam import resnet50
from efficientnet_pytorch import EfficientNet

# 필요 시 유지
from hornet.focal_loss import FocalLoss
# from cbam_v2 import LazyCBAM2d

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Frequency_step2"))
)

from models.convnextv2 import convnextv2_tiny


# ───────── FF++ 디렉터리 맵 ──────────────────────────────────────
DATASETS = {
    "original": "original_sequences/youtube",
    "DeepFakeDetection_original": "original_sequences/actors",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "DeepFakeDetection": "manipulated_sequences/DeepFakeDetection",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceShifter": "manipulated_sequences/FaceShifter",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}


# ───────── Early Stopping ───────────────────────────────────────
class EarlyStopping:
    """
    val_loss가 patience epoch 동안 개선되지 않으면 조기 종료.
    여기서는 val_loss 대신 1 - val_f1을 넣어서 사용함.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
        verbose=False,
        path="checkpoint_earlystop.pth",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.checkpoint_path = path

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)

            if self.verbose:
                print(
                    f"[EarlyStopping] val_loss improved → {val_loss:.4f} "
                    f"(saved: {self.checkpoint_path})"
                )
        else:
            self.counter += 1

            if self.verbose:
                print(
                    f"[EarlyStopping] no improve "
                    f"({self.counter}/{self.patience})"
                )

            if self.counter >= self.patience:
                self.early_stop = True


# ───────── Dataset ─────────────────────────────────────────────
class FFPP_RGB(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        compression="raw",
        transform=None,
        selected_keys=None,
    ):
        self.t = transform
        self.samples = []

        if selected_keys is None:
            selected_keys = list(DATASETS.keys())

        for key in selected_keys:
            base = os.path.join(root_dir, DATASETS[key], compression, "mtcnn")

            if not os.path.isdir(base):
                continue

            label = 0 if "original" in key else 1

            for sub, _, fs in os.walk(base):
                for f in fs:
                    if f.lower().endswith(("png", "jpg", "jpeg")):
                        self.samples.append((os.path.join(sub, f), label, key))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label, key = self.samples[idx]

        try:
            img = Image.open(p).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

        if self.t:
            img = self.t(img)

        return img, label


# ───────── Model Loader ─────────────────────────────────────────
def load_backbone(
    name: str,
    cbam: bool,
    cbam_kernel: int,
    cbam_reduction: int,
):
    if name == "xception":
        return xception(
            num_classes=2,
            use_cbam=cbam,
            cbam_kernel=cbam_kernel,
            cbam_reduction=cbam_reduction,
        )

    if name == "maxvit":
        return MaxViT(
            num_classes=2,
            use_cbam=cbam,
        )

    if name == "hornet":
        return hornet_large_gf(
            num_classes=2,
            use_cbam=cbam,
        )

    if name == "coatnet":
        return coatnet_0(
            num_classes=2,
        )

    if name == "efficientnet-b7":
        return EfficientNet.from_pretrained(
            "efficientnet-b7",
            num_classes=2,
        )

    if name == "efficientnet-b4":
        return EfficientNet.from_pretrained(
            "efficientnet-b4",
            num_classes=2,
        )

    if name == "convnext-tiny":
        return convnextv2_tiny(
            in_chans=3,
            num_classes=2,
            use_cbam=cbam,
        )

    if name == "resnet50":
        # resnet.resnet_cbam.resnet50 구현에 따라 인자 지원 여부가 다를 수 있음.
        # 현재 네 코드 기준으로는 pretrained=True만 전달하는 구조가 가장 안전함.
        return resnet50(
            num_classes=2,
            pretrained=True,
        )

    raise ValueError(f"Unsupported model: {name}")


# ───────── Metrics ──────────────────────────────────────────────
@torch.no_grad()
def metrics(model, loader, device):
    model.eval()

    preds = []
    targets = []

    for batch in loader:
        if batch is None:
            continue

        x, y = batch
        x = x.to(device)

        logits = model(x)
        pred = logits.argmax(1).cpu().tolist()

        preds.extend(pred)
        targets.extend(y.tolist())

    return {
        "acc": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, average="macro"),
        "prec": precision_score(
            targets,
            preds,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            targets,
            preds,
            average="macro",
            zero_division=0,
        ),
    }


def safe_collate(batch):
    batch = [x for x in batch if x is not None]

    if len(batch) == 0:
        return None

    return default_collate(batch)


# ───────── Main ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gpu", type=int, default=0, help="GPU 번호")
    ap.add_argument(
        "--model",
        required=True,
        choices=[
            "xception",
            "resnet50",
            "maxvit",
            "hornet",
            "coatnet",
            "efficientnet-b7",
            "efficientnet-b4",
            "convnext-tiny",
        ],
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True, help="Checkpoint directory")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--use-cbam", action="store_true")
    ap.add_argument("--cbam-kernel", type=int, default=7)
    ap.add_argument("--cbam-reduction", type=int, default=16)

    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, nargs="+", default=None)

    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="resume 또는 validation에 사용할 weight path",
    )
    ap.add_argument("--mode", choices=["train", "val"], default="train")

    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 재현성 고정
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ImageNet pretrained 모델을 사용하는 경우에는 ImageNet normalization 권장
    # 단, 기존 실험과의 공정 비교를 위해 [0.5, 0.5, 0.5] normalization을 유지하고 싶다면
    # 아래 Normalize 부분만 기존 방식으로 바꾸면 됨.
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    ds = FFPP_RGB(
        args.data_dir,
        transform=tfm,
    )

    print(f"Total frames: {len(ds):,}")

    if len(ds) == 0:
        raise RuntimeError(
            "Dataset is empty. Check --data-dir and FF++ directory structure."
        )

    tr_len = int(0.8 * len(ds))
    va_len = len(ds) - tr_len

    tr_ds, va_ds = random_split(
        ds,
        [tr_len, va_len],
        torch.Generator().manual_seed(seed),
    )

    nw = 4

    tr_ld = DataLoader(
        tr_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
        prefetch_factor=2,
        collate_fn=safe_collate,
    )

    va_ld = DataLoader(
        va_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(nw > 0),
        prefetch_factor=2,
        collate_fn=safe_collate,
    )

    model = load_backbone(
        args.model,
        args.use_cbam,
        args.cbam_kernel,
        args.cbam_reduction,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model}")
    print(f"Params: {num_params:.1f}M")

    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag = args.model + ("_cbam" if args.use_cbam else "")

    best_path = ckpt_dir / f"{tag}_best.pth"
    earlystop_path = ckpt_dir / f"{tag}_earlystop.pth"
    last_path = ckpt_dir / f"{tag}_last.pth"

    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Best weight      : {best_path.name}")
    print(f"EarlyStop weight : {earlystop_path.name}")
    print(f"Last weight      : {last_path.name}")

    # ── resume weight load ─────────────────────────────
    start_epoch = 1
    best_f1 = 0.0

    resume_path = Path(args.resume) if args.resume else None

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)

        # 과거 full checkpoint도 호환 가능
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_f1 = ckpt.get("best_f1", 0.0)
        else:
            model.load_state_dict(ckpt)

        print(f"▶ Loaded weight from: {resume_path}")
        print(f"▶ start_epoch={start_epoch}, best_f1={best_f1:.4f}")

    # ── Train ────────────────────────────────
    if args.mode == "train":
        criterion = nn.CrossEntropyLoss()

        # 필요하면 FocalLoss로 교체 가능
        # criterion = FocalLoss(
        #     gamma=args.gamma,
        #     alpha=args.alpha,
        #     task_type="multi-class",
        #     num_classes=2,
        # ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=5e-2,
        )

        stopper = EarlyStopping(
            patience=args.patience,
            min_delta=0.0,
            verbose=True,
            path=earlystop_path,
        )

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            run_loss = 0.0
            valid_steps = 0

            pbar = tqdm(tr_ld, desc=f"Epoch {epoch}/{args.epochs}")

            for batch in pbar:
                if batch is None:
                    continue

                x, y = batch
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                logits = model(x)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                run_loss += loss.item()
                valid_steps += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if valid_steps == 0:
                raise RuntimeError("No valid training batch. Check image files.")

            val_m = metrics(model, va_ld, device)
            avg_loss = run_loss / valid_steps

            print(
                f"[Ep {epoch}] "
                f"loss={avg_loss:.4f}  "
                f"F1={val_m['f1']:.4f}  "
                f"ACC={val_m['acc']:.4f}  "
                f"PREC={val_m['prec']:.4f}  "
                f"RECALL={val_m['recall']:.4f}"
            )

            # Early-stopping 기준: 1 - macro F1
            stopper(1.0 - val_m["f1"], model)

            # Best 저장
            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                torch.save(model.state_dict(), best_path)
                print(f"✓ Best model saved: {best_path}")

            # epoch별 저장은 하지 않음
            if stopper.early_stop:
                print("Early stopping triggered")
                break

        # Last 저장
        torch.save(model.state_dict(), last_path)
        print(f"✓ Last model saved: {last_path}")

        print(f"Training finished. Best F1 = {best_f1:.4f}")

    # ── Validation ────────────────────────────────
    else:
        if resume_path is None:
            raise ValueError(
                "--mode val에서는 --resume으로 평가할 weight path를 지정해야 함"
            )

        model.eval()
        val_m = metrics(model, va_ld, device)

        print("Validation:")
        print(f"  ACC    : {val_m['acc']:.4f}")
        print(f"  F1     : {val_m['f1']:.4f}")
        print(f"  PREC   : {val_m['prec']:.4f}")
        print(f"  RECALL : {val_m['recall']:.4f}")


if __name__ == "__main__":
    main()