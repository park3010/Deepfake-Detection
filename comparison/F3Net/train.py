# f3net_train.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
F3Net FF++ RGB training script
- Dataset structure aligned with previous RGB model training script
- Supports F3Net mode: Original / FAD / LFS / Both
- Saves epoch checkpoints + best + earlystop
- 80:20 train/val split
"""

import os
import re
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# -----------------------------
# FF++ directory mapping
# -----------------------------
DATASETS = {
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures',
}


# -----------------------------
# Seed
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Early stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False, path='checkpoint_es.pth'):
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
                print(f"[EarlyStopping] val_loss improved → {val_loss:.4f} (ckpt saved)")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] no improve ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


# -----------------------------
# Dataset
# -----------------------------
class FFPP_RGB(torch.utils.data.Dataset):
    def __init__(self, root_dir, compression='raw', transform=None, selected_keys=None):
        self.t = transform
        self.samples = []

        if selected_keys is None:
            selected_keys = list(DATASETS.keys())

        for key in selected_keys:
            base = os.path.join(root_dir, DATASETS[key], compression, 'mtcnn')
            if not os.path.isdir(base):
                continue

            label = 0 if 'original' in key else 1
            for sub, _, fs in os.walk(base):
                for f in fs:
                    if f.lower().endswith(('png', 'jpg', 'jpeg')):
                        self.samples.append((os.path.join(sub, f), label, key))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label, key = self.samples[idx]
        try:
            img = Image.open(p).convert('RGB')
        except (UnidentifiedImageError, OSError):
            return None

        if self.t:
            img = self.t(img)
        return img, label


# -----------------------------
# F3Net loader
# -----------------------------
def find_f3net_class(f3net_root: str):
    if f3net_root:
        sys.path.append(f3net_root)

    try:
        from models import F3Net as F3NetModel
        return F3NetModel
    except Exception:
        # fallback: if user saved model file under another module path
        try:
            from model import F3Net as F3NetModel
            return F3NetModel
        except Exception as e:
            raise ImportError(
                "Could not import F3Net. Check --f3net-root and the location of models.py"
            ) from e


def build_f3net(args, device):
    F3NetModel = find_f3net_class(args.f3net_root)

    tried = []
    model = None
    for ctor in [
        lambda: F3NetModel(mode=args.f3net_mode, num_classes=2),
        lambda: F3NetModel(mode=args.f3net_mode, num_classes=1),
        lambda: F3NetModel(num_classes=2),
        lambda: F3NetModel(num_classes=1),
        lambda: F3NetModel(),
    ]:
        try:
            model = ctor().to(device)
            break
        except Exception as e:
            tried.append(str(e))

    if model is None:
        raise RuntimeError("F3Net instantiation failed:\n- " + "\n- ".join(tried))
    return model


# -----------------------------
# Logits helper
# -----------------------------
def extract_logits(output):
    """
    F3Net forward() returns (feature, logits)
    but allow tensor / tuple / list robustly.
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        # prefer 2D tensor with class dim
        cands = [x for x in output if torch.is_tensor(x)]
        for x in cands:
            if x.ndim == 2 and x.size(1) in [1, 2]:
                return x
        return cands[-1]

    raise TypeError(f"Unsupported model output type: {type(output)}")


def probs_and_preds_from_logits(logits):
    if logits.ndim == 2 and logits.size(1) == 2:
        probs_fake = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        return probs_fake, preds
    elif logits.ndim == 2 and logits.size(1) == 1:
        probs_fake = torch.sigmoid(logits[:, 0])
        preds = (probs_fake >= 0.5).long()
        return probs_fake, preds
    else:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_all, trues_all = [], []
    run_loss = 0.0
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        logits = extract_logits(output)

        if logits.ndim == 2 and logits.size(1) == 2:
            loss = criterion_ce(logits, y)
        elif logits.ndim == 2 and logits.size(1) == 1:
            loss = criterion_bce(logits[:, 0], y.float())
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        _, preds = probs_and_preds_from_logits(logits)

        run_loss += loss.item()
        preds_all.extend(preds.cpu().tolist())
        trues_all.extend(y.cpu().tolist())

    return {
        "loss": run_loss / max(1, len(loader)),
        "acc": accuracy_score(trues_all, preds_all),
        "f1_macro": f1_score(trues_all, preds_all, average="macro", zero_division=0),
        "precision": precision_score(trues_all, preds_all, average="macro", zero_division=0),
        "recall": recall_score(trues_all, preds_all, average="macro", zero_division=0),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0, help='GPU index')
    ap.add_argument('--data-dir', required=True, help='FF++ root')
    ap.add_argument('--compression', type=str, default='raw')
    ap.add_argument('--ckpt', required=True, help='Checkpoint directory')

    ap.add_argument('--f3net-root', required=True, help='Directory containing F3Net models.py')
    ap.add_argument('--f3net-mode', default='Both',
                    choices=['Original', 'FAD', 'LFS', 'Both'])

    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mode', choices=['train', 'val'], default='train')
    args = ap.parse_args()

    set_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}")

    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    ds = FFPP_RGB(
        root_dir=args.data_dir,
        compression=args.compression,
        transform=tfm
    )
    print(f"Total frames: {len(ds):,}")

    tr_len = int(0.8 * len(ds))
    va_len = len(ds) - tr_len
    tr_ds, va_ds = random_split(ds, [tr_len, va_len], torch.Generator().manual_seed(args.seed))

    collate = lambda b: default_collate([x for x in b if x is not None])

    tr_ld = DataLoader(
        tr_ds, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=(device.type == 'cuda'),
        persistent_workers=True, prefetch_factor=2,
        collate_fn=collate
    )
    va_ld = DataLoader(
        va_ds, batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=(device.type == 'cuda'),
        persistent_workers=True, prefetch_factor=2,
        collate_fn=collate
    )

    model = build_f3net(args, device)
    print(f"Model F3Net-{args.f3net_mode} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"f3net_{args.f3net_mode.lower()}"

    pattern = re.compile(fr"{re.escape(tag)}_ep(\d+)\.pth$")
    start_epoch, best_f1 = 1, 0.0
    optim_state = None

    if args.resume:
        resume_path = Path(args.resume)
    else:
        saved = list(ckpt_dir.glob(f"{tag}_ep*.pth"))
        if saved:
            last_ep = max(int(pattern.match(p.name).group(1)) for p in saved if pattern.match(p.name))
            resume_path = ckpt_dir / f"{tag}_ep{last_ep:03d}.pth"
        else:
            resume_path = None

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)

    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        if 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'], strict=False)
            optim_state = ckpt.get('optim_state', None)
            best_f1 = ckpt.get('best_f1', 0.0)
            start_epoch = ckpt.get('epoch', 1) + 1
        else:
            model.load_state_dict(ckpt, strict=False)
            m = re.search(r'_ep(\d+)\.pth$', resume_path.name)
            start_epoch = int(m.group(1)) + 1 if m else 1

        if optim_state is not None:
            optimizer.load_state_dict(optim_state)

        print(f"▶ Resumed from {resume_path.name} (start_epoch={start_epoch})")

    if args.mode == 'train':
        stopper = EarlyStopping(
            patience=args.patience,
            min_delta=0.0,
            verbose=True,
            path=ckpt_dir / f"{tag}_earlystop.pth"
        )

        criterion_ce = nn.CrossEntropyLoss()
        criterion_bce = nn.BCEWithLogitsLoss()

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            run_loss = 0.0

            pbar = tqdm(tr_ld, desc=f"Epoch {epoch}/{args.epochs}")
            for batch in pbar:
                x, y = batch
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                output = model(x)
                logits = extract_logits(output)

                if logits.ndim == 2 and logits.size(1) == 2:
                    loss = criterion_ce(logits, y)
                elif logits.ndim == 2 and logits.size(1) == 1:
                    loss = criterion_bce(logits[:, 0], y.float())
                else:
                    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

                loss.backward()
                optimizer.step()

                run_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            val_m = evaluate(model, va_ld, device)
            train_loss = run_loss / max(1, len(tr_ld))

            print(
                f"[Ep {epoch}] "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_m['loss']:.4f}  "
                f"F1_macro={val_m['f1_macro']:.4f}  "
                f"ACC={val_m['acc']:.4f}  "
                f"PREC={val_m['precision']:.4f}  "
                f"REC={val_m['recall']:.4f}"
            )

            stopper(val_m['loss'], model)
            if stopper.early_stop:
                print("Early stopping triggered")
                break

            if val_m['f1_macro'] > best_f1:
                best_f1 = val_m['f1_macro']
                torch.save(model.state_dict(), ckpt_dir / f"{tag}_best.pth")

            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_f1': best_f1,
            }, ckpt_dir / f"{tag}_ep{epoch:03d}.pth")

        print(f"Training finished. Best F1_macro = {best_f1:.4f}")

    else:
        val_m = evaluate(model, va_ld, device)
        print("Validation:", val_m)


if __name__ == '__main__':
    main()