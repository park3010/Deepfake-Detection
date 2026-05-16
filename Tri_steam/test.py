# test_tri.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tri/Multi-stream external test script
- Video-level evaluation:
  frame fake probability mean -> video prediction
- Metrics:
  Accuracy, Precision, Recall, F1-macro, F1-binary, AUC
"""

import os
import gc
import glob
import argparse
import math
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.cuda.amp import autocast

from transformers import CLIPImageProcessor

from train import (
    parse_streams,
    make_tag,
    make_rgb_input,
    make_wavelet_input,
    make_dct_input,
    build_model,
)


TEST_DATASETS = {
    "Celeb": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-real",
            "/home/oem/deepfake/hdd_5TB/Celeb/YouTube-real",
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-synthesis",
        ],
    },
    "DFD": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences",
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences",
        ],
    },
    "DeepfakeTIMIT": {
        "real": [],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality",
        ],
    },
    "WildDeepfake": {
        "root": "/home/oem/deepfake/hdd_5TB/WildDeepfake",
        "splits": ["train", "test"],
    },
}


class TriFrameDataset(Dataset):
    def __init__(
        self,
        frame_paths: List[str],
        streams: List[str],
        args,
        clip_processor=None,
    ):
        self.frames = frame_paths
        self.streams = streams
        self.args = args
        self.clip_processor = clip_processor
        self.resize = transforms.Resize((args.img_size, args.img_size))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path = self.frames[idx]

        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

        img_resized = self.resize(img)
        arr_rgb = np.array(img_resized).astype(np.float32)
        arr_rgb_uint8 = arr_rgb.astype(np.uint8)
        arr_bgr = arr_rgb[:, :, ::-1].copy()

        item = {}

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
            item["semantic"] = self.clip_processor(
                images=img,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)

        return item


def tri_frame_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    keys = batch[0].keys()
    out = {}

    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    return out


def collect_frames(video_dir: str):
    frames = sorted(glob.glob(os.path.join(video_dir, "*.png")))
    frames += sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    frames += sorted(glob.glob(os.path.join(video_dir, "*.jpeg")))
    return frames


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

        return {
            "real": real_roots,
            "fake": fake_roots,
        }

    if ds_name == "DeepfakeTIMIT":
        fake_roots = []

        for quality_root in cfg["fake"]:
            if not os.path.isdir(quality_root):
                continue

            for speaker in os.listdir(quality_root):
                sp_path = os.path.join(quality_root, speaker)
                if os.path.isdir(sp_path):
                    fake_roots.append(sp_path)

        return {
            "real": [],
            "fake": fake_roots,
        }

    return cfg


@torch.no_grad()
def evaluate_dataset(
    model,
    device,
    roots: List[str],
    label_value: int,
    streams: List[str],
    args,
    clip_processor=None,
):
    y_true, y_pred, y_score = [], [], []
    # use_amp = device.type == "cuda"
    use_amp = False

    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] 경로 없음: {root}")
            continue

        vids = sorted(os.listdir(root))

        for vid in tqdm(vids, desc=f"[{label_value}] {os.path.basename(root)}"):
            vid_dir = os.path.join(root, vid)

            if not os.path.isdir(vid_dir):
                continue

            frames = collect_frames(vid_dir)

            if len(frames) == 0:
                continue

            ds = TriFrameDataset(
                frame_paths=frames,
                streams=streams,
                args=args,
                clip_processor=clip_processor,
            )

            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=(device.type == "cuda"),
                collate_fn=tri_frame_collate,
            )

            sum_p = 0.0
            cnt = 0

            for batch in tqdm(loader, desc=f" frames of {vid}", leave=False):
                if batch is None:
                    continue

                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)

                with torch.inference_mode():
                    with autocast(enabled=use_amp):
                        out = model(batch)
                        logits = out["logits"]

                    # softmax는 항상 fp32에서 계산
                    logits = logits.float()

                    if not torch.isfinite(logits).all():
                        bad = (~torch.isfinite(logits)).sum().item()
                        print(f"[WARN] non-finite logits detected: {bad} values | vid={vid}")
                        logits = torch.nan_to_num(
                            logits,
                            nan=0.0,
                            posinf=50.0,
                            neginf=-50.0,
                        )

                    prob_fake = torch.softmax(logits, dim=1)[:, 1]

                    if not torch.isfinite(prob_fake).all():
                        bad = (~torch.isfinite(prob_fake)).sum().item()
                        print(f"[WARN] non-finite probability detected: {bad} values | vid={vid}")
                        prob_fake = torch.nan_to_num(
                            prob_fake,
                            nan=0.5,
                            posinf=1.0,
                            neginf=0.0,
                        )

                sum_p += float(prob_fake.sum().item())
                cnt += int(prob_fake.numel())

                del batch, out, logits, prob_fake

            if cnt == 0:
                continue

            avg_p = sum_p / cnt

            if not np.isfinite(avg_p):
                print(f"[WARN] non-finite avg_p -> skip video | root={root} | vid={vid} | avg_p={avg_p}")
                continue

            pred = 1 if avg_p >= args.threshold else 0

            y_true.append(label_value)
            y_pred.append(pred)
            y_score.append(avg_p)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    return y_true, y_pred, y_score


def safe_confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def build_metrics_dict(dataset_name, y_true, y_pred, y_score):
    if len(y_true) == 0:
        return {
            "dataset": dataset_name,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_macro": 0.0,
            "f1_binary": 0.0,
            "pred_real_count": 0,
            "pred_fake_count": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "TP": 0,
            "prob_fake_mean": None,
            "prob_fake_std": None,
            "roc_auc": None,
        }

    tn, fp, fn, tp = safe_confusion_counts(y_true, y_pred)

    return {
        "dataset": dataset_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "pred_real_count": int(np.sum(np.asarray(y_pred) == 0)),
        "pred_fake_count": int(np.sum(np.asarray(y_pred) == 1)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "prob_fake_mean": float(np.mean(y_score)) if len(y_score) > 0 else None,
        "prob_fake_std": float(np.std(y_score)) if len(y_score) > 0 else None,
        "roc_auc": safe_auc(y_true, y_score),
    }


def load_model(args, streams: List[str], device):
    model = build_model(args, streams)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=args.strict)

    if not args.strict:
        if missing:
            print(f"[load] missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[load] unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strict", action="store_true")

    parser.add_argument("--streams", type=str, required=True)
    parser.add_argument("--query-stream", type=str, default="rgb",
                        choices=["rgb", "wavelet", "dct", "semantic"])

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--csv", type=str, default="./result_tri")

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

    args = parser.parse_args()

    os.makedirs(args.csv, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streams = parse_streams(args.streams)
    tag = make_tag(streams)

    print(f"▶ Device: {device}")
    print(f"▶ Streams: {streams}")
    print(f"▶ Checkpoint: {args.checkpoint}")

    clip_processor = None

    if "semantic" in streams:
        clip_processor = CLIPImageProcessor.from_pretrained(args.clip_backbone)

    model = load_model(args, streams, device)

    results = []
    all_true, all_pred, all_score = [], [], []
    auc_true, auc_score = [], []
    auc_excluded_datasets = []

    for ds_name, cfg in TEST_DATASETS.items():
        print(f"\n>>> Evaluating {ds_name}")

        ds_paths = get_dataset_roots(ds_name, cfg)

        rt, rp, rs = evaluate_dataset(
            model=model,
            device=device,
            roots=ds_paths.get("real", []),
            label_value=0,
            streams=streams,
            args=args,
            clip_processor=clip_processor,
        )

        ft, fp, fs = evaluate_dataset(
            model=model,
            device=device,
            roots=ds_paths.get("fake", []),
            label_value=1,
            streams=streams,
            args=args,
            clip_processor=clip_processor,
        )

        y_t = rt + ft
        y_p = rp + fp
        y_s = rs + fs

        if len(y_t) == 0:
            print(f"[{ds_name}] skip: 유효 샘플 없음")
            continue

        y_t = np.asarray(y_t)
        y_p = np.asarray(y_p)
        y_s = np.asarray(y_s, dtype=np.float64)

        finite_mask = np.isfinite(y_s)

        if not finite_mask.all():
            print(
                f"[WARN] {ds_name}: remove non-finite scores: "
                f"{(~finite_mask).sum()} / {len(y_s)}"
            )
            y_t = y_t[finite_mask]
            y_p = y_p[finite_mask]
            y_s = y_s[finite_mask]

        if len(y_t) == 0:
            print(f"[{ds_name}] skip: finite score 없음")
            continue

        m = build_metrics_dict(ds_name, y_t, y_p, y_s)

        auc_str = "None" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
        prob_mean_str = "None" if m["prob_fake_mean"] is None else f"{m['prob_fake_mean']:.4f}"
        prob_std_str = "None" if m["prob_fake_std"] is None else f"{m['prob_fake_std']:.4f}"

        print(
            f"[{ds_name}] "
            f"Acc={m['accuracy']:.4f} "
            f"Prec={m['precision']:.4f} "
            f"Rec={m['recall']:.4f} "
            f"F1-macro={m['f1_macro']:.4f} "
            f"F1-binary={m['f1_binary']:.4f} "
            f"pred_real={m['pred_real_count']} "
            f"pred_fake={m['pred_fake_count']} "
            f"TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']} "
            f"prob_fake_mean={prob_mean_str} "
            f"prob_fake_std={prob_std_str} "
            f"ROC-AUC={auc_str}"
        )

        m["roc_auc_excluding_fake_only"] = None
        m["excluded_from_auc"] = ""
        results.append(m)

        all_true.extend(y_t.tolist())
        all_pred.extend(y_p.tolist())
        all_score.extend(y_s.tolist())

        if len(set(y_t.tolist())) >= 2:
            auc_true.extend(y_t.tolist())
            auc_score.extend(y_s.tolist())
        else:
            auc_excluded_datasets.append(ds_name)

    if len(all_true) > 0:
        all_true_np = np.asarray(all_true)
        all_pred_np = np.asarray(all_pred)
        all_score_np = np.asarray(all_score, dtype=np.float64)

        finite_mask = np.isfinite(all_score_np)

        if not finite_mask.all():
            print(
                f"[WARN] Overall: remove non-finite scores: "
                f"{(~finite_mask).sum()} / {len(all_score_np)}"
            )
            all_true_np = all_true_np[finite_mask]
            all_pred_np = all_pred_np[finite_mask]
            all_score_np = all_score_np[finite_mask]

        if len(all_true_np) > 0:
            overall = build_metrics_dict("Overall", all_true_np, all_pred_np, all_score_np)

            overall_auc_excl_fake_only = safe_auc(auc_true, auc_score)
            overall["roc_auc"] = overall_auc_excl_fake_only
            overall["roc_auc_excluding_fake_only"] = overall_auc_excl_fake_only
            overall["excluded_from_auc"] = ",".join(auc_excluded_datasets)

            overall_auc_str = "None" if overall["roc_auc"] is None else f"{overall['roc_auc']:.4f}"
            overall_prob_mean_str = "None" if overall["prob_fake_mean"] is None else f"{overall['prob_fake_mean']:.4f}"
            overall_prob_std_str = "None" if overall["prob_fake_std"] is None else f"{overall['prob_fake_std']:.4f}"

            print("\n=== Overall Metrics ===")
            print(
                f"Acc={overall['accuracy']:.4f} "
                f"Prec={overall['precision']:.4f} "
                f"Rec={overall['recall']:.4f} "
                f"F1-macro={overall['f1_macro']:.4f} "
                f"F1-binary={overall['f1_binary']:.4f} "
                f"pred_real={overall['pred_real_count']} "
                f"pred_fake={overall['pred_fake_count']} "
                f"TN={overall['TN']} FP={overall['FP']} FN={overall['FN']} TP={overall['TP']} "
                f"prob_fake_mean={overall_prob_mean_str} "
                f"prob_fake_std={overall_prob_std_str} "
                f"ROC-AUC(excl fake-only)={overall_auc_str}"
            )

            results.append(overall)

    csv_path = os.path.join(args.csv, f"tri_{tag}_results.csv")
    pd.DataFrame(
        results,
        columns=[
            "dataset",
            "accuracy",
            "precision",
            "recall",
            "f1_macro",
            "f1_binary",
            "pred_real_count",
            "pred_fake_count",
            "TN",
            "FP",
            "FN",
            "TP",
            "prob_fake_mean",
            "prob_fake_std",
            "roc_auc",
            "roc_auc_excluding_fake_only",
            "excluded_from_auc",
        ],
    ).to_csv(csv_path, index=False)

    print(f"\n▶ Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()