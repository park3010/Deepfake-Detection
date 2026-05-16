# tools/test_video_level.py
import os

GPU_ID = "3"   # 쓰고 싶은 GPU 번호
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from tqdm import tqdm
import sys
import glob
import csv
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from M2TR.utils.build_helper import build_model
from M2TR.utils.checkpoint import load_test_checkpoint
from tools.utils import load_config

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
    "DeepfakeTIMIT": {
        "real": [],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality"
        ],
    },
    "DFD": {
        "real": ["/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences"],
        "fake": ["/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences"],
    },
    "WildDeepfake": {
        "root":   "/home/oem/deepfake/hdd_5TB/WildDeepfake",
        "splits": ["train", "test"],
    },
}

class VideoFrameDataset(Dataset):
    def __init__(self, frame_paths, img_size=320):
        self.frames = frame_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path = self.frames[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        img = np.asarray(img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img)

def collect_roots_for_dataset(ds_name, cfg):
    if ds_name == "WildDeepfake":
        real_roots, fake_roots = [], []
        root = cfg["root"]
        for split in cfg["splits"]:
            sd = os.path.join(root, split)
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
        return {"real": real_roots, "fake": fake_roots}

    elif ds_name == "DeepfakeTIMIT":
        fake_roots = []
        for quality_root in cfg["fake"]:
            if not os.path.isdir(quality_root):
                continue
            for speaker in os.listdir(quality_root):
                sp = os.path.join(quality_root, speaker)
                if os.path.isdir(sp):
                    fake_roots.append(sp)
        return {"real": [], "fake": fake_roots}

    return cfg

def list_frame_paths(video_dir):
    frames = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        frames.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(frames)

def logits_to_fake_prob(logits):
    # M2TR 출력 logits: [B, 2]
    if logits.ndim == 2 and logits.size(1) == 2:
        return F.softmax(logits, dim=1)[:, 1]
    elif logits.ndim == 2 and logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    else:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

@torch.no_grad()
def predict_video(model, device, frame_paths, img_size=320, batch_size=16, num_workers=4, desc=None):
    ds = VideoFrameDataset(frame_paths, img_size=img_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    probs = []
    model.eval()

    for batch in tqdm(loader, desc=desc or "batches", leave=False):
        batch = batch.to(device, non_blocking=True)
        outputs = model({"img": batch})
        p = logits_to_fake_prob(outputs["logits"])
        probs.extend(p.detach().cpu().numpy().tolist())

    return float(np.mean(probs)) if len(probs) > 0 else 0.0

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
            "f1_binary": 0.0,
            "f1_macro": 0.0,
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
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_binary": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "pred_real_count": int(sum(1 for p in y_pred if p == 0)),
        "pred_fake_count": int(sum(1 for p in y_pred if p == 1)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "prob_fake_mean": float(np.mean(y_score)) if len(y_score) > 0 else None,
        "prob_fake_std": float(np.std(y_score)) if len(y_score) > 0 else None,
        "roc_auc": safe_auc(y_true, y_score),
    }

def evaluate_one_label(model, device, roots, label_value, img_size, batch_size, threshold, num_workers, ds_name=""):
    y_true, y_pred, y_score = [], [], []

    for root in tqdm(roots, desc=f"{ds_name}-label{label_value}-roots", leave=False):
        if not os.path.isdir(root):
            print(f"[WARN] missing root: {root}")
            continue

        videos = sorted(os.listdir(root))
        for vid in tqdm(videos, desc=f"{ds_name}-label{label_value}-videos", leave=False):
            video_dir = os.path.join(root, vid)
            if not os.path.isdir(video_dir):
                continue

            frames = list_frame_paths(video_dir)
            if len(frames) == 0:
                continue

            avg_p = predict_video(
                model=model,
                device=device,
                frame_paths=frames,
                img_size=img_size,
                batch_size=batch_size,
                num_workers=num_workers,
                desc=f"{vid}"
            )
            pred = 1 if avg_p >= threshold else 0
            y_true.append(label_value)
            y_pred.append(pred)
            y_score.append(avg_p)

    return y_true, y_pred, y_score

def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_binary": 0.0,
            "f1_macro": 0.0,
        }

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_binary": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="config file name, e.g. m2tr_ffpp_video.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-csv", type=str, default="./video_level_results.csv")
    args = parser.parse_args()

    class Args:
        cfg_file = args.cfg
        shard_id = 0
        base_lr = None

    cfg = load_config(Args)
    cfg["NUM_GPUS"] = 1
    cfg["TEST"]["ENABLE"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    load_test_checkpoint(cfg, model)
    model.eval()

    img_size = cfg["DATASET"]["IMG_SIZE"]

    rows = []
    all_true, all_pred, all_score = [], [], []
    auc_true, auc_score = [], []
    auc_excluded_datasets = []

    for ds_name, ds_cfg in tqdm(TEST_DATASETS.items(), desc="datasets"):
        ds_paths = collect_roots_for_dataset(ds_name, ds_cfg)

        rt, rp, rs = evaluate_one_label(
            model, device, ds_paths.get("real", []), 0,
            img_size, args.batch_size, args.threshold, args.num_workers,
            ds_name=ds_name
        )
        ft, fp, fs = evaluate_one_label(
            model, device, ds_paths.get("fake", []), 1,
            img_size, args.batch_size, args.threshold, args.num_workers,
            ds_name=ds_name
        )

        y_true = rt + ft
        y_pred = rp + fp
        y_score = rs + fs

        m = build_metrics_dict(ds_name, y_true, y_pred, y_score)

        auc_str = "None" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
        prob_mean_str = "None" if m["prob_fake_mean"] is None else f"{m['prob_fake_mean']:.4f}"
        prob_std_str = "None" if m["prob_fake_std"] is None else f"{m['prob_fake_std']:.4f}"

        print(
            f"[{ds_name}] "
            f"Acc={m['accuracy']:.4f}  "
            f"Prec={m['precision']:.4f}  "
            f"Rec={m['recall']:.4f}  "
            f"F1-binary={m['f1_binary']:.4f}  "
            f"F1-macro={m['f1_macro']:.4f}  "
            f"pred_real={m['pred_real_count']}  "
            f"pred_fake={m['pred_fake_count']}  "
            f"TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']}  "
            f"prob_fake_mean={prob_mean_str}  "
            f"prob_fake_std={prob_std_str}  "
            f"ROC-AUC={auc_str}"
        )

        m["roc_auc_excluding_fake_only"] = None
        m["excluded_from_auc"] = ""
        rows.append(m)

        all_true.extend(y_true)
        all_pred.extend(y_pred)
        all_score.extend(y_score)

        if len(set(y_true)) >= 2:
            auc_true.extend(y_true)
            auc_score.extend(y_score)
        else:
            auc_excluded_datasets.append(ds_name)

    overall = build_metrics_dict("Overall", all_true, all_pred, all_score)
    overall_auc_excl_fake_only = safe_auc(auc_true, auc_score)
    overall["roc_auc"] = overall_auc_excl_fake_only
    overall["roc_auc_excluding_fake_only"] = overall_auc_excl_fake_only
    overall["excluded_from_auc"] = ",".join(auc_excluded_datasets)

    overall_auc_str = "None" if overall["roc_auc"] is None else f"{overall['roc_auc']:.4f}"
    overall_prob_mean_str = "None" if overall["prob_fake_mean"] is None else f"{overall['prob_fake_mean']:.4f}"
    overall_prob_std_str = "None" if overall["prob_fake_std"] is None else f"{overall['prob_fake_std']:.4f}"

    print(
        f"[Overall] "
        f"Acc={overall['accuracy']:.4f}  "
        f"Prec={overall['precision']:.4f}  "
        f"Rec={overall['recall']:.4f}  "
        f"F1-binary={overall['f1_binary']:.4f}  "
        f"F1-macro={overall['f1_macro']:.4f}  "
        f"pred_real={overall['pred_real_count']}  "
        f"pred_fake={overall['pred_fake_count']}  "
        f"TN={overall['TN']} FP={overall['FP']} FN={overall['FN']} TP={overall['TP']}  "
        f"prob_fake_mean={overall_prob_mean_str}  "
        f"prob_fake_std={overall_prob_std_str}  "
        f"ROC-AUC(excl fake-only)={overall_auc_str}"
    )
    rows.append(overall)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "accuracy", "precision", "recall", "f1_binary", "f1_macro",
                "pred_real_count", "pred_fake_count",
                "TN", "FP", "FN", "TP",
                "prob_fake_mean", "prob_fake_std",
                "roc_auc", "roc_auc_excluding_fake_only", "excluded_from_auc",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved to: {args.out_csv}")

if __name__ == "__main__":
    main()