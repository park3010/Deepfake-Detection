# UCF_eval.py
"""
Evaluate trained CNN on external datasets with VIDEO-LEVEL metrics.
- binary classification: real vs fake
- For each video:
    1) collect all frames
    2) run frame-level prediction
    3) average fake probability
    4) threshold to get one video prediction
- Then compute dataset-wise video-level metrics
"""

import os

GPU_ID = "2"   # 쓰고 싶은 GPU 번호
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


import glob
import math
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

CLASSES = ["real", "fake"]
BATCH_SIZE = 32
TARGET_SIZE = (299, 299)
THRESHOLD = 0.5

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
        "real": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences"
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences"
        ],
    },
    "WildDeepfake": {
        "root": "/home/oem/deepfake/hdd_5TB/WildDeepfake",
        "splits": ["train", "test"],
    },
}


def collect_roots_for_dataset(ds_name, cfg):
    """
    데이터셋별 real/fake 폴더 경로 리스트 구성
    """
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

    else:
        return cfg


def collect_video_dirs_from_roots(roots):
    """
    root 아래의 video 폴더들을 모아서 반환
    각 video 폴더 안에 png/jpg 프레임들이 있다고 가정
    """
    video_dirs = []

    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] Missing root: {root}")
            continue

        for vid in sorted(os.listdir(root)):
            vid_dir = os.path.join(root, vid)
            if os.path.isdir(vid_dir):
                video_dirs.append(vid_dir)

    return video_dirs


def collect_frame_paths_for_video(video_dir):
    frames = sorted(glob.glob(os.path.join(video_dir, "*.png")))
    frames += sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    frames += sorted(glob.glob(os.path.join(video_dir, "*.jpeg")))
    return frames


def build_frame_dataframe(frame_paths, class_name="fake"):
    """
    Keras flow_from_dataframe용 frame dataframe
    y_col은 predict에서는 안 써도 되지만 형식을 맞추기 위해 유지
    """
    rows = []
    for p in frame_paths:
        rows.append({
            "filename": p,
            "class": class_name
        })
    return pd.DataFrame(rows)


def build_frame_generator(frame_df):
    if len(frame_df) == 0:
        raise ValueError("Frame dataframe is empty.")

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_dataframe(
        dataframe=frame_df,
        x_col='filename',
        y_col='class',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        class_mode='categorical',
        shuffle=False
    )
    return generator


def predict_one_video(model, video_dir, threshold=0.5):
    """
    비디오 폴더 안의 모든 프레임을 사용해
    fake 확률 평균으로 video-level prediction 생성
    """
    frame_paths = collect_frame_paths_for_video(video_dir)
    if len(frame_paths) == 0:
        return None

    frame_df = build_frame_dataframe(frame_paths, class_name="fake")
    frame_gen = build_frame_generator(frame_df)

    steps = max(1, math.ceil(frame_gen.samples / float(frame_gen.batch_size)))
    probs = model.predict(frame_gen, steps=steps, verbose=0)

    # softmax output: [P(real), P(fake)]
    fake_probs = probs[:, 1]
    avg_fake_prob = float(np.mean(fake_probs))

    pred_label = 1 if avg_fake_prob >= threshold else 0

    return {
        "video_dir": video_dir,
        "n_frames": len(frame_paths),
        "avg_fake_prob": avg_fake_prob,
        "pred_label": pred_label
    }


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
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_binary = f1_score(y_true, y_pred, average='binary', zero_division=0)

    tn, fp, fn, tp = safe_confusion_counts(y_true, y_pred)

    pred_real_count = int(sum(1 for p in y_pred if p == 0))
    pred_fake_count = int(sum(1 for p in y_pred if p == 1))

    prob_fake_mean = float(np.mean(y_score)) if len(y_score) > 0 else None
    prob_fake_std = float(np.std(y_score)) if len(y_score) > 0 else None
    roc_auc = safe_auc(y_true, y_score)

    return {
        "dataset": dataset_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_macro": float(f1_macro),
        "f1_binary": float(f1_binary),
        "pred_real_count": pred_real_count,
        "pred_fake_count": pred_fake_count,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "prob_fake_mean": prob_fake_mean,
        "prob_fake_std": prob_fake_std,
        "roc_auc": roc_auc,
        "n_videos": len(y_true),
    }


def evaluate_dataset_video_level(model, ds_name, cfg, threshold=0.5):
    cfg2 = collect_roots_for_dataset(ds_name, cfg)

    real_video_dirs = collect_video_dirs_from_roots(cfg2.get("real", []))
    fake_video_dirs = collect_video_dirs_from_roots(cfg2.get("fake", []))

    print("=" * 80)
    print(f"Dataset: {ds_name}")
    print(f"Real videos: {len(real_video_dirs)}")
    print(f"Fake videos: {len(fake_video_dirs)}")

    y_true, y_pred, y_score = [], [], []
    video_rows = []

    # real videos
    for vid_dir in real_video_dirs:
        result = predict_one_video(model, vid_dir, threshold=threshold)
        if result is None:
            continue

        y_true.append(0)
        y_pred.append(result["pred_label"])
        y_score.append(result["avg_fake_prob"])

        video_rows.append({
            "dataset": ds_name,
            "video_dir": vid_dir,
            "true_label": 0,
            "pred_label": result["pred_label"],
            "n_frames": result["n_frames"],
            "avg_fake_prob": result["avg_fake_prob"]
        })

    # fake videos
    for vid_dir in fake_video_dirs:
        result = predict_one_video(model, vid_dir, threshold=threshold)
        if result is None:
            continue

        y_true.append(1)
        y_pred.append(result["pred_label"])
        y_score.append(result["avg_fake_prob"])

        video_rows.append({
            "dataset": ds_name,
            "video_dir": vid_dir,
            "true_label": 1,
            "pred_label": result["pred_label"],
            "n_frames": result["n_frames"],
            "avg_fake_prob": result["avg_fake_prob"]
        })

    if len(y_true) == 0:
        print("[SKIP] No valid videos found.")
        return None, None

    metrics_row = build_metrics_dict(ds_name, y_true, y_pred, y_score)

    auc_str = "None" if metrics_row["roc_auc"] is None else f"{metrics_row['roc_auc']:.4f}"
    prob_mean_str = "None" if metrics_row["prob_fake_mean"] is None else f"{metrics_row['prob_fake_mean']:.4f}"
    prob_std_str = "None" if metrics_row["prob_fake_std"] is None else f"{metrics_row['prob_fake_std']:.4f}"

    print(
        f"[{ds_name}] "
        f"Acc={metrics_row['accuracy']:.4f}  "
        f"Prec={metrics_row['precision']:.4f}  "
        f"Rec={metrics_row['recall']:.4f}  "
        f"F1-macro={metrics_row['f1_macro']:.4f}  "
        f"F1-binary={metrics_row['f1_binary']:.4f}  "
        f"pred_real={metrics_row['pred_real_count']}  "
        f"pred_fake={metrics_row['pred_fake_count']}  "
        f"TN={metrics_row['TN']} FP={metrics_row['FP']} FN={metrics_row['FN']} TP={metrics_row['TP']}  "
        f"prob_fake_mean={prob_mean_str}  "
        f"prob_fake_std={prob_std_str}  "
        f"ROC-AUC={auc_str}"
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0))

    return metrics_row, pd.DataFrame(video_rows)


def main(weights_file='/home/oem/deepfake/Ourmethod/comparison/_ckpt/ucf/train/inception.029-0.0368.hdf5',
         threshold=THRESHOLD,
         save_dir='/home/oem/deepfake/Ourmethod/comparison/_result/ucf2'):
    os.makedirs(save_dir, exist_ok=True)

    model = load_model(weights_file)

    metrics_rows = []
    all_true, all_pred, all_score = [], [], []
    auc_true, auc_score = [], []
    auc_excluded_datasets = []
    per_video_dfs = []

    for ds_name, cfg in TEST_DATASETS.items():
        metrics_row, video_df = evaluate_dataset_video_level(model, ds_name, cfg, threshold=threshold)

        if metrics_row is not None:
            metrics_row["roc_auc_excluding_fake_only"] = None
            metrics_row["excluded_from_auc"] = ""

            metrics_rows.append(metrics_row)
            per_video_dfs.append(video_df)

            ds_true = video_df["true_label"].tolist()
            ds_pred = video_df["pred_label"].tolist()
            ds_score = video_df["avg_fake_prob"].tolist()

            all_true.extend(ds_true)
            all_pred.extend(ds_pred)
            all_score.extend(ds_score)

            if len(set(ds_true)) >= 2:
                auc_true.extend(ds_true)
                auc_score.extend(ds_score)
            else:
                auc_excluded_datasets.append(ds_name)

    # overall metrics
    if len(all_true) > 0:
        overall = build_metrics_dict("Overall", all_true, all_pred, all_score)

        overall_auc_excl_fake_only = safe_auc(auc_true, auc_score)
        overall["roc_auc"] = overall_auc_excl_fake_only
        overall["roc_auc_excluding_fake_only"] = overall_auc_excl_fake_only
        overall["excluded_from_auc"] = ",".join(auc_excluded_datasets)

        overall_auc_str = "None" if overall["roc_auc"] is None else f"{overall['roc_auc']:.4f}"
        overall_prob_mean_str = "None" if overall["prob_fake_mean"] is None else f"{overall['prob_fake_mean']:.4f}"
        overall_prob_std_str = "None" if overall["prob_fake_std"] is None else f"{overall['prob_fake_std']:.4f}"

        print("=" * 80)
        print("Overall Video-Level Metrics")
        print(
            f"Acc={overall['accuracy']:.4f}  "
            f"Prec={overall['precision']:.4f}  "
            f"Rec={overall['recall']:.4f}  "
            f"F1-macro={overall['f1_macro']:.4f}  "
            f"F1-binary={overall['f1_binary']:.4f}  "
            f"pred_real={overall['pred_real_count']}  "
            f"pred_fake={overall['pred_fake_count']}  "
            f"TN={overall['TN']} FP={overall['FP']} FN={overall['FN']} TP={overall['TP']}  "
            f"prob_fake_mean={overall_prob_mean_str}  "
            f"prob_fake_std={overall_prob_std_str}  "
            f"ROC-AUC(excl fake-only)={overall_auc_str}"
        )

        metrics_rows.append(overall)

    # save csv
    if len(metrics_rows) > 0:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = os.path.join(save_dir, "video_level_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"[Saved] {metrics_csv}")

    if len(per_video_dfs) > 0:
        per_video_df = pd.concat(per_video_dfs, ignore_index=True)
        per_video_csv = os.path.join(save_dir, "video_level_predictions.csv")
        per_video_df.to_csv(per_video_csv, index=False)
        print(f"[Saved] {per_video_csv}")

if __name__ == '__main__':
    main()