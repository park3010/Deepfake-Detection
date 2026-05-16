# main.py
import logging
from pathlib import Path
import re
import torch
from torch.utils.data import DataLoader

from exp_strategy import differ_backbone_exp
import model_abc
import mydataset
import mytransforms
import pipleline

import os
import glob
import gc
import json
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def get_train_loader(t_batch_size=16, v_batch_size=16, num_workers=4, balance=False, aug=None):
    """
    train / val = FF++ only
    """
    if aug is None:
        aug = mytransforms.Transform4()

    train_dataset = mydataset.TrainDataset1(
        mode="train",
        transform=aug,
        normalize=None,
        balanced=balance,
    )

    val_dataset = mydataset.TrainDataset1(
        mode="val",
        transform=mytransforms.DefaultTransform(),
        normalize=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=t_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=v_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, val_loader


def parse_auc(file_name):
    match = re.search(r'val_auc=([0-9]+(?:\.[0-9]+)?)', file_name)
    if match:
        return float(match.group(1))
    raise ValueError(f'Cannot find val_auc in {file_name}')


def get_ckpt(save_dir, backbone_pair, mode='resume'):
    curr_dir = save_dir / f"{backbone_pair[0]}_{backbone_pair[1]}_recw_10"
    checkpoint = curr_dir / 'checkpoints'

    if not curr_dir.exists() or not checkpoint.exists():
        return None

    ckpt_list = list(checkpoint.glob('*.ckpt'))
    if not ckpt_list:
        return None

    if mode == 'resume':
        latest_file = max(ckpt_list, key=lambda x: x.stat().st_ctime)
        logging.info(f'Loading latest ckpt {latest_file}')
        return str(latest_file)

    if mode == 'best':
        best_file = max(ckpt_list, key=lambda x: parse_auc(x.stem))
        logging.info(f'Loading best ckpt {best_file}')
        return str(best_file)

    raise ValueError(f"Invalid mode: {mode}")


def get_test_loaders(batch_size=16, num_workers=4):
    """
    external datasets only
    - Celeb
    - DFD
    - DeepfakeTIMIT
    - WildDeepfake

    mydataset.build_external_test_dict()가
    key별로 분리된 test dict를 반환한다고 가정.
    """
    test_dict = mydataset.build_external_test_dict()

    loaders = []
    dataset_names = []

    for dataset_name, one_dataset_dict in test_dict.items():
        tmp_dataset = mydataset.TestDataset1(
            test_dict={dataset_name: one_dataset_dict},
            transform=mytransforms.DefaultTransform(),
            normalize=None,
        )
        tmp_loader = DataLoader(
            tmp_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(tmp_loader)
        dataset_names.append(dataset_name)

    return loaders, dataset_names

def get_external_video_test_roots():
    ds_roots = {
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
            "fake": [],
        },
        "WildDeepfake": {
            "real": [],
            "fake": [],
        },
    }

    # DeepfakeTIMIT: quality_root 아래 speaker/video 폴더 단위
    for quality_root in [
        "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
        "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality",
    ]:
        if not os.path.isdir(quality_root):
            continue
        for speaker in os.listdir(quality_root):
            sp_path = os.path.join(quality_root, speaker)
            if os.path.isdir(sp_path):
                ds_roots["DeepfakeTIMIT"]["fake"].append(sp_path)

    # WildDeepfake: split/method/{real,fake}
    wild_root = "/home/oem/deepfake/hdd_5TB/WildDeepfake"
    for split in ["train", "test"]:
        split_root = os.path.join(wild_root, split)
        if not os.path.isdir(split_root):
            continue
        for method in os.listdir(split_root):
            base = os.path.join(split_root, method)
            r = os.path.join(base, "real")
            f = os.path.join(base, "fake")
            if os.path.isdir(r):
                ds_roots["WildDeepfake"]["real"].append(r)
            if os.path.isdir(f):
                ds_roots["WildDeepfake"]["fake"].append(f)

    return ds_roots


def safe_confusion_counts(y_true, y_pred):
    """
    항상 TN, FP, FN, TP를 반환.
    한 클래스만 존재하는 경우도 labels=[0,1]로 고정해서 처리.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def safe_auc(y_true, y_score):
    """
    real/fake 두 클래스가 모두 있을 때만 ROC-AUC 계산.
    아니면 None 반환.
    """
    unique_classes = set(y_true)
    if len(unique_classes) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def build_metrics_dict(dataset_name, y_true, y_pred, y_score):
    """
    y_score: fake probability list
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0)

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
    }


def build_model(backbone1='ResNet50', backbone2='EfficientNetB4', lr=1e-4):
    model_strategy = differ_backbone_exp.get_specific_strategy(backbone1, backbone2)

    # strategy hyperparameters
    model_strategy.rec_w = 10.0
    model_strategy.distill_w = 0.1
    model_strategy.cls_w = 1.0
    model_strategy.threshold_bath_idx = 1800

    model_lgn = model_abc.MCLModel(
        strategy=model_strategy,
        using_cls_metric=True,
        lr=lr,
    )
    return model_lgn, model_strategy

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frame_paths, transform=None, normalize=None):
        self.frame_paths = frame_paths
        self.transform = transform if transform is not None else mytransforms.DefaultTransform()
        self.normalize = normalize

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img_path = self.frame_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.normalize is not None:
            img = self.normalize(img)

        return img

def train_ResNet50_EfficientNetB4():
    backbone1 = 'ResNet50'
    backbone2 = 'EfficientNetB4'

    save_dir = Path(f"/home/oem/deepfake/Ourmethod/comparison/_ckpt/idcnet/{backbone1}_{backbone2}_recw_10")
    save_dir.mkdir(exist_ok=True, parents=True)

    resume_ckpt = get_ckpt(save_dir.parent, [backbone1, backbone2], mode='resume')

    model_lgn, _ = build_model(backbone1, backbone2, lr=3e-4)

    train_loader, val_loader = get_train_loader(
        t_batch_size=16,
        v_batch_size=16,
        num_workers=5,
        balance=True,
        aug=mytransforms.Transform4(),   # 가장 단순한 train transform
    )

    ckpt_name = "epoch={epoch:02d}-val_acc={val_acc:.4f}-val_auc={val_auc:.4f}"

    pipleline.train_model(
        seed=42,
        model=model_lgn,
        save_dir=save_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        num_device=1,              # DDP 제거
        val_check_interval=1.0,    # epoch마다 validation
        resume_ckpt=resume_ckpt,
        epochs=10,
        monitor="val_auc",
        mode="max",
        save_top_k=10,
        ckpt_name=ckpt_name,
        limit_val_batches=1.0,
    )


def test_ResNet50_EfficientNetB4():
    backbone_pair = ['ResNet50', 'EfficientNetB4']
    save_root = Path("/home/oem/deepfake/Ourmethod/comparison/_ckpt/idcnet")
    ckpt_path = get_ckpt(save_root, backbone_pair, mode='best')

    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found. Train first.")

    model_lgn, _ = build_model(backbone_pair[0], backbone_pair[1], lr=3e-4)
    model_lgn.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_lgn = model_lgn.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model_lgn.load_state_dict(state_dict, strict=True)

    ds_roots = get_external_video_test_roots()

    result_save = Path("/home/oem/deepfake/Ourmethod/comparison/_result/idcnet_video_level2") / Path(ckpt_path).stem
    result_save.mkdir(exist_ok=True, parents=True)

    results = []

    all_true, all_pred, all_score = [], [], []
    auc_true, auc_score = [], []   # overall AUC 계산용 (fake-only dataset 제외)

    auc_excluded_datasets = []

    for ds_name, paths in ds_roots.items():
        print(f"\n>>> Evaluating {ds_name}")

        rt, rp, rs = evaluate_video_level_dataset(
            model_lgn=model_lgn,
            device=device,
            roots=paths.get("real", []),
            label_value=0,
            batch_size=30,
            num_workers=4,
            threshold=0.5,
        )

        ft, fp, fs = evaluate_video_level_dataset(
            model_lgn=model_lgn,
            device=device,
            roots=paths.get("fake", []),
            label_value=1,
            batch_size=30,
            num_workers=4,
            threshold=0.5,
        )

        y_t = rt + ft
        y_p = rp + fp
        y_s = rs + fs   # fake probability

        if len(y_t) == 0:
            print(f"[{ds_name}] (skip) no valid samples")
            continue

        metrics = build_metrics_dict(ds_name, y_t, y_p, y_s)
        auc_str = "None" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}"

        print(
            f"[{ds_name}] "
            f"Acc={metrics['accuracy']:.4f}  "
            f"Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  "
            f"F1-macro={metrics['f1_macro']:.4f}  "
            f"F1-binary={metrics['f1_binary']:.4f}  "
            f"pred_real={metrics['pred_real_count']}  "
            f"pred_fake={metrics['pred_fake_count']}  "
            f"TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']} TP={metrics['TP']}  "
            f"prob_fake_mean={metrics['prob_fake_mean']:.4f}  "
            f"prob_fake_std={metrics['prob_fake_std']:.4f}  "
            f"ROC-AUC={auc_str}"
        )

        with open(result_save / f"{ds_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        results.append(metrics)

        # overall metric용: 전체 데이터셋 포함
        all_true.extend(y_t)
        all_pred.extend(y_p)
        all_score.extend(y_s)

        # overall AUC용: 클래스가 2개 모두 있는 데이터셋만 포함
        if len(set(y_t)) >= 2:
            auc_true.extend(y_t)
            auc_score.extend(y_s)
        else:
            auc_excluded_datasets.append(ds_name)

    if len(all_true) > 0:
        overall = build_metrics_dict("Overall", all_true, all_pred, all_score)

        # overall AUC는 fake-only dataset 제외
        overall_auc_excl_fake_only = safe_auc(auc_true, auc_score)
        overall["roc_auc"] = overall_auc_excl_fake_only
        overall["roc_auc_excluding_fake_only"] = overall_auc_excl_fake_only
        overall["excluded_from_auc"] = auc_excluded_datasets
        overall_auc_str = "None" if overall["roc_auc"] is None else f"{overall['roc_auc']:.4f}"

        print("\n=== Overall Metrics ===")
        print(
            f"Acc={overall['accuracy']:.4f}  "
            f"Prec={overall['precision']:.4f}  "
            f"Rec={overall['recall']:.4f}  "
            f"F1-macro={overall['f1_macro']:.4f}  "
            f"F1-binary={overall['f1_binary']:.4f}  "
            f"pred_real={overall['pred_real_count']}  "
            f"pred_fake={overall['pred_fake_count']}  "
            f"TN={overall['TN']} FP={overall['FP']} FN={overall['FN']} TP={overall['TP']}  "
            f"prob_fake_mean={overall['prob_fake_mean']:.4f}  "
            f"prob_fake_std={overall['prob_fake_std']:.4f}  "
            f"ROC-AUC(excl fake-only)={overall_auc_str}"
        )

        with open(result_save / "Overall_metrics.json", "w") as f:
            json.dump(overall, f, indent=4)

        with open(result_save / "all_results_summary.json", "w") as f:
            json.dump(results + [overall], f, indent=4)


def evaluate_video_level_dataset(
    model_lgn,
    device,
    roots,
    label_value,
    batch_size=30,
    num_workers=4,
    threshold=0.5,
):
    y_true, y_pred, y_score = [], [], []

    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] path not found: {root}")
            continue

        # root 아래의 각 하위 폴더를 1개 비디오로 간주
        video_dirs = sorted([
            os.path.join(root, d)
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        for vid_dir in video_dirs:
            frame_paths = []
            frame_paths += sorted(glob.glob(os.path.join(vid_dir, "*.png")))
            frame_paths += sorted(glob.glob(os.path.join(vid_dir, "*.jpg")))
            frame_paths += sorted(glob.glob(os.path.join(vid_dir, "*.jpeg")))
            frame_paths += sorted(glob.glob(os.path.join(vid_dir, "*.bmp")))

            if len(frame_paths) == 0:
                continue

            ds = VideoFrameDataset(
                frame_paths=frame_paths,
                transform=mytransforms.DefaultTransform(),
                normalize=None,
            )

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            sum_prob = 0.0
            count = 0

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    model_lgn.strategy.device = device

                    dummy_y = torch.full(
                        (batch.size(0),),
                        fill_value=label_value,
                        dtype=torch.long,
                        device=device,
                    )

                    model_out = model_lgn.strategy.test_step((batch, dummy_y))
                    prob_fake = model_out.y_pred[:, 1]

                    sum_prob += float(prob_fake.sum().item())
                    count += int(prob_fake.numel())

            if count == 0:
                continue

            avg_prob = sum_prob / count
            pred = 1 if avg_prob >= threshold else 0

            y_true.append(label_value)
            y_pred.append(pred)
            y_score.append(float(avg_prob))

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    return y_true, y_pred, y_score


if __name__ == '__main__':
    # 1) train
    # train_ResNet50_EfficientNetB4()

    # 2) test on external datasets
    test_ResNet50_EfficientNetB4()