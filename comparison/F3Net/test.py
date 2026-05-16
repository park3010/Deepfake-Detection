# f3net_test.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F3Net 전용 다중 데이터셋 평가 스크립트
- 입력: 각 데이터셋별 real/fake 폴더에 저장된 프레임 이미지(이미 존재하는 구조 재사용)
- 모델: F3Net (yyk-wew/F3Net 레포 기준, models.py 내 F3Net 클래스)
- 출력: 데이터셋별/전체 Acc/Prec/Rec/F1 (터미널 출력 + CSV 저장)

기본 가정
- 이진 분류 (0=real, 1=fake)
- 로짓이 2차원(배치,2)이면 softmax 후 idx=1을 fake 확률로 사용
- 로짓이 1차원(배치,1)이면 sigmoid 후 그 값을 fake 확률로 사용
- 입력 크기 기본 299 (Xception 사전학습 호환)
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# -----------------------------
# 데이터셋 루트 (현 스크립트가 쓰던 구조 그대로)
# -----------------------------
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


# -----------------------------
# Dataset
# -----------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, frame_paths, transform):
        self.frames = frame_paths
        self.tf = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path = self.frames[idx]
        img = Image.open(path).convert('RGB')
        return self.tf(img)


# -----------------------------
# 유틸
# -----------------------------
def to_fake_prob_from_any(output, pos_index=1):
    """
    model(output)이 텐서가 아닐 수도(예: tuple/list) 있으므로 안전하게 fake 확률로 변환.
    - 우선 (B,2) → softmax[:, pos_index]
    - 없으면 (B,1) → sigmoid
    - 그래도 없으면 첫 텐서로 시도
    """
    import torch
    if isinstance(output, (tuple, list)):
        tensors = [x for x in output if torch.is_tensor(x)]
        cand = next((x for x in tensors if x.ndim == 2 and x.size(1) == 2), None)
        if cand is None:
            cand = next((x for x in tensors if x.ndim == 2 and x.size(1) == 1), None)
        if cand is None and tensors:
            cand = tensors[0]
        output = cand if cand is not None else output[0]

    return logits_to_fake_prob(output, pos_index=pos_index)


def logits_to_fake_prob(logits, pos_index=1):
    """
    logits shape:
      - (B, 2): softmax[:, pos_index]
      - (B, 1): sigmoid
    """
    if logits.dim() == 2 and logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, pos_index]
    elif logits.dim() == 2 and logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    elif logits.dim() == 1:
        return torch.sigmoid(logits)
    else:
        # 예외적으로 다른 형태면 softmax 후 마지막 채널을 pos로 가정
        return torch.softmax(logits, dim=-1)[..., -1]


def find_f3net_class(args):
    """
    F3Net 클래스를 가져온다.
    --f3net-root 로 레포 경로(models.py가 있는 디렉토리 또는 그 부모)를 받을 수 있다.
    """
    if args.f3net_root:
        root = os.path.abspath(args.f3net_root)

        # 1) 사용자가 부모 경로를 준 경우
        if os.path.isdir(os.path.join(root, "F3Net")):
            parent_dir = root
        # 2) 사용자가 F3Net 폴더 자체를 준 경우
        elif os.path.basename(root) == "F3Net":
            parent_dir = os.path.dirname(root)
        else:
            parent_dir = root

        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

    try:
        from F3Net.models import F3Net as F3NetModel
        return F3NetModel
    except Exception as e1:
        try:
            from models import F3Net as F3NetModel
            return F3NetModel
        except Exception as e2:
            raise ImportError(
                f"F3Net 클래스를 불러오지 못했습니다. --f3net-root 경로를 확인하세요.\n"
                f"원인1: {e1}\n원인2: {e2}"
            )


def build_f3net(args, device):
    F3NetModel = find_f3net_class(args)
    # 다양한 ctor 시그니처 대응
    model = None
    tried = []
    for ctor in [
        lambda: F3NetModel(mode=args.f3net_mode, num_classes=2),
        lambda: F3NetModel(args.f3net_mode, num_classes=2),
        lambda: F3NetModel(num_classes=2),
        lambda: F3NetModel()
    ]:
        try:
            model = ctor().to(device)
            break
        except Exception as e:
            tried.append(str(e))
            continue
    if model is None:
        raise RuntimeError("F3Net 인스턴스 생성 실패. 시도한 형태들 오류:\n- " + "\n- ".join(tried))
    return model


def clean_state_dict(state_dict):
    """
    DataParallel/Lightning 등 접두사 정리
    """
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_sd[k] = v
    return new_sd


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        # 여러 케이스 대응
        state_dict = ckpt.get("state_dict", None)
        if state_dict is None:
            state_dict = ckpt.get("model_state", None)
        if state_dict is None:
            # 그냥 state_dict가 바깥 dict일 수도 있음
            # (키가 텐서인 형태인지 검사)
            if all(isinstance(k, str) for k in ckpt.keys()):
                state_dict = ckpt
            else:
                raise RuntimeError("체크포인트에서 state_dict를 찾을 수 없습니다.")
    else:
        state_dict = ckpt

    state_dict = clean_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")


def collect_roots_for_dataset(ds_name, cfg):
    """
    데이터셋별 real/fake 폴더 경로 리스트 구성(WildDeepfake, DeepfakeTIMIT 특수 처리)
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
                r, f = os.path.join(base, "real"), os.path.join(base, "fake")
                if os.path.isdir(r): real_roots.append(r)
                if os.path.isdir(f): fake_roots.append(f)
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
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1b = f1_score(y_true, y_pred, average="binary", zero_division=0)
    else:
        acc = prec = rec = f1m = f1b = 0.0

    tn, fp, fn, tp = safe_confusion_counts(y_true, y_pred) if y_true else (0, 0, 0, 0)

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
        "f1_macro": float(f1m),
        "f1_binary": float(f1b),
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


def evaluate_dataset(model, device, transform, roots, label_value,
                     batch_size=4, threshold=0.5, pos_index=1, num_workers=0):
    """
    폴더 구조:
      root/
        video_aaa/
          000001.jpg ...
        video_bbb/
          ...
    비디오 단위로 프레임 확률 평균 → 비디오 예측값 → 지표 계산
    """
    y_true, y_pred, y_score = [], [], []

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

            ds = VideoFrameDataset(frames, transform)
            loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
            probs = []

            for batch in tqdm(loader, desc=f" frames of {vid}", leave=False):
                batch = batch.to(device, non_blocking=True)
                with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
                    logits = model(batch)
                    p = to_fake_prob_from_any(logits, pos_index=pos_index)
                probs.append(p.detach().cpu().numpy())

            avg_p = float(np.concatenate(probs).mean())
            pred = 1 if avg_p >= threshold else 0

            y_true.append(label_value)
            y_pred.append(pred)
            y_score.append(avg_p)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return y_true, y_pred, y_score


def main():
    parser = argparse.ArgumentParser(description="F3Net 전용 평가 스크립트")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", required=True, help=".pt/.pth 경로 (F3Net 가중치)")
    parser.add_argument("--f3net-root", type=str, default=None,
                        help="F3Net 레포 루트(models.py가 있는 경로). pip 설치가 아니라면 지정")
    parser.add_argument("--f3net-mode", type=str, default="Both",
                        choices=["FAD", "LFS", "Both"],
                        help="F3Net 분기/모드 (기본 Both)")
    parser.add_argument("--img-size", type=int, default=299, help="입력 해상도(기본 299)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5, help="fake 판정 임계값(비디오 평균 확률)")
    parser.add_argument("--pos-index", type=int, default=1,
                        help="(로짓이 2개일 때) fake 클래스 인덱스(기본 1)")
    parser.add_argument("--csv", type=str, default="/home/sujin/psj2003/deepfake/code/result",
                        help="결과 CSV 저장 디렉터리")
    args = parser.parse_args()

    os.makedirs(args.csv, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    # 모델 준비
    model = build_f3net(args, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    print(f"▶ Loaded F3Net from {args.checkpoint}")
    print(f"Params = {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 변환 (RGB + ImageNet 정규화)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    results = []
    all_true, all_pred, all_score = [], [], []
    auc_true, auc_score = [], []
    auc_excluded_datasets = []

    # 데이터셋 루프
    for ds_name, cfg in TEST_DATASETS.items():
        ds_paths = collect_roots_for_dataset(ds_name, cfg)

        print(f"\n>>> Evaluating {ds_name}")
        rt, rp, rs = evaluate_dataset(
            model, device, transform,
            ds_paths.get("real", []), 0,
            args.batch_size, args.threshold,
            args.pos_index, args.num_workers
        )
        ft, fp, fs = evaluate_dataset(
            model, device, transform,
            ds_paths.get("fake", []), 1,
            args.batch_size, args.threshold,
            args.pos_index, args.num_workers
        )

        y_t, y_p, y_s = rt + ft, rp + fp, rs + fs

        metrics = build_metrics_dict(ds_name, y_t, y_p, y_s)

        auc_str = "None" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}"
        prob_mean_str = "None" if metrics["prob_fake_mean"] is None else f"{metrics['prob_fake_mean']:.4f}"
        prob_std_str = "None" if metrics["prob_fake_std"] is None else f"{metrics['prob_fake_std']:.4f}"

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
            f"prob_fake_mean={prob_mean_str}  "
            f"prob_fake_std={prob_std_str}  "
            f"ROC-AUC={auc_str}"
        )
        
        metrics["roc_auc_excluding_fake_only"] = None
        metrics["excluded_from_auc"] = ""

        results.append(metrics)

        all_true.extend(y_t)
        all_pred.extend(y_p)
        all_score.extend(y_s)

        if len(set(y_t)) >= 2:
            auc_true.extend(y_t)
            auc_score.extend(y_s)
        else:
            auc_excluded_datasets.append(ds_name)

    # 전체 통합
    if all_true:
        overall = build_metrics_dict("Overall", all_true, all_pred, all_score)

        overall_auc_excl_fake_only = safe_auc(auc_true, auc_score)
        overall["roc_auc"] = overall_auc_excl_fake_only
        overall["roc_auc_excluding_fake_only"] = overall_auc_excl_fake_only
        overall["excluded_from_auc"] = ",".join(auc_excluded_datasets)

        overall_auc_str = "None" if overall["roc_auc"] is None else f"{overall['roc_auc']:.4f}"
        overall_prob_mean_str = "None" if overall["prob_fake_mean"] is None else f"{overall['prob_fake_mean']:.4f}"
        overall_prob_std_str = "None" if overall["prob_fake_std"] is None else f"{overall['prob_fake_std']:.4f}"

        print("\n=== Overall Metrics ===")
        print(
            f"Acc={overall['accuracy']:.4f}  "
            f"Prec={overall['precision']:.4f}  "
            f"Rec={overall['recall']:.4f}  "
            f"F1-Macro={overall['f1_macro']:.4f}  "
            f"F1-Binary={overall['f1_binary']:.4f}  "
            f"pred_real={overall['pred_real_count']}  "
            f"pred_fake={overall['pred_fake_count']}  "
            f"TN={overall['TN']} FP={overall['FP']} FN={overall['FN']} TP={overall['TP']}  "
            f"prob_fake_mean={overall_prob_mean_str}  "
            f"prob_fake_std={overall_prob_std_str}  "
            f"ROC-AUC(excl fake-only)={overall_auc_str}"
        )
        results.append(overall)

    # CSV 저장
    out_name = f"f3net_{args.f3net_mode.lower()}_rgb_results.csv"
    out_path = os.path.join(args.csv, out_name)

    columns = [
        "dataset",
        "accuracy", "precision", "recall", "f1_macro", "f1_binary",
        "pred_real_count", "pred_fake_count",
        "TN", "FP", "FN", "TP",
        "prob_fake_mean", "prob_fake_std",
        "roc_auc",
        "roc_auc_excluding_fake_only",
        "excluded_from_auc",
    ]

    pd.DataFrame(results).to_csv(out_path, index=False, columns=columns)
    print(f"\n▶ Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
