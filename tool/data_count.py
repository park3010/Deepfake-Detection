#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import defaultdict
from pathlib import Path
import random

# =========================================================
# train_rgb.py 기준 FF++ 디렉터리 매핑
# =========================================================
TRAIN_DATASETS = {
    "original": "original_sequences/youtube",
    "DeepFakeDetection_original": "original_sequences/actors",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "DeepFakeDetection": "manipulated_sequences/DeepFakeDetection",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceShifter": "manipulated_sequences/FaceShifter",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}

# =========================================================
# test_rgb.py 기준 테스트 데이터셋 경로
# =========================================================
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


IMG_EXTS = (".png", ".jpg", ".jpeg")


def count_images_in_dir(dir_path: str) -> int:
    """디렉터리 내부(하위 포함)의 이미지 개수"""
    cnt = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                cnt += 1
    return cnt


def count_frames_in_video_dir(video_dir: str) -> int:
    """비디오 폴더 바로 아래 프레임 이미지 개수"""
    if not os.path.isdir(video_dir):
        return 0
    cnt = 0
    for f in os.listdir(video_dir):
        p = os.path.join(video_dir, f)
        if os.path.isfile(p) and f.lower().endswith(IMG_EXTS):
            cnt += 1
    return cnt


# =========================================================
# 1) train_rgb.py 구조 집계
# =========================================================
def collect_train_ffpp_stats(root_dir: str, compression: str = "raw", seed: int = 42):
    """
    train_rgb.py와 동일하게:
    root_dir / DATASETS[key] / compression / mtcnn
    아래의 모든 이미지를 수집.

    추가로 '비디오 수'는 mtcnn 하위의 '프레임을 직접 담고 있는 폴더' 개수로 계산.
    """
    rows = []
    all_samples = []

    for key, rel_path in TRAIN_DATASETS.items():
        base = os.path.join(root_dir, rel_path, compression, "mtcnn")
        label = 0 if "original" in key else 1

        if not os.path.isdir(base):
            rows.append({
                "dataset": key,
                "label": label,
                "base_path": base,
                "video_count": 0,
                "frame_count": 0,
            })
            continue

        video_count = 0
        frame_count = 0

        # train_rgb.py는 os.walk로 모든 이미지 파일을 모음
        # 여기서는 "이미지가 하나라도 들어있는 폴더"를 video 폴더로 간주
        for sub, _, files in os.walk(base):
            img_files = [f for f in files if f.lower().endswith(IMG_EXTS)]
            if img_files:
                video_count += 1
                frame_count += len(img_files)
                for f in img_files:
                    all_samples.append((os.path.join(sub, f), label, key))

        rows.append({
            "dataset": key,
            "label": label,
            "base_path": base,
            "video_count": video_count,
            "frame_count": frame_count,
        })

    # train_rgb.py와 동일하게 프레임 단위 8:2 split 재현
    total_frames = len(all_samples)
    tr_len = int(0.8 * total_frames)
    va_len = total_frames - tr_len

    rng = random.Random(seed)
    indices = list(range(total_frames))
    rng.shuffle(indices)

    train_indices = set(indices[:tr_len])
    val_indices = set(indices[tr_len:])

    split_stats = defaultdict(lambda: {
        "train_frames": 0,
        "val_frames": 0,
    })

    for idx, (_, _, key) in enumerate(all_samples):
        if idx in train_indices:
            split_stats[key]["train_frames"] += 1
        else:
            split_stats[key]["val_frames"] += 1

    return rows, split_stats, total_frames, tr_len, va_len


# =========================================================
# 2) test_rgb.py 구조 집계
# =========================================================
def collect_test_stats():
    rows = []

    for ds_name, cfg in TEST_DATASETS.items():
        if ds_name == "WildDeepfake":
            # test_rgb.py와 동일한 방식
            real_video_count = 0
            fake_video_count = 0
            real_frame_count = 0
            fake_frame_count = 0

            for split in cfg["splits"]:
                sd = os.path.join(cfg["root"], split)
                if not os.path.isdir(sd):
                    continue

                for m in os.listdir(sd):
                    base = os.path.join(sd, m)
                    real_dir = os.path.join(base, "real")
                    fake_dir = os.path.join(base, "fake")

                    if os.path.isdir(real_dir):
                        for vid in os.listdir(real_dir):
                            vid_dir = os.path.join(real_dir, vid)
                            if os.path.isdir(vid_dir):
                                frames = count_frames_in_video_dir(vid_dir)
                                if frames > 0:
                                    real_video_count += 1
                                    real_frame_count += frames

                    if os.path.isdir(fake_dir):
                        for vid in os.listdir(fake_dir):
                            vid_dir = os.path.join(fake_dir, vid)
                            if os.path.isdir(vid_dir):
                                frames = count_frames_in_video_dir(vid_dir)
                                if frames > 0:
                                    fake_video_count += 1
                                    fake_frame_count += frames

            rows.append({
                "dataset": ds_name,
                "split_or_label": "real",
                "video_count": real_video_count,
                "frame_count": real_frame_count,
            })
            rows.append({
                "dataset": ds_name,
                "split_or_label": "fake",
                "video_count": fake_video_count,
                "frame_count": fake_frame_count,
            })

        elif ds_name == "DeepfakeTIMIT":
            # test_rgb.py 기준: higher_quality / lower_quality 아래 speaker 폴더를 roots로 사용
            # 그리고 evaluate_dataset()에서 speaker 아래의 하위 폴더들을 비디오로 순회함
            fake_video_count = 0
            fake_frame_count = 0

            for quality_root in cfg["fake"]:
                if not os.path.isdir(quality_root):
                    continue

                for speaker in os.listdir(quality_root):
                    sp_path = os.path.join(quality_root, speaker)
                    if not os.path.isdir(sp_path):
                        continue

                    for vid in os.listdir(sp_path):
                        vid_dir = os.path.join(sp_path, vid)
                        if os.path.isdir(vid_dir):
                            frames = count_frames_in_video_dir(vid_dir)
                            if frames > 0:
                                fake_video_count += 1
                                fake_frame_count += frames

            rows.append({
                "dataset": ds_name,
                "split_or_label": "real",
                "video_count": 0,
                "frame_count": 0,
            })
            rows.append({
                "dataset": ds_name,
                "split_or_label": "fake",
                "video_count": fake_video_count,
                "frame_count": fake_frame_count,
            })

        else:
            # Celeb, DFD
            for label_name in ["real", "fake"]:
                video_count = 0
                frame_count = 0

                for root in cfg.get(label_name, []):
                    if not os.path.isdir(root):
                        continue

                    for vid in os.listdir(root):
                        vid_dir = os.path.join(root, vid)
                        if os.path.isdir(vid_dir):
                            frames = count_frames_in_video_dir(vid_dir)
                            if frames > 0:
                                video_count += 1
                                frame_count += frames

                rows.append({
                    "dataset": ds_name,
                    "split_or_label": label_name,
                    "video_count": video_count,
                    "frame_count": frame_count,
                })

    return rows


# =========================================================
# 출력
# =========================================================
def print_train_stats(rows, split_stats, total_frames, tr_len, va_len):
    print("\n" + "=" * 90)
    print("[TRAIN DATASET STATS]")
    print("=" * 90)
    print(f"{'Dataset':30s} {'Label':>5s} {'Videos':>12s} {'Frames':>12s} {'TrainFrames':>12s} {'ValFrames':>12s}")
    print("-" * 90)

    sum_videos = 0
    sum_frames = 0

    for row in rows:
        key = row["dataset"]
        train_frames = split_stats[key]["train_frames"]
        val_frames = split_stats[key]["val_frames"]

        print(
            f"{key:30s} "
            f"{row['label']:>5d} "
            f"{row['video_count']:>12,d} "
            f"{row['frame_count']:>12,d} "
            f"{train_frames:>12,d} "
            f"{val_frames:>12,d}"
        )

        sum_videos += row["video_count"]
        sum_frames += row["frame_count"]

    print("-" * 90)
    print(
        f"{'TOTAL':30s} "
        f"{'':>5s} "
        f"{sum_videos:>12,d} "
        f"{sum_frames:>12,d} "
        f"{tr_len:>12,d} "
        f"{va_len:>12,d}"
    )
    print("=" * 90)
    print(f"Total frames collected: {total_frames:,}")
    print(f"Train split (80%): {tr_len:,}")
    print(f"Val split   (20%): {va_len:,}")


def print_test_stats(rows):
    print("\n" + "=" * 90)
    print("[TEST DATASET STATS]")
    print("=" * 90)
    print(f"{'Dataset':20s} {'Label':10s} {'Videos':>12s} {'Frames':>12s}")
    print("-" * 90)

    total_videos = 0
    total_frames = 0

    for row in rows:
        print(
            f"{row['dataset']:20s} "
            f"{row['split_or_label']:10s} "
            f"{row['video_count']:>12,d} "
            f"{row['frame_count']:>12,d}"
        )
        total_videos += row["video_count"]
        total_frames += row["frame_count"]

    print("-" * 90)
    print(f"{'TOTAL':20s} {'':10s} {total_videos:>12,d} {total_frames:>12,d}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-root",
        type=str,
        default="/home/oem/deepfake/hdd",
        help="FF++ root path (train_rgb.py의 --data-dir)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="raw",
        help="FF++ compression type (raw/c23/c40 등)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train_rgb.py의 random split seed",
    )
    args = parser.parse_args()

    train_rows, split_stats, total_frames, tr_len, va_len = collect_train_ffpp_stats(
        root_dir=args.train_root,
        compression=args.compression,
        seed=args.seed,
    )
    print_train_stats(train_rows, split_stats, total_frames, tr_len, va_len)

    test_rows = collect_test_stats()
    print_test_stats(test_rows)


if __name__ == "__main__":
    main()