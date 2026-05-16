#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path
import shutil
from typing import List, Optional

# ====== 사용자 설정 ======
OUTPUT_DIR = Path("/home/oem/deepfake/LLMs_test")  # 저장 루트
RANDOM_SEED = 42
MAX_VIDEOS = 1000                 # 프레임 폴더(= 영상 단위) 최대 선택 개수
PICK_STRATEGY = "random"          # "random" | "middle" | "first"

# ====== 데이터셋 정의 (fake만 대상) ======
DATASETS = {
    "Celeb": {
        "fake": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-synthesis"
        ]
    },
    "DFD": {
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences"
        ]
    },
    "DeepfakeTIMIT": {
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality"
        ]
    },
    "WildDeepfake": {
        "root": "/home/oem/deepfake/hdd_5TB/WildDeepfake",
        "splits": ["train", "test"]
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def find_frame_dirs(fake_root: Path, min_images: int = 1) -> List[Path]:
    """
    fake_root 하위에서 '이미지(프레임)가 min_images 장 이상 들어있는 디렉토리'를 영상 단위로 간주하여 수집.
    DeepfakeTIMIT처럼 speaker/utterance-frame_dir/0001.png 구조를 자연스럽게 처리함.
    """
    if not fake_root.exists():
        return []

    candidate_dirs = set()
    for img in fake_root.rglob("*"):
        if is_image_file(img):
            candidate_dirs.add(img.parent)

    frame_dirs: List[Path] = []
    for d in candidate_dirs:
        try:
            n_imgs = sum(1 for x in d.iterdir() if is_image_file(x))
        except PermissionError:
            continue
        if n_imgs >= min_images:
            frame_dirs.append(d)

    return frame_dirs

def pick_one_image(img_dir: Path, strategy: str = "random") -> Optional[Path]:
    imgs = sorted([p for p in img_dir.iterdir() if is_image_file(p)])
    if not imgs:
        return None
    if strategy == "middle":
        return imgs[len(imgs) // 2]
    elif strategy == "first":
        return imgs[0]
    else:
        return random.choice(imgs)

def safe_copy(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"    [x] copy fail: {src} -> {dst} ({e})")
        return False

def collect_fake_roots(ds_name: str, cfg: dict) -> List[str]:
    """
    네가 제공한 수집 규칙을 그대로 반영:
      - WildDeepfake: root/split/*/(real|fake) 중 fake 디렉토리만 모음
      - DeepfakeTIMIT: qroot/<speaker> 디렉토리들을 fake root로 모음
      - 그 외: cfg['fake'] 그대로 사용
    """
    fake_roots: List[str] = []

    if ds_name == "WildDeepfake":
        real_roots, fake_roots_local = [], []
        root = cfg.get("root", "")
        splits = cfg.get("splits", [])
        if root and os.path.isdir(root):
            for split in splits:
                sd = os.path.join(root, split)
                if not os.path.isdir(sd):
                    continue
                for subj in os.listdir(sd):
                    base = os.path.join(sd, subj)
                    if not os.path.isdir(base):
                        continue
                    r = os.path.join(base, "real")
                    f = os.path.join(base, "fake")
                    if os.path.isdir(r):
                        real_roots.append(r)
                    if os.path.isdir(f):
                        fake_roots_local.append(f)
        # fake만 사용
        fake_roots = fake_roots_local

    elif ds_name == "DeepfakeTIMIT":
        # 각 품질 루트 아래 speaker 디렉토리들을 fake root로 사용
        for qroot in cfg.get("fake", []):
            if not os.path.isdir(qroot):
                continue
            for spk in os.listdir(qroot):
                spk_path = os.path.join(qroot, spk)
                if os.path.isdir(spk_path):
                    fake_roots.append(spk_path)

    else:
        # 기본: 설정된 fake 경로를 그대로 사용
        for d in cfg.get("fake", []):
            if os.path.isdir(d):
                fake_roots.append(d)

    return fake_roots

def process_dataset(ds_name: str, cfg: dict, out_root: Path):
    print(f"\n[Dataset: {ds_name}] collecting fake roots ...")
    fake_roots = collect_fake_roots(ds_name, cfg)
    print(f"  - found {len(fake_roots)} fake root(s)")

    # 각 fake root에서 프레임 디렉토리 수집
    all_units: List[Path] = []
    for fr in fake_roots:
        froot = Path(fr)
        frame_dirs = find_frame_dirs(froot, min_images=1)
        print(f"    · {froot}: {len(frame_dirs)} frame-dirs")
        all_units.extend(frame_dirs)

    # 중복 제거
    uniq_units = sorted(set(p.resolve() for p in all_units))

    # 출력 폴더(요청: 데이터셋명 폴더)
    out_dir = out_root / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not uniq_units:
        print("  ! no frame-dirs found.")
        return

    random.shuffle(uniq_units)
    selected = uniq_units[: min(MAX_VIDEOS, len(uniq_units))]
    print(f"  → selected {len(selected)} frame-dirs (max {MAX_VIDEOS})")

    saved = 0
    for idx, udir in enumerate(selected, 1):
        img = pick_one_image(udir, strategy=PICK_STRATEGY)
        if img is None:
            print(f"    [x] no images in: {udir}")
            continue

        # 파일명: 데이터셋명_선택인덱스__마지막두레벨폴더명__원본파일명
        parts = udir.parts
        tail_levels = parts[-2:] if len(parts) >= 2 else parts[-1:]
        tail_tag = "__".join(tail_levels)
        out_name = f"{ds_name}_{idx:04d}__{tail_tag}__{img.name}"
        out_path = out_dir / out_name

        # 이름 충돌 시 접미사
        k = 1
        stem, suf = out_path.stem, out_path.suffix
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}({k}){suf}")
            k += 1

        if safe_copy(img, out_path):
            saved += 1
            if saved % 50 == 0:
                print(f"    ... saved {saved} images")

    print(f"  Done. Saved {saved} images -> {out_dir}")

def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 요청: 항상 이 네 개 폴더로 저장
    for ds_name in ["Celeb", "DFD", "DeepfakeTIMIT", "WildDeepfake"]:
        cfg = DATASETS.get(ds_name, {})
        process_dataset(ds_name, cfg, OUTPUT_DIR)

if __name__ == "__main__":
    main()
