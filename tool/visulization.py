from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor
from transformers import AutoModelForImageClassification, AutoFeatureExtractor


# =========================
# 폰트(한글)
# =========================
mpl.rcParams["axes.unicode_minus"] = False
# font_candidates = ["Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "AppleGothic", "Malgun Gothic"]
# available = {f.name for f in fm.fontManager.ttflist}
# chosen = next((name for name in font_candidates if name in available), None)
# assert chosen is not None, (
#     "No Korean font found. Install one (e.g., apt-get install -y fonts-noto-cjk) "
#     "or add a font file and register it."
# )
# mpl.rcParams["font.family"] = chosen

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# =========================
# Config
# =========================
@dataclass(frozen=True)
class VisConfig:
    # ---- roots ----
    root_ffpp_train: Path = Path("/home/oem/deepfake/hdd")       # FF++ train root
    root_test: Path = Path("/home/oem/deepfake/hdd_5TB")         # test datasets root (Celeb/DFD/DeepfakeTIMIT/WildDeepfake)

    out_dir: Path = Path("/home/oem/deepfake/vis_dir3")  # output directory
    seed: int = 42

    # ---- embedding ----
    embed_backend: str = "clip"  # "clip" | "midjourney_cls"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    midjourney_cls_model_name: str = "ideepankarsharma2003/AI_ImageClassification_MidjourneyV6_SDXL"
    batch_size: int = 256
    num_workers: int = 8

    # ---- sampling caps (속도/가독성 조절) ----
    # dataset 단위 cap (train은 FF++ 하나지만 내부 서브소스가 많아서 group cap을 둠)
    cap_train_per_source: int = 8000
    cap_test_per_dataset: int = 8000

    # ---- reduction ----
    reduce_method: str = "tsne"   # "umap" | "tsne"
    pca_dim: int = 50             # 0이면 PCA off

    # UMAP (train fit -> test transform)
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.10
    umap_metric: str = "euclidean"

    # t-SNE (train+test concat fit)
    tsne_perplexity: float = 30.0
    tsne_learning_rate: str | float = "auto"
    tsne_n_iter: int = 1500
    tsne_metric: str = "euclidean"
    tsne_init: str = "pca"


CFG = VisConfig()


# =========================
# Utils
# =========================
def is_image_path_str(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXTS


def list_images_recursive(root: Path) -> List[Path]:
    assert root.exists(), f"Root not found: {root}"
    paths: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if is_image_path_str(fn):
                paths.append((Path(dirpath) / fn).resolve())
    return paths


def stable_seed(name: str, base: int) -> int:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return base + int(h[:8], 16)


def cap_list(items: List[Path], cap: int, seed_key: str) -> List[Path]:
    if cap <= 0 or len(items) <= cap:
        return items
    rng = random.Random(stable_seed(seed_key, CFG.seed))
    idx = rng.sample(range(len(items)), k=cap)
    return [items[i] for i in idx]


def infer_label_from_path(p: Path) -> int:
    """
    휴리스틱 라벨:
      - real/original/pristine/youtube/actors => 0
      - fake/deepfake/manipulated/synthesis/forged => 1
      - 모호하면 -1
    """
    s = str(p).lower()
    real_keys = ["real", "original", "pristine", "youtube-real", "actors", "original_sequences"]
    fake_keys = ["fake", "deepfake", "manipulated", "synthesis", "forged", "swap", "faceshifter", "neuraltextures", "face2face"]
    if any(k in s for k in fake_keys) and not any(k in s for k in ["real", "original", "pristine"]):
        return 1
    if any(k in s for k in real_keys) and not any(k in s for k in ["fake", "deepfake", "manipulated", "synthesis"]):
        return 0
    # 둘 다 섞여 있거나 애매하면 -1
    return -1


# =========================
# Scanners
# Each row: (path, label, dataset_name, split_name, domain)
#  - dataset_name: FF++ / Celeb / DFD / DeepfakeTIMIT / WildDeepfake
#  - domain: finer category for debugging (method/actors/youtube/...)
# =========================
Row = Tuple[Path, int, str, str, str]


def scan_ffpp_train(root: Path) -> List[Row]:
    """
    train 데이터셋: ./hdd
    - manipulated_sequences/*/raw/mtcnn (fake=1)
    - original_sequences/{actors,youtube}/raw/mtcnn (real=0)
    """
    assert root.exists(), f"FF++ train root not found: {root}"
    rows: List[Row] = []
    dataset = "FF++"

    # originals
    orig_actors = root / "original_sequences" / "actors" / "raw" / "mtcnn"
    orig_youtube = root / "original_sequences" / "youtube" / "raw" / "mtcnn"
    assert orig_actors.exists(), f"Missing: {orig_actors}"
    assert orig_youtube.exists(), f"Missing: {orig_youtube}"

    actors_imgs = cap_list(list_images_recursive(orig_actors), CFG.cap_train_per_source, "ffpp:actors")
    yt_imgs = cap_list(list_images_recursive(orig_youtube), CFG.cap_train_per_source, "ffpp:youtube")

    for p in actors_imgs:
        rows.append((p, 0, dataset, "train", "original_actors"))
    for p in yt_imgs:
        rows.append((p, 0, dataset, "train", "original_youtube"))

    # manipulated
    mani_root = root / "manipulated_sequences"
    assert mani_root.exists(), f"Missing: {mani_root}"

    # methods present in your tree
    methods = [
        "DeepFakeDetection",
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures",
    ]
    for m in methods:
        mtcnn_dir = mani_root / m / "raw" / "mtcnn"
        assert mtcnn_dir.exists(), f"Missing: {mtcnn_dir}"
        imgs = cap_list(list_images_recursive(mtcnn_dir), CFG.cap_train_per_source, f"ffpp:{m}")
        for p in imgs:
            rows.append((p, 1, dataset, "train", f"manip_{m}"))

    assert len(rows) > 0, "No FF++ train images collected."
    return rows


def scan_celeb_test(root_test: Path) -> List[Row]:
    """
    ./hdd_5TB/Celeb
      - Celeb-real (real=0)
      - YouTube-real (real=0)
      - Celeb-synthesis (fake=1)
    """
    d = root_test / "Celeb"
    assert d.exists(), f"Missing: {d}"
    rows: List[Row] = []
    dataset = "Celeb"

    mapping = [
        ("Celeb-real", 0, "celeb_real"),
        ("YouTube-real", 0, "youtube_real"),
        ("Celeb-synthesis", 1, "celeb_synthesis"),
    ]
    for sub, y, dom in mapping:
        p = d / sub
        assert p.exists(), f"Missing: {p}"
        imgs = cap_list(list_images_recursive(p), CFG.cap_test_per_dataset, f"test:celeb:{sub}")
        for img in imgs:
            rows.append((img, y, dataset, "test", dom))

    assert len(rows) > 0, "No Celeb test images collected."
    return rows


def scan_dfd_test(root_test: Path) -> List[Row]:
    """
    ./hdd_5TB/DFD
      - DFD_original_sequences (real=0)
      - DFD_manipulated_sequences (fake=1)
    (내부 구조가 더 깊어도 재귀로 다 긁음)
    """
    d = root_test / "DFD"
    assert d.exists(), f"Missing: {d}"
    rows: List[Row] = []
    dataset = "DFD"

    orig = d / "DFD_original_sequences"
    mani = d / "DFD_manipulated_sequences"
    assert orig.exists(), f"Missing: {orig}"
    assert mani.exists(), f"Missing: {mani}"

    orig_imgs = cap_list(list_images_recursive(orig), CFG.cap_test_per_dataset, "test:dfd:orig")
    mani_imgs = cap_list(list_images_recursive(mani), CFG.cap_test_per_dataset, "test:dfd:mani")

    for p in orig_imgs:
        rows.append((p, 0, dataset, "test", "original"))
    for p in mani_imgs:
        rows.append((p, 1, dataset, "test", "manipulated"))

    assert len(rows) > 0, "No DFD test images collected."
    return rows


def scan_deepfaketimit_test(root_test: Path) -> List[Row]:
    """
    ./hdd_5TB/DeepfakeTIMIT
      - higher_quality
      - lower_quality
    내부 real/fake 폴더명이 케이스마다 달라서 휴리스틱으로 라벨 추정.
    """
    d = root_test / "DeepfakeTIMIT"
    assert d.exists(), f"Missing: {d}"
    rows: List[Row] = []
    dataset = "DeepfakeTIMIT"

    for sub in ["higher_quality", "lower_quality"]:
        p = d / sub
        assert p.exists(), f"Missing: {p}"
        imgs = cap_list(list_images_recursive(p), CFG.cap_test_per_dataset, f"test:dftimit:{sub}")
        for img in imgs:
            y = infer_label_from_path(img)
            rows.append((img, y, dataset, "test", sub))

    assert len(rows) > 0, "No DeepfakeTIMIT test images collected."
    return rows


def scan_wilddeepfake_test(root_test: Path) -> List[Row]:
    """
    ./hdd_5TB/WildDeepfake
      - train
      - test
    내부 real/fake 폴더명이 케이스마다 달라서 휴리스틱 라벨 추정.
    (너가 'test로 사용한' subset만 쓰려면 여기서 test만 스캔하도록 바꾸면 됨)
    """
    d = root_test / "WildDeepfake"
    assert d.exists(), f"Missing: {d}"
    rows: List[Row] = []
    dataset = "WildDeepfake"

    # 너가 원하면 아래를 ["test"]만으로 바꿔도 됨
    for sub in ["test"]:  # <--- 기본: test만 시각화
        p = d / sub
        assert p.exists(), f"Missing: {p}"
        imgs = cap_list(list_images_recursive(p), CFG.cap_test_per_dataset, f"test:wild:{sub}")
        for img in imgs:
            y = infer_label_from_path(img)
            rows.append((img, y, dataset, "test", sub))

    assert len(rows) > 0, "No WildDeepfake test images collected."
    return rows


def collect_all_rows() -> Tuple[List[Row], List[Row]]:
    train_rows = scan_ffpp_train(CFG.root_ffpp_train)

    test_rows: List[Row] = []
    test_rows.extend(scan_celeb_test(CFG.root_test))
    test_rows.extend(scan_dfd_test(CFG.root_test))
    test_rows.extend(scan_deepfaketimit_test(CFG.root_test))
    test_rows.extend(scan_wilddeepfake_test(CFG.root_test))

    return train_rows, test_rows


# =========================
# Embedding
# =========================
class ImagePathPILDataset(Dataset):
    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Image.Image:
        return Image.open(self.paths[idx]).convert("RGB")


def embed_paths_clip(paths: List[Path]) -> np.ndarray:
    ds = ImagePathPILDataset(paths)
    dl = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=lambda xs: xs,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(CFG.clip_model_name)
    model = CLIPModel.from_pretrained(CFG.clip_model_name)
    model.eval().to(device)

    feats: List[np.ndarray] = []
    with torch.inference_mode():
        for batch_pil in tqdm(dl, desc="embed (clip)", unit="batch"):
            inputs = processor(images=batch_pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
            img_feat = model.get_image_features(pixel_values=pixel_values)
            feats.append(img_feat.detach().float().cpu().numpy())

    emb = np.concatenate(feats, axis=0)
    assert emb.shape[0] == len(paths)
    return emb


def embed_paths_midjourney_cls(paths: List[Path]) -> np.ndarray:
    ds = ImagePathPILDataset(paths)
    dl = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=lambda xs: xs,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = AutoFeatureExtractor.from_pretrained(CFG.midjourney_cls_model_name)
    model = AutoModelForImageClassification.from_pretrained(CFG.midjourney_cls_model_name)
    model.eval().to(device)

    feats: List[np.ndarray] = []
    with torch.inference_mode():
        for batch_pil in tqdm(dl, desc="embed (midjourney_cls)", unit="batch"):
            inputs = fe(images=batch_pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            hs = outputs.hidden_states[-1]   # [B, seq, hidden]
            cls_feat = hs[:, 0, :]           # CLS token
            feats.append(cls_feat.detach().float().cpu().numpy())

    emb = np.concatenate(feats, axis=0)
    assert emb.shape[0] == len(paths)
    return emb


def embed_paths(paths: List[Path]) -> np.ndarray:
    assert CFG.embed_backend in ("clip", "midjourney_cls")
    if CFG.embed_backend == "clip":
        return embed_paths_clip(paths)
    return embed_paths_midjourney_cls(paths)


# =========================
# Reduction: UMAP or t-SNE
# =========================
def reduce_umap_train_fit_test_transform(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import umap  # pip install umap-learn
    except Exception as e:
        raise RuntimeError("UMAP not available. Install with: pip install umap-learn") from e

    um = umap.UMAP(
        n_neighbors=int(CFG.umap_n_neighbors),
        min_dist=float(CFG.umap_min_dist),
        n_components=2,
        metric=str(CFG.umap_metric),
        random_state=CFG.seed,
    )
    um.fit(X_train)
    Z_train = um.embedding_
    Z_test = um.transform(X_test)
    return Z_train, Z_test


def reduce_tsne_all_fit(X_all: np.ndarray) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=float(CFG.tsne_perplexity),
        learning_rate=CFG.tsne_learning_rate,
        n_iter=int(CFG.tsne_n_iter),
        metric=str(CFG.tsne_metric),
        init=str(CFG.tsne_init),
        random_state=CFG.seed,
        verbose=1,
    )
    return tsne.fit_transform(X_all)


# =========================
# Plot
# - color: dataset_name (FF++/Celeb/DFD/DeepfakeTIMIT/WildDeepfake)
# - marker: label (real o / fake x / unknown .)
# - size/alpha: train smaller+lighter, test bigger+stronger
# =========================
def plot_2d(
    Z: np.ndarray,
    is_test: np.ndarray,
    labels: np.ndarray,   # marker에 안 쓰지만 시그니처 유지
    datasets: List[str],
    out_path: Path,
    title: str,
) -> None:
    assert len(Z) == len(is_test) == len(labels) == len(datasets)

    # === legend에 표시될 dataset 이름 매핑 ===
    DISPLAY_NAME = {
        "FF++": "FaceForensics++",
        "Celeb": "Celeb-DF v2",
        "DFD": "DFDC",
        "DeepfakeTIMIT": "DeepfakeTIMIT",
        "WildDeepfake": "WildDeepfake",
    }

    ds_list = sorted(set(datasets))
    cmap = plt.get_cmap("tab10")
    ds_to_color = {d: cmap(i % 10) for i, d in enumerate(ds_list)}

    fig, ax = plt.subplots(figsize=(12, 9))

    train_alpha, test_alpha = 0.25, 0.90
    train_size, test_size = 10, 22

    idx_train = np.where(is_test == 0)[0]
    idx_test = np.where(is_test == 1)[0]

    marker = "o"
    for d in ds_list:
        # train points
        tr_idx = idx_train[[i for i, k in enumerate(idx_train) if datasets[k] == d]]
        if len(tr_idx) > 0:
            pts = Z[tr_idx]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=train_size,
                c=[ds_to_color[d]],
                marker=marker,
                alpha=train_alpha,
                linewidths=0.0,
                edgecolors="none",
            )

        # test points
        te_idx = idx_test[[i for i, k in enumerate(idx_test) if datasets[k] == d]]
        if len(te_idx) > 0:
            pts = Z[te_idx]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=test_size,
                c=[ds_to_color[d]],
                marker=marker,
                alpha=test_alpha,
                linewidths=0.0,
                edgecolors="none",
            )

    # === legend: dataset(color)만, 타이틀 제거, 그래프 내부 좌상단 ===
    from matplotlib.patches import Patch
    ds_patches = [
        Patch(
            facecolor=ds_to_color[d],
            edgecolor="none",
            label=DISPLAY_NAME.get(d, d),   # <-- 여기서 표시 이름 적용
        )
        for d in ds_list
    ]
    ax.legend(
        handles=ds_patches,
        loc="upper left",
        frameon=True,
        fontsize=10,
        title=None,
    )

    ax.set_title(title)
    ax.set_xlabel("comp-1")
    ax.set_ylabel("comp-2")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_hue_with_label_marker(
    Z: np.ndarray,
    labels: np.ndarray,          # 0 real, 1 fake
    datasets: List[str],         # hue
    out_path: Path,
    title: str,
) -> None:
    assert len(Z) == len(labels) == len(datasets)

    DISPLAY_NAME = {
        "FF++": "FaceForensics++",
        "Celeb": "Celeb-DF v2",
        "DFD": "DFDC",
        "DeepfakeTIMIT": "DeepfakeTIMIT",
        "WildDeepfake": "WildDeepfake",
    }

    ds_list = sorted(set(datasets))
    cmap = plt.get_cmap("tab10")
    ds_to_color = {d: cmap(i % 10) for i, d in enumerate(ds_list)}

    fig, ax = plt.subplots(figsize=(12, 9))

    # label -> marker (real/fake만)
    marker_map = {0: "o", 1: "x"}
    size_map = {0: 18, 1: 22}
    alpha = 0.85

    # dataset 색 + label 마커를 같이 적용 (unknown 제외)
    for d in ds_list:
        idx_d = [i for i in range(len(Z)) if datasets[i] == d]
        if not idx_d:
            continue

        for y in [0, 1]:
            idx = [i for i in idx_d if labels[i] == y]
            if not idx:
                continue
            pts = Z[idx]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=size_map[y],
                c=[ds_to_color[d]],
                marker=marker_map[y],
                alpha=alpha,
                linewidths=1.0 if y == 1 else 0.0,   # x는 선이 보여야 해서
                edgecolors="none" if y != 1 else None,
            )

    # legends (dataset hue/color)
    from matplotlib.patches import Patch
    ds_patches = [
        Patch(facecolor=ds_to_color[d], edgecolor="none", label=DISPLAY_NAME.get(d, d))
        for d in ds_list
    ]
    leg_ds = ax.legend(
        handles=ds_patches,
        loc="upper left",
        frameon=True,
        fontsize=10,
        title=None,
    )
    ax.add_artist(leg_ds)

    # legends (label marker) - 그래프 구석에 똑같이
    from matplotlib.lines import Line2D
    label_legend = [
        Line2D([0], [0], marker="o", color="black", linestyle="none", label="real", markersize=8),
        Line2D([0], [0], marker="x", color="black", linestyle="none", label="fake", markersize=8),
    ]
    ax.legend(
        handles=label_legend,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.70),  # dataset legend 아래쪽에 붙이기 (원하면 숫자만 조절)
        frameon=True,
        fontsize=10,
        title=None,
    )

    ax.set_title(title)
    ax.set_xlabel("comp-1")
    ax.set_ylabel("comp-2")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    
    
def plot_label_only_color(
    Z: np.ndarray,
    labels: np.ndarray,      # 0 real, 1 fake
    out_path: Path,
    title: str,
) -> None:
    assert len(Z) == len(labels)

    # real/fake만
    label_order = [0, 1]
    label_name = {0: "real", 1: "fake"}
    label_color = {0: "tab:blue", 1: "tab:red"}

    fig, ax = plt.subplots(figsize=(12, 9))

    for y in label_order:
        idx = np.where(labels == y)[0]
        if len(idx) == 0:
            continue
        pts = Z[idx]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=18,
            c=label_color[y],
            marker="o",
            alpha=0.85,
            linewidths=0.0,
            edgecolors="none",
            label=label_name[y],
        )

    ax.legend(loc="upper left", frameon=True, fontsize=10, title=None)
    ax.set_title(title)
    ax.set_xlabel("comp-1")
    ax.set_ylabel("comp-2")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main() -> None:
    CFG.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) scan
    train_rows, test_rows = collect_all_rows()

    # 2) unpack
    train_paths = [r[0] for r in train_rows]
    train_labels = np.array([r[1] for r in train_rows], dtype=np.int64)
    train_datasets = [r[2] for r in train_rows]
    train_splits = [r[3] for r in train_rows]
    train_domains = [r[4] for r in train_rows]

    test_paths = [r[0] for r in test_rows]
    test_labels = np.array([r[1] for r in test_rows], dtype=np.int64)
    test_datasets = [r[2] for r in test_rows]
    test_splits = [r[3] for r in test_rows]
    test_domains = [r[4] for r in test_rows]

    print(f"[SCAN] train FF++: {len(train_paths)}")
    print(f"[SCAN] test (Celeb/DFD/DeepfakeTIMIT/WildDeepfake): {len(test_paths)}")

    # 3) embed
    train_emb = embed_paths(train_paths)
    test_emb = embed_paths(test_paths)
    np.save(CFG.out_dir / f"train_emb_{CFG.embed_backend}.npy", train_emb)
    np.save(CFG.out_dir / f"test_emb_{CFG.embed_backend}.npy", test_emb)

    # 4) PCA (권장)
    X_train = train_emb
    X_test = test_emb
    if CFG.pca_dim > 0 and X_train.shape[1] > CFG.pca_dim:
        if CFG.reduce_method == "umap":
            # 누수 최소화: PCA도 train으로 fit → test transform
            pca = PCA(n_components=CFG.pca_dim, random_state=CFG.seed)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        else:
            # t-SNE는 어차피 all-fit이라 PCA도 all-fit로 간단히
            pca = PCA(n_components=CFG.pca_dim, random_state=CFG.seed)
            X_all = np.concatenate([X_train, X_test], axis=0)
            X_all = pca.fit_transform(X_all)
            X_train = X_all[: len(X_train)]
            X_test = X_all[len(X_train):]

        np.save(CFG.out_dir / f"train_pca_{CFG.embed_backend}.npy", X_train)
        np.save(CFG.out_dir / f"test_pca_{CFG.embed_backend}.npy", X_test)

    # 5) reduce
    if CFG.reduce_method == "umap":
        Z_train, Z_test = reduce_umap_train_fit_test_transform(X_train, X_test)
        Z_all = np.concatenate([Z_train, Z_test], axis=0)
        method_tag = f"umap_n{CFG.umap_n_neighbors}_d{CFG.umap_min_dist}"
    else:
        X_all = np.concatenate([X_train, X_test], axis=0)
        Z_all = reduce_tsne_all_fit(X_all)
        method_tag = f"tsne_p{CFG.tsne_perplexity}_it{CFG.tsne_n_iter}"

    np.save(CFG.out_dir / f"Z_all_{CFG.embed_backend}_{CFG.reduce_method}.npy", Z_all)

    # 6) plot meta
    all_paths = train_paths + test_paths
    is_test = np.zeros(len(all_paths), dtype=np.int64)
    is_test[len(train_paths):] = 1

    labels = np.full(len(all_paths), -1, dtype=np.int64)
    labels[: len(train_labels)] = train_labels
    labels[len(train_labels):] = test_labels

    datasets = train_datasets + test_datasets

    # 7) plot
    # out_png = CFG.out_dir / f"vis_{CFG.embed_backend}_{method_tag}.png"
    # plot_2d(
    #     Z=Z_all,
    #     is_test=is_test,
    #     labels=labels,
    #     datasets=datasets,
    #     out_path=out_png,
    #     title=f"t-SNE projection of 5-dataset benchmarks",
    # )
    
    # (A) dataset(hue) + real/fake(marker)
    out_png_ds_label = CFG.out_dir / f"vis_{CFG.embed_backend}_{method_tag}_dataset+label.png"
    plot_dataset_hue_with_label_marker(
        Z=Z_all,
        labels=labels,
        datasets=datasets,
        out_path=out_png_ds_label,
        title="t-SNE projection (dataset hue + real/fake marker).",
    )

    # (B) label(real/fake)만 color로
    out_png_label_only = CFG.out_dir / f"vis_{CFG.embed_backend}_{method_tag}_label_only.png"
    plot_label_only_color(
        Z=Z_all,
        labels=labels,
        out_path=out_png_label_only,
        title="t-SNE projection (real vs fake only).",
    )

    # 8) save CSV meta
    meta_rows: List[dict] = []
    for i, p in enumerate(all_paths):
        if is_test[i] == 0:
            dom = train_domains[i]
            sp = train_splits[i]
        else:
            j = i - len(train_paths)
            dom = test_domains[j]
            sp = test_splits[j]
        meta_rows.append(
            {
                "idx": i,
                "is_test": int(is_test[i]),
                "dataset": datasets[i],
                "label": int(labels[i]),
                "split": sp,
                "domain": dom,
                "path": str(p),
                "z1": float(Z_all[i, 0]),
                "z2": float(Z_all[i, 1]),
            }
        )

    pd.DataFrame(meta_rows).to_csv(
        CFG.out_dir / f"meta_{CFG.embed_backend}_{CFG.reduce_method}.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"[OUT] {CFG.out_dir}")
    # print(f"- fig: {out_png}")
    print(f"- reduce_method: {CFG.reduce_method}")
    print(f"- embed_backend: {CFG.embed_backend}")
    print(f"- PCA dim: {CFG.pca_dim}")
    print(f"- train: {len(train_paths)} | test: {len(test_paths)}")


if __name__ == "__main__":
    main()