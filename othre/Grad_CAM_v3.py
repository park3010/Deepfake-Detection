#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final-feature Grad-CAM exporter for four deepfake models:
- RGB+Wavelet+Semantic tri-stream model (final fused attention query-token CAM)
- F3Net-FAD
- M2TR
- Xception

Design goal
-----------
The user asked for CAMs "right before the final output".
So this script uses:
1) Xception / F3Net / M2TR:
   the last spatial Conv2d feature map before the classifier head.
2) Tri-stream RGB+Wavelet+Semantic:
   the final cross-attention output tokens (attn_out) that are produced just before
   attn_global -> fused_feat -> classifier.
   Since attn_out is token-based rather than a 2D conv map, the script computes a
   Grad-CAM-like token importance map and projects it back to the selected query stream
   (default: RGB) for visualization.

This script intentionally follows the configuration style of the user's previous
Grad-CAM export script: TEST_DATASETS + MODEL_CONFIGS + export-by-dataset layout.

Usage examples
--------------
# Export 5 real + 5 fake samples per dataset/model using current MODEL_CONFIGS
python gradcam_final_feature_compare.py

# Only Celeb and DFD, 3 images per class
python gradcam_final_feature_compare.py --datasets Celeb DFD --num-per-class 3

# Only one model
python gradcam_final_feature_compare.py --only-model ours

# Directly test specific image paths
python gradcam_final_feature_compare.py \
    --image /path/to/img1.png /path/to/img2.jpg \
    --only-model ours

Important notes
---------------
- Update checkpoint/config paths in MODEL_CONFIGS to your actual local paths.
- M2TR config path may need adjustment depending on your repo layout.
- For the tri-stream model, the most faithful "final-feature" visualization is token CAM.
  It is not a conventional last-conv Grad-CAM because the final discriminative stage is
  cross-attention + fusion, not a single conv block.
"""

import os
import sys
import gc
import glob
import math
import random
import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# =========================================================
# Basic paths (edit if needed)
# =========================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RGB_ROOT = os.path.join(BASE_DIR, "RGBsparial_step1")
TRI_STREAM_ROOT = os.path.join(BASE_DIR, "Tri_stream")
M2TR_ROOT = os.path.join(
    BASE_DIR,
    "comparison",
    "M2TR_v2",
    "M2TR-Multi-modal-Multi-scale-Transformers-for-Deepfake-Detection",
)
F3NET_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "F3Net"))
EXPORT_ROOT = os.path.join(BASE_DIR, "gradcam_final_feature_export_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
PRED_FAKE_THRESHOLD = 0.5

# =========================================================
# Dataset config (same style as your previous script)
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


# =========================================================
# Model configs (edit checkpoint paths if needed)
# =========================================================
MODEL_CONFIGS = [
    {
        "enable": True,
        "model_type": "single",
        "model_name": "xception",
        "export_key": "xception",
        "checkpoint_path": "/home/oem/deepfake/Ourmethod/RGBsparial_step1/checkpoints/xception/xception_best.pth",
        "image_size": 224,
        "num_classes": 2,
        # Optional explicit target layer path. If None, last Conv2d is used.
        "target_layer": None,
    },
    {
        "enable": True,
        "model_type": "f3net",
        "model_name": "f3net_fad",
        "export_key": "f3net_fad",
        "checkpoint_path": "/home/oem/deepfake/F3Net/checkpoints/FAD/F3Net_last.pth",
        "image_size": 299,
        "num_classes": 1,
        "target_layer": None,
        # Optional: repo root if different
        "f3net_root": F3NET_ROOT,
        "f3net_mode": "FAD",
    },
    {
        "enable": True,
        "model_type": "m2tr",
        "model_name": "m2tr",
        "export_key": "m2tr",
        "checkpoint_path": "/home/oem/deepfake/Ourmethod/comparison/_ckpt/m2tr/checkpoints/M2TR_FFPPVideo_epoch_00010.pyth",
        "config_path": os.path.join(M2TR_ROOT, "configs", "m2tr_ffpp_video.yaml"),
        "image_size": 320,
        "num_classes": 2,
        "target_layer": None,
    },
    {
        "enable": True,
        "model_type": "tri_stream",
        "model_name": "rgb_wavelet_semantic",
        "export_key": "ours",
        "checkpoint_path": "/home/oem/deepfake/Ourmethod/Tri_stream/_ckpt/rgb_wavelet_semantic/v1/best_tri_rgb_wavelet_semantic.pth",
        "image_size": 224,
        "streams": "rgb,wavelet,semantic",
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "dropout": 0.2,
        "wavelet": "sym4",
        "wavelet_level": 2,
        "wavelet_type": "swt",
        "subband": "ll_energy",
        "wavelet_gray": False,
        "no_robust_norm": False,
        "freq_in": "ycbcr",
        "block_energy": "ac",
        "clip_backbone": "openai/clip-vit-base-patch32",
        "finetune_clip": False,
        "resnet_pretrained_wavelet": False,
        # Which query stream slice to visualize from final attn_out
        # one of: rgb, wavelet, avg
        "tri_query_stream_to_show": "avg",
    },
]


# =========================================================
# Utility helpers
# =========================================================
def set_seed(seed: int = 153):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_sys_path(path: str):
    if path and path not in sys.path:
        sys.path.insert(0, path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def remove_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        out[nk] = v
    return out


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from path: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    cur = model
    for attr in module_name.split("."):
        if attr.isdigit():
            cur = cur[int(attr)]
        else:
            cur = getattr(cur, attr)
    return cur


def find_last_conv(module: nn.Module) -> Optional[nn.Module]:
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    denom = cam.max() + 1e-8
    cam = cam / denom
    return cam


def overlay_cam_on_rgb(raw_rgb_uint8: np.ndarray, cam_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = raw_rgb_uint8.shape[:2]
    cam = cv2.resize(cam_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    cam = normalize_cam(cam)
    heatmap = np.uint8(cam * 255.0)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.5 * raw_rgb_uint8 + 0.5 * heatmap, 0, 255).astype(np.uint8)
    return heatmap, overlay


def save_rgb_image(path: str, img_rgb_uint8: np.ndarray):
    cv2.imwrite(path, cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR))


def load_image_rgb(path: str, image_size: int) -> Tuple[Image.Image, np.ndarray]:
    img = Image.open(path).convert("RGB")
    raw = np.array(img.resize((image_size, image_size)))
    return img, raw


def build_rgb_transform(image_size: int, mean=None, std=None):
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_imagenet_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def logits_to_target_index(logits: torch.Tensor, target_mode: str = "pred") -> int:
    """
    Return predicted label using configurable fake threshold.
    0 = real, 1 = fake

    For 2-logit output:
        prob_fake = softmax(logits)[1]
    For 1-logit output:
        prob_fake = sigmoid(logit)
    """
    global PRED_FAKE_THRESHOLD

    if logits.ndim == 2 and logits.size(0) == 1:
        logits = logits[0]

    # 2-class logits: [real_logit, fake_logit]
    if logits.ndim == 1 and logits.numel() == 2:
        probs = torch.softmax(logits.float(), dim=0)
        prob_fake = float(probs[1].item())
        return 1 if prob_fake >= PRED_FAKE_THRESHOLD else 0

    # 1-logit binary output
    if logits.ndim == 1 and logits.numel() == 1:
        prob_fake = float(torch.sigmoid(logits[0].float()).item())
        return 1 if prob_fake >= PRED_FAKE_THRESHOLD else 0

    # scalar binary logit
    if logits.ndim == 0:
        prob_fake = float(torch.sigmoid(logits.float()).item())
        return 1 if prob_fake >= PRED_FAKE_THRESHOLD else 0

    raise ValueError(f"Unexpected logits shape for prediction: {tuple(logits.shape)}")


def logits_to_probs(logits: torch.Tensor) -> Tuple[float, float]:
    """returns (prob_real, prob_fake)"""
    if logits.ndim == 2 and logits.size(0) == 1:
        logits = logits[0]

    if logits.ndim == 1 and logits.numel() == 2:
        probs = torch.softmax(logits.float(), dim=0)
        return float(probs[0].item()), float(probs[1].item())

    if logits.ndim == 1 and logits.numel() == 1:
        prob_fake = float(torch.sigmoid(logits[0].float()).item())
        return 1.0 - prob_fake, prob_fake

    if logits.ndim == 0:
        prob_fake = float(torch.sigmoid(logits.float()).item())
        return 1.0 - prob_fake, prob_fake

    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


def unwrap_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        raise ValueError(f"dict output has no 'logits' key: {list(output.keys())}")
    if isinstance(output, (tuple, list)):
        tensors = [x for x in output if torch.is_tensor(x)]
        if not tensors:
            raise ValueError("No tensor in model output tuple/list.")
        # Prefer 2D logits-like tensor
        for t in reversed(tensors):
            if t.ndim in (1, 2):
                return t
        return tensors[-1]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported output type: {type(output)}")


def select_score_from_logits(logits: torch.Tensor, target_index: Optional[int] = None) -> torch.Tensor:
    if logits.ndim == 2 and logits.size(1) == 2:
        idx = logits_to_target_index(logits) if target_index is None else int(target_index)
        return logits[:, idx].sum()
    if logits.ndim == 2 and logits.size(1) == 1:
        return logits[:, 0].sum()
    if logits.ndim == 1 and logits.numel() == 2:
        idx = logits_to_target_index(logits.unsqueeze(0)) if target_index is None else int(target_index)
        return logits[idx]
    if logits.ndim == 1 and logits.numel() == 1:
        return logits[0]
    raise ValueError(f"Unexpected logits shape for score selection: {tuple(logits.shape)}")


def collect_roots_for_dataset(ds_name: str, cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    if ds_name == "WildDeepfake":
        real_roots, fake_roots = [], []
        root = cfg["root"]
        for split in cfg["splits"]:
            split_dir = os.path.join(root, split)
            if not os.path.isdir(split_dir):
                continue
            for meta in os.listdir(split_dir):
                base = os.path.join(split_dir, meta)
                r = os.path.join(base, "real")
                f = os.path.join(base, "fake")
                if os.path.isdir(r):
                    real_roots.append(r)
                if os.path.isdir(f):
                    fake_roots.append(f)
        return {"real": real_roots, "fake": fake_roots}

    if ds_name == "DeepfakeTIMIT":
        fake_roots = []
        for quality_root in cfg["fake"]:
            if not os.path.isdir(quality_root):
                continue
            for speaker in os.listdir(quality_root):
                sp_path = os.path.join(quality_root, speaker)
                if os.path.isdir(sp_path):
                    fake_roots.append(sp_path)
        return {"real": [], "fake": fake_roots}

    return cfg


def collect_frame_paths(video_dir: str) -> List[str]:
    frames = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        frames.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(frames)


def sample_images_from_roots(roots: List[str], num_per_class: int) -> List[str]:
    candidates = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        videos = sorted(os.listdir(root))
        for vid in videos:
            vid_dir = os.path.join(root, vid)
            if not os.path.isdir(vid_dir):
                continue
            frames = collect_frame_paths(vid_dir)
            if frames:
                candidates.append(frames[len(frames) // 2])
    return candidates[:num_per_class]


# =========================================================
# Common Grad-CAM core for conv feature maps
# =========================================================
class ConvGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def compute(self, score: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        score.backward()   # retain_graph=True 제거

        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        if acts.ndim != 4 or grads.ndim != 4:
            raise ValueError(
                f"ConvGradCAM expects 4D activations/gradients, got acts={acts.shape}, grads={grads.shape}"
            )

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)
        cam = cam[0].detach().cpu().numpy()
        return normalize_cam(cam)


# =========================================================
# Runner dataclass
# =========================================================
@dataclass
class ModelRunner:
    export_key: str
    image_size: int
    run_cam: Any
    model: Optional[nn.Module] = None


# =========================================================
# Xception builder
# =========================================================
def build_xception_runner(cfg: Dict[str, Any]) -> ModelRunner:
    add_sys_path(RGB_ROOT)
    xception_mod = load_module_from_path(
        "xception_module_for_cam",
        os.path.join(RGB_ROOT, "Xception", "xception.py"),
    )
    model = xception_mod.xception(num_classes=2, use_cbam=False)

    ckpt = torch.load(cfg["checkpoint_path"], map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt.get("model", ckpt))) if isinstance(ckpt, dict) else ckpt
    state = remove_module_prefix(state)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()

    target_layer = get_module_by_name(model, cfg["target_layer"]) if cfg.get("target_layer") else find_last_conv(model)
    if target_layer is None:
        raise RuntimeError("Could not find target conv layer for Xception.")

    tfm = build_rgb_transform(cfg["image_size"], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def run_cam(image_path: str) -> Dict[str, Any]:
        pil_img, raw_rgb = load_image_rgb(image_path, cfg["image_size"])
        x = tfm(pil_img).unsqueeze(0).to(DEVICE)

        cam_engine = ConvGradCAM(model, target_layer)
        try:
            out = model(x)
            logits = unwrap_logits(out)
            score = select_score_from_logits(logits)
            cam = cam_engine.compute(score)
            heatmap, overlay = overlay_cam_on_rgb(raw_rgb, cam)
            pred = logits_to_target_index(logits)
            prob_real, prob_fake = logits_to_probs(logits)
            return {
                "raw_rgb": raw_rgb,
                "cam": cam,
                "heatmap": heatmap,
                "overlay": overlay,
                "pred": pred,
                "prob_real": prob_real,
                "prob_fake": prob_fake,
                "logits": logits.detach().cpu(),
                "note": f"target_layer={target_layer.__class__.__name__}",
            }
        finally:
            cam_engine.remove()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return ModelRunner(cfg["export_key"], cfg["image_size"], run_cam)


# =========================================================
# F3Net builder
# =========================================================
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model_state", "state_dict", "model"]:
            if k in ckpt:
                return ckpt[k]
    return ckpt


def _clean_state_dict(sd):
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_sd[k] = v
    return new_sd


def _infer_f3net_num_classes(state):
    # 우선순위: 마지막 FC 또는 FAD_xcep.fc
    for key in ["fc.weight", "FAD_xcep.fc.weight"]:
        if key in state and hasattr(state[key], "shape"):
            return int(state[key].shape[0])
    # 못 찾으면 기본 2
    return 2


def _drop_mismatched_classifier_keys(state, model_state):
    filtered = {}
    dropped = []

    for k, v in state.items():
        if k in model_state:
            if tuple(model_state[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                dropped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
        else:
            filtered[k] = v

    if dropped:
        print("[F3Net] dropped mismatched keys:")
        for k, s1, s2 in dropped:
            print(f"  - {k}: ckpt{s1} != model{s2}")

    return filtered

def build_f3net_runner(cfg):
    device = DEVICE

    f3net_root = cfg.get("f3net_root", F3NET_ROOT)
    f3net_models_py = os.path.join(f3net_root, "models.py")
    if not os.path.isfile(f3net_models_py):
        raise FileNotFoundError(f"F3Net models.py not found: {f3net_models_py}")

    f3net_mod = load_module_from_path("f3net_models_for_cam", f3net_models_py)
    F3NetModel = f3net_mod.F3Net

    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    state = _clean_state_dict(_extract_state_dict(ckpt))

    num_classes = _infer_f3net_num_classes(state)
    print(f"[F3Net] inferred num_classes from checkpoint = {num_classes}")

    model = None
    tried = []
    for ctor in [
        lambda: F3NetModel(mode=cfg.get("f3net_mode", "FAD"), num_classes=num_classes),
        lambda: F3NetModel(cfg.get("f3net_mode", "FAD"), num_classes=num_classes),
        lambda: F3NetModel(num_classes=num_classes),
        lambda: F3NetModel(),
    ]:
        try:
            model = ctor().to(device)
            break
        except Exception as e:
            tried.append(str(e))

    if model is None:
        raise RuntimeError("F3Net build failed:\n- " + "\n- ".join(tried))

    filtered_state = _drop_mismatched_classifier_keys(state, model.state_dict())
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    if missing:
        print(f"[F3Net] missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[F3Net] unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    model.eval()

    target_layer = get_module_by_name(model, cfg["target_layer"]) if cfg.get("target_layer") else find_last_conv(model)
    if target_layer is None:
        raise RuntimeError("Could not find target conv layer for F3Net.")

    tfm = build_imagenet_transform(cfg["image_size"])

    def run_cam(image_path: str) -> Dict[str, Any]:
        pil_img, raw_rgb = load_image_rgb(image_path, cfg["image_size"])
        x = tfm(pil_img).unsqueeze(0).to(device)

        cam_engine = ConvGradCAM(model, target_layer)
        try:
            out = model(x)
            logits = unwrap_logits(out)
            score = select_score_from_logits(logits)
            cam = cam_engine.compute(score)
            heatmap, overlay = overlay_cam_on_rgb(raw_rgb, cam)
            pred = logits_to_target_index(logits if logits.ndim == 2 else logits.unsqueeze(0))
            prob_real, prob_fake = logits_to_probs(logits)
            return {
                "raw_rgb": raw_rgb,
                "cam": cam,
                "heatmap": heatmap,
                "overlay": overlay,
                "pred": pred,
                "prob_real": prob_real,
                "prob_fake": prob_fake,
                "logits": logits.detach().cpu(),
                "note": f"target_layer={target_layer.__class__.__name__}",
            }
        finally:
            cam_engine.remove()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return ModelRunner(cfg["export_key"], cfg["image_size"], run_cam)

# =========================================================
# M2TR builder
# =========================================================
def build_m2tr_runner(cfg: Dict[str, Any]) -> ModelRunner:
    add_sys_path(M2TR_ROOT)
    add_sys_path(os.path.join(M2TR_ROOT, "tools"))

    from M2TR.utils.build_helper import build_model  # type: ignore
    from M2TR.utils.checkpoint import load_test_checkpoint  # type: ignore
    from tools.utils import load_config  # type: ignore

    if not os.path.isfile(cfg["config_path"]):
        raise FileNotFoundError(f"M2TR config not found: {cfg['config_path']}")

    # load_config()는 argparse 스타일 객체를 기대함
    # 그리고 내부에서 ./configs/default.yaml, ./configs/<cfg_file> 를 열기 때문에
    # cwd를 M2TR_ROOT로 바꾼 뒤 cfg_file에는 basename만 넘겨야 함
    m2tr_args = SimpleNamespace(
        cfg_file=os.path.basename(cfg["config_path"]),
        shard_id=0,
        base_lr=None,
    )

    old_cwd = os.getcwd()
    try:
        os.chdir(M2TR_ROOT)
        m2tr_cfg = load_config(m2tr_args)
    finally:
        os.chdir(old_cwd)

    model = build_model(m2tr_cfg)
    m2tr_cfg["TEST"]["CHECKPOINT_TEST_PATH"] = cfg["checkpoint_path"]
    load_test_checkpoint(m2tr_cfg, model)
    model.to(DEVICE).eval()

    target_layer = get_module_by_name(model, cfg["target_layer"]) if cfg.get("target_layer") else find_last_conv(model)
    if target_layer is None:
        raise RuntimeError("Could not find target conv layer for M2TR.")

    def preprocess(image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        pil_img, raw_rgb = load_image_rgb(image_path, cfg["image_size"])
        x = np.asarray(pil_img.resize((cfg["image_size"], cfg["image_size"]))).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
        return x, raw_rgb

    def run_cam(image_path: str) -> Dict[str, Any]:
        x, raw_rgb = preprocess(image_path)
        cam_engine = ConvGradCAM(model, target_layer)

        out = None
        logits = None
        score = None

        try:
            model.zero_grad(set_to_none=True)

            out = model({"img": x})
            logits = unwrap_logits(out)
            score = select_score_from_logits(logits)
            cam = cam_engine.compute(score)
            heatmap, overlay = overlay_cam_on_rgb(raw_rgb, cam)
            pred = logits_to_target_index(logits)
            prob_real, prob_fake = logits_to_probs(logits)

            result = {
                "raw_rgb": raw_rgb,
                "cam": cam,
                "heatmap": heatmap,
                "overlay": overlay,
                "pred": pred,
                "prob_real": prob_real,
                "prob_fake": prob_fake,
                "logits": logits.detach().cpu(),
                "note": f"target_layer={target_layer.__class__.__name__}",
            }
            return result

        finally:
            cam_engine.remove()

            if cam_engine.activations is not None:
                del cam_engine.activations
            if cam_engine.gradients is not None:
                del cam_engine.gradients

            del score
            del logits
            del out
            del x

            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    return ModelRunner(cfg["export_key"], cfg["image_size"], run_cam)


# =========================================================
# Tri-stream final-feature CAM builder
# =========================================================
class BranchFeatMapGradCAM:
    def __init__(self, model: nn.Module, branch_module: nn.Module, name: str):
        self.model = model
        self.branch_module = branch_module
        self.name = name
        self.feat_map = None
        self.handle = branch_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.feat_map = None
        if isinstance(output, dict) and "feat_map" in output:
            self.feat_map = output["feat_map"]
            if torch.is_tensor(self.feat_map) and self.feat_map.requires_grad:
                self.feat_map.retain_grad()

    def remove(self):
        self.handle.remove()

    def get_cam(self) -> np.ndarray:
        if self.feat_map is None:
            raise RuntimeError(f"[{self.name}] feat_map was not captured.")
        if self.feat_map.grad is None:
            raise RuntimeError(f"[{self.name}] feat_map grad is None.")

        acts = self.feat_map
        grads = self.feat_map.grad

        if acts.ndim != 4 or grads.ndim != 4:
            raise ValueError(
                f"[{self.name}] expected 4D feat_map/grads, got acts={acts.shape}, grads={grads.shape}"
            )

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)
        cam = cam[0].detach().cpu().numpy()
        return normalize_cam(cam)

def build_tri_stream_runner(cfg: Dict[str, Any]) -> ModelRunner:
    import importlib

    for k in list(sys.modules.keys()):
        if k == "models" or k.startswith("models."):
            del sys.modules[k]

    if TRI_STREAM_ROOT in sys.path:
        sys.path.remove(TRI_STREAM_ROOT)
    sys.path.insert(0, TRI_STREAM_ROOT)
    importlib.invalidate_caches()

    tri_train_py = os.path.join(TRI_STREAM_ROOT, "train.py")
    if not os.path.isfile(tri_train_py):
        tri_train_py = os.path.join(BASE_DIR, "train.py")
    tri_mod = load_module_from_path("tri_train_for_cam", tri_train_py)

    args = SimpleNamespace(
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        wavelet=cfg["wavelet"],
        wavelet_level=cfg["wavelet_level"],
        wavelet_type=cfg["wavelet_type"],
        subband=cfg["subband"],
        wavelet_gray=cfg["wavelet_gray"],
        no_robust_norm=cfg["no_robust_norm"],
        freq_in=cfg["freq_in"],
        block_energy=cfg["block_energy"],
        clip_backbone=cfg["clip_backbone"],
        finetune_clip=cfg["finetune_clip"],
        resnet_pretrained_wavelet=cfg["resnet_pretrained_wavelet"],
        img_size=cfg["image_size"],
        rgb_ckpt=None,
        wavelet_ckpt=None,
        dct_ckpt=None,
    )

    streams = tri_mod.parse_streams(cfg["streams"])
    model = tri_mod.build_model(args, streams)

    ckpt = torch.load(cfg["checkpoint_path"], map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt.get("model", ckpt))) if isinstance(ckpt, dict) else ckpt
    state = remove_module_prefix(state)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()

    clip_processor = None
    if "semantic" in streams:
        from transformers import CLIPImageProcessor
        clip_processor = CLIPImageProcessor.from_pretrained(cfg["clip_backbone"])

    def preprocess(image_path: str) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        pil_img = Image.open(image_path).convert("RGB")
        resized = pil_img.resize((cfg["image_size"], cfg["image_size"]))
        raw_rgb = np.array(resized)
        arr_rgb = np.array(resized).astype(np.float32)
        arr_rgb_uint8 = arr_rgb.astype(np.uint8)
        arr_bgr = arr_rgb[:, :, ::-1].copy()

        batch: Dict[str, torch.Tensor] = {}
        if "rgb" in streams:
            batch["rgb"] = tri_mod.make_rgb_input(arr_rgb_uint8).unsqueeze(0).to(DEVICE)
        if "wavelet" in streams:
            wav = tri_mod.make_wavelet_input(
                arr_bgr=arr_bgr,
                wavelet=cfg["wavelet"],
                level=cfg["wavelet_level"],
                wavelet_type=cfg["wavelet_type"],
                wavelet_gray=cfg["wavelet_gray"],
                subband=cfg["subband"],
                robust=(not cfg["no_robust_norm"]),
            )
            wav = np.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=0.0)
            batch["wavelet"] = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(DEVICE)
        if "dct" in streams:
            dct = tri_mod.make_dct_input(
                arr_bgr=arr_bgr,
                freq_in=cfg["freq_in"],
                block_energy=cfg["block_energy"],
            )
            dct = np.nan_to_num(dct, nan=0.0, posinf=1.0, neginf=0.0)
            batch["dct"] = torch.from_numpy(dct.astype(np.float32)).unsqueeze(0).to(DEVICE)
        if "semantic" in streams:
            batch["semantic"] = clip_processor(images=pil_img, return_tensors="pt")["pixel_values"].to(DEVICE)
        return batch, raw_rgb

    def run_cam(image_path: str) -> Dict[str, Any]:
        batch, raw_rgb = preprocess(image_path)

        rgb_cam_hook = None
        wav_cam_hook = None

        if "rgb" in model.branches:
            rgb_cam_hook = BranchFeatMapGradCAM(model, model.branches["rgb"], "rgb")
        if "wavelet" in model.branches:
            wav_cam_hook = BranchFeatMapGradCAM(model, model.branches["wavelet"], "wavelet")

        out = None
        logits = None
        score = None

        try:
            model.zero_grad(set_to_none=True)

            out = model(batch)
            logits = out["logits"]
            score = select_score_from_logits(logits)

            score.backward()

            cam_candidates = {}

            if rgb_cam_hook is not None:
                cam_candidates["rgb"] = rgb_cam_hook.get_cam()

            if wav_cam_hook is not None:
                cam_candidates["wavelet"] = wav_cam_hook.get_cam()

            if not cam_candidates:
                raise RuntimeError("No RGB/Wavelet feat_map CAM could be computed.")

            show_mode = cfg.get("tri_query_stream_to_show", "rgb")
            if show_mode == "avg":
                cams = [
                    cv2.resize(c, (cfg["image_size"], cfg["image_size"]), interpolation=cv2.INTER_LINEAR)
                    for c in cam_candidates.values()
                ]
                cam = np.mean(np.stack(cams, axis=0), axis=0)
                cam = normalize_cam(cam)
            else:
                if show_mode not in cam_candidates:
                    fallback = "rgb" if "rgb" in cam_candidates else list(cam_candidates.keys())[0]
                    show_mode = fallback
                cam = cam_candidates[show_mode]

            heatmap, overlay = overlay_cam_on_rgb(raw_rgb, cam)
            pred = logits_to_target_index(logits)
            prob_real, prob_fake = logits_to_probs(logits)

            return {
                "raw_rgb": raw_rgb,
                "cam": cam,
                "heatmap": heatmap,
                "overlay": overlay,
                "pred": pred,
                "prob_real": prob_real,
                "prob_fake": prob_fake,
                "logits": logits.detach().cpu(),
                "note": f"feat_map Grad-CAM | available={list(cam_candidates.keys())} | shown={show_mode}",
                "extra_cams": cam_candidates,
            }

        finally:
            if rgb_cam_hook is not None:
                rgb_cam_hook.remove()
            if wav_cam_hook is not None:
                wav_cam_hook.remove()

            del score
            del logits
            del out
            del batch

            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    return ModelRunner(cfg["export_key"], cfg["image_size"], run_cam, model)


# =========================================================
# Builder dispatcher
# =========================================================
def build_runner(cfg: Dict[str, Any]) -> ModelRunner:
    mt = cfg["model_type"]
    if mt == "single":
        return build_xception_runner(cfg)
    if mt == "f3net":
        return build_f3net_runner(cfg)
    if mt == "m2tr":
        return build_m2tr_runner(cfg)
    if mt == "tri_stream":
        return build_tri_stream_runner(cfg)
    raise ValueError(f"Unknown model_type: {mt}")


# =========================================================
# Export helpers
# =========================================================
def save_result_bundle(out_dir: str, stem: str, result: Dict[str, Any]):
    ensure_dir(out_dir)

    pred = result.get("pred", None)
    if pred == 0:
        pred_name = "real"
        conf = result.get("prob_real", None)
    elif pred == 1:
        pred_name = "fake"
        conf = result.get("prob_fake", None)
    else:
        pred_name = "unknown"
        conf = None

    if conf is not None:
        stem_with_pred = f"{stem}_pred-{pred_name}_conf-{conf:.4f}"
    else:
        stem_with_pred = f"{stem}_pred-{pred_name}"

    save_rgb_image(os.path.join(out_dir, f"{stem_with_pred}_raw.png"), result["raw_rgb"])
    save_rgb_image(os.path.join(out_dir, f"{stem_with_pred}_overlay.png"), result["overlay"])

    extra_cams = result.get("extra_cams")
    if isinstance(extra_cams, dict):
        for name, cam in extra_cams.items():
            _, overlay = overlay_cam_on_rgb(result["raw_rgb"], cam)
            save_rgb_image(os.path.join(out_dir, f"{stem_with_pred}_{name}_overlay.png"), overlay)


def is_strong_sample(result: Dict[str, Any], real_thr: float, fake_thr: float) -> bool:
    pred = result.get("pred", None)
    prob_real = float(result.get("prob_real", 0.0) or 0.0)
    prob_fake = float(result.get("prob_fake", 0.0) or 0.0)

    if pred == 0:
        return prob_real >= real_thr
    if pred == 1:
        return prob_fake >= fake_thr
    return False


def gt_label_to_int(label_name: str) -> Optional[int]:
    if label_name == "real":
        return 0
    if label_name == "fake":
        return 1
    return None


def is_misclassified_sample(label_name: str, result: Dict[str, Any]) -> bool:
    gt = gt_label_to_int(label_name)
    pred = result.get("pred", None)

    if gt is None or pred is None:
        return False

    return int(pred) != int(gt)


def export_for_direct_images(runners: List[ModelRunner], image_paths: List[str], args):
    for image_path in image_paths:
        stem = Path(image_path).stem
        for runner in runners:
            print(f"[RUN] model={runner.export_key} | image={image_path}")
            try:
                result = runner.run_cam(image_path)
                if args.save_only_strong and not is_strong_sample(result, args.strong_real_threshold, args.strong_fake_threshold):
                    print(
                        f"[SKIP] weak-confidence sample | model={runner.export_key} | "
                        f"pred={result.get('pred')} | prob_real={result.get('prob_real', 0.0):.4f} | "
                        f"prob_fake={result.get('prob_fake', 0.0):.4f}"
                    )
                    continue
                out_dir = os.path.join(EXPORT_ROOT, "direct", runner.export_key)
                save_result_bundle(out_dir, stem, result)
            except Exception as e:
                print(f"[FAIL] model={runner.export_key} | image={image_path} | err={e}")


def export_for_datasets(runners: List[ModelRunner], datasets: List[str], num_per_class: int, args):
    for ds_name in datasets:
        if ds_name not in TEST_DATASETS:
            print(f"[WARN] Unknown dataset: {ds_name}")
            continue

        roots = collect_roots_for_dataset(ds_name, TEST_DATASETS[ds_name])
        per_label_paths = {
            "real": sample_images_from_roots(roots.get("real", []), num_per_class),
            "fake": sample_images_from_roots(roots.get("fake", []), num_per_class),
        }

        for label_name, paths in per_label_paths.items():
            print(f"\n[DATASET] {ds_name} | label={label_name} | n={len(paths)}")
            for idx, image_path in enumerate(paths, start=1):
                stem = f"img_{idx:02d}"
                for runner in runners:
                    print(f"[RUN] dataset={ds_name} | label={label_name} | model={runner.export_key} | image={image_path}")
                    try:
                        result = runner.run_cam(image_path)
                        out_dir = os.path.join(EXPORT_ROOT, ds_name.lower(), runner.export_key, label_name)
                        save_result_bundle(out_dir, stem, result)
                    except Exception as e:
                        print(f"[FAIL] dataset={ds_name} | label={label_name} | model={runner.export_key} | err={e}")

def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def export_for_datasets_single_runner(runner: ModelRunner, datasets: List[str], num_per_class: int, args):
    selected_labels = set(args.labels)

    for ds_name in datasets:
        if ds_name not in TEST_DATASETS:
            print(f"[WARN] Unknown dataset: {ds_name}")
            continue

        roots = collect_roots_for_dataset(ds_name, TEST_DATASETS[ds_name])
        per_label_paths = {
            "real": sample_images_from_roots(roots.get("real", []), num_per_class),
            "fake": sample_images_from_roots(roots.get("fake", []), num_per_class),
        }

        for label_name, paths in per_label_paths.items():
            if label_name not in selected_labels:
                print(f"[SKIP-LABEL] dataset={ds_name} | label={label_name} not selected")
                continue

            print(f"\n[DATASET] {ds_name} | label={label_name} | model={runner.export_key} | n={len(paths)}")
            for idx, image_path in enumerate(paths, start=1):
                stem = f"img_{idx:02d}"
                print(f"[RUN] dataset={ds_name} | label={label_name} | model={runner.export_key} | image={image_path}")
                try:
                    result = runner.run_cam(image_path)

                    if args.save_only_misclassified and not is_misclassified_sample(label_name, result):
                        print(
                            f"[SKIP] correctly-classified sample | dataset={ds_name} | label={label_name} | model={runner.export_key} | "
                            f"pred={result.get('pred')} | prob_real={result.get('prob_real', 0.0):.4f} | "
                            f"prob_fake={result.get('prob_fake', 0.0):.4f}"
                        )
                        continue

                    if args.save_only_strong and not is_strong_sample(result, args.strong_real_threshold, args.strong_fake_threshold):
                        print(
                            f"[SKIP] weak-confidence sample | dataset={ds_name} | label={label_name} | model={runner.export_key} | "
                            f"pred={result.get('pred')} | prob_real={result.get('prob_real', 0.0):.4f} | "
                            f"prob_fake={result.get('prob_fake', 0.0):.4f}"
                        )
                        continue

                    out_dir = os.path.join(EXPORT_ROOT, ds_name.lower(), runner.export_key, label_name)
                    save_result_bundle(out_dir, stem, result)
                except Exception as e:
                    print(f"[FAIL] dataset={ds_name} | label={label_name} | model={runner.export_key} | err={e}")
                finally:
                    free_cuda()


# =========================================================
# Main
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(TEST_DATASETS.keys()))
    ap.add_argument("--num-per-class", type=int, default=5)
    ap.add_argument("--only-model", nargs="*", default=None,
                    help="export_key filter, e.g. --only-model ours xception")
    ap.add_argument("--image", nargs="*", default=None,
                    help="direct image paths instead of dataset sampling")
    ap.add_argument("--export-root", type=str, default=EXPORT_ROOT)
    ap.add_argument("--save-only-strong", action="store_true",
                    help="save only strong-confidence samples based on predicted label")
    ap.add_argument("--strong-real-threshold", type=float, default=0.95,
                    help="minimum prob_real for saving pred-real samples")
    ap.add_argument("--strong-fake-threshold", type=float, default=0.95,
                    help="minimum prob_fake for saving pred-fake samples")
    ap.add_argument("--save-only-misclassified", action="store_true",
                help="save only misclassified samples: GT real predicted fake, or GT fake predicted real")
    ap.add_argument("--pred-fake-threshold", type=float, default=0.5,
                help="prediction threshold for fake class probability")
    ap.add_argument(
        "--labels",
        nargs="*",
        default=["real", "fake"],
        choices=["real", "fake"],
        help="GT labels to export Grad-CAM for. Use: --labels real, --labels fake, or --labels real fake"
    )
    return ap.parse_args()


def main():
    global EXPORT_ROOT, PRED_FAKE_THRESHOLD
    args = parse_args()
    EXPORT_ROOT = args.export_root
    ensure_dir(EXPORT_ROOT)
    PRED_FAKE_THRESHOLD = args.pred_fake_threshold

    set_seed(SEED)
    add_sys_path(BASE_DIR)
    add_sys_path(RGB_ROOT)
    add_sys_path(TRI_STREAM_ROOT)
    add_sys_path(M2TR_ROOT)
    add_sys_path(F3NET_ROOT)

    selected_cfgs = []
    for cfg in MODEL_CONFIGS:
        if not cfg.get("enable", True):
            continue
        if args.only_model and cfg["export_key"] not in args.only_model:
            continue
        selected_cfgs.append(cfg)

    if not selected_cfgs:
        raise RuntimeError("No models selected. Check MODEL_CONFIGS / --only-model.")

    for cfg in selected_cfgs:
        print(f"[BUILD] {cfg['export_key']} ({cfg['model_type']})")
        runner = build_runner(cfg)

        try:
            if args.image:
                export_for_direct_images([runner], args.image, args)
            else:
                export_for_datasets_single_runner(runner, args.datasets, args.num_per_class, args)
        finally:
            # 모델 해제
            if hasattr(runner, "model"):
                try:
                    runner.model.cpu()
                except Exception:
                    pass
            del runner
            free_cuda()

    print(f"\nDone. Results saved under: {EXPORT_ROOT}")

if __name__ == "__main__":
    main()
