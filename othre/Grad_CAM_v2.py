#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset-based Grad-CAM++ / Token-CAM++ export script for paper figure generation.

Exports:
  <EXPORT_ROOT>/<dataset_name_lower>/
    origin/real/img_1.png ... img_5.png
    origin/fake/img_1.png ... img_5.png
    xcep/real/...
    xcep/fake/...
    f3net_fad/real/...
    f3net_fad/fake/...
    m2tr/real/...
    m2tr/fake/...
    ours/real/rgb/...
    ours/real/wavelet/...
    ours/fake/rgb/...
    ours/fake/wavelet/...

Supported model families:
- Xception            -> final conv Grad-CAM++
- F3Net-FAD           -> final conv Grad-CAM++
- M2TR                -> final token attribution if available, else final conv Grad-CAM++
- RGB+Wavelet+Semantic (Ours) -> post-cross-attention token attribution (RGB/Wavelet)

Important:
- For Ours, Tri_stream/train.py forward() must return:
    query_tokens, kv_tokens, attn_out
- For M2TR token-level attribution, model forward should return:
    {"logits": ..., "final_tokens": ...}
  If not, this script falls back to final conv Grad-CAM++.
"""

import os
import sys
import glob
import random
import types
import inspect
import importlib.util

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# =========================================================
# Paths
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

EXPORT_ROOT = os.path.join(BASE_DIR, "gradcam_export2")
os.makedirs(EXPORT_ROOT, exist_ok=True)


# =========================================================
# General utils
# =========================================================
def add_sys_path(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)


add_sys_path(BASE_DIR)
add_sys_path(RGB_ROOT)
add_sys_path(TRI_STREAM_ROOT)
add_sys_path(os.path.join(TRI_STREAM_ROOT, "models"))
add_sys_path(M2TR_ROOT)
add_sys_path(F3NET_ROOT)


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def remove_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_checkpoint(model, checkpoint_path, strict=False):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)}")

    state_dict = remove_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    print(f"[LOAD] {os.path.basename(checkpoint_path)}")
    print(f"  Missing keys   : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    return model


def find_last_conv(module: nn.Module):
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def normalize_map(x: np.ndarray):
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def overlay_map_on_image(raw_img_rgb: np.ndarray, cam_map: np.ndarray):
    h, w = raw_img_rgb.shape[:2]
    cam_map = cv2.resize(cam_map.astype(np.float32), (w, h))
    cam_map = normalize_map(cam_map)

    heatmap = np.uint8(255 * cam_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.clip(0.5 * raw_img_rgb + 0.5 * heatmap, 0, 255).astype(np.uint8)
    return heatmap, overlay


def build_transform(image_size, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_image_for_model(image_path, image_size, mean=None, std=None):
    img = Image.open(image_path).convert("RGB")
    raw_img = np.array(img.resize((image_size, image_size)))
    x = build_transform(image_size, mean, std)(img).unsqueeze(0)
    return x, raw_img


def save_rgb_image(path: str, img_rgb_uint8: np.ndarray):
    cv2.imwrite(path, cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR))


def unwrap_model_output(output):
    if isinstance(output, dict):
        if "logits" in output:
            output = output["logits"]
        else:
            raise ValueError(f"dict output has no 'logits': {list(output.keys())}")
    elif isinstance(output, (tuple, list)):
        tensor_candidates = [x for x in output if torch.is_tensor(x)]
        if len(tensor_candidates) == 0:
            raise ValueError("No tensor output found in tuple/list output.")
        output = tensor_candidates[-1]

    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported output type: {type(output)}")

    return output


def get_target_class_from_output(output, mode="pred"):
    output = unwrap_model_output(output)

    if output.ndim != 2:
        raise ValueError(f"Unsupported output shape: {tuple(output.shape)}")

    if output.shape[1] > 1:
        prob = torch.softmax(output, dim=1)[0]
        pred_idx = int(prob.argmax().item())
        pred_score = float(prob[pred_idx].item())

        if mode == "fake":
            class_idx = min(1, output.shape[1] - 1)
        else:
            class_idx = pred_idx

        score = output[:, class_idx]
        return class_idx, score, pred_idx, pred_score

    else:
        logit = output[:, 0]
        prob_fake = torch.sigmoid(logit)[0].item()
        pred_idx = 1 if prob_fake >= 0.5 else 0
        pred_score = float(prob_fake)
        class_idx = 0
        score = output[:, 0]
        return class_idx, score, pred_idx, pred_score


# =========================================================
# Device / seed
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# TEST DATASETS
# =========================================================
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
    "DFD": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences"
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences"
        ],
    },
    "DeepfakeTIMIT": {
        "real": [],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality"
        ],
    },
    "WildDeepfake": {
        "root": "/home/oem/deepfake/hdd_5TB/WildDeepfake",
        "splits": ["train", "test"],
    },
}


# =========================================================
# MODEL CONFIGS
# =========================================================
MODEL_CONFIGS = [
    {
        "enable": True,
        "model_type": "single",
        "model_name": "xception",
        "export_key": "xcep",
        "checkpoint_path": "/home/oem/deepfake/Ourmethod/gradcam_export/_ckpt/xception_best.pth",
        "image_size": 224,
        "num_classes": 2,
        "target_class_mode": "pred",
    },
    {
        "enable": True,
        "model_type": "f3net",
        "model_name": "f3net_fad",
        "export_key": "f3net_fad",
        "checkpoint_path": "/home/oem/deepfake/F3Net/checkpoints/FAD/F3Net_last.pth",
        "image_size": 299,
        "num_classes": 1,
        "target_class_mode": "pred",
    },
    {
        "enable": True,
        "model_type": "m2tr",
        "model_name": "m2tr_ffpp",
        "export_key": "m2tr",
        "checkpoint_path": "/home/oem/deepfake/Ourmethod/comparison/_ckpt/m2tr/checkpoints/M2TR_FFPPVideo_epoch_00010.pyth",
        "config_path": "m2tr_ffpp_video.yaml",
        "image_size": 320,
        "num_classes": 2,
        "target_class_mode": "pred",
    },
    {
        "enable": True,
        "model_type": "tri_stream",
        "model_name": "rgb_wavelet_semantic",
        "export_key": "ours",
        "checkpoint_path": "/home/oem/deepfake/Ourmethod/Tri_stream/_ckpt/rgb_wavelet_semantic/v1/best_tri_rgb_wavelet_semantic.pth",
        "image_size": 224,
        "target_class_mode": "pred",

        "streams": "rgb,wavelet,semantic",
        "img_size": 224,

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

        "dct_mode": "block",
        "freq_in": "ycbcr",
        "block_energy": "ac",

        "clip_backbone": "openai/clip-vit-base-patch32",
        "finetune_clip": False,
        "resnet_pretrained_wavelet": False,
    },
]


# =========================================================
# Dataset sampling utils
# =========================================================
def collect_frames_from_video_dir(video_dir):
    frames = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        frames.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(frames)


def collect_dataset_roots(ds_name, cfg):
    if ds_name == "WildDeepfake":
        real_roots, fake_roots = [], []
        for split in cfg["splits"]:
            split_dir = os.path.join(cfg["root"], split)
            if not os.path.isdir(split_dir):
                continue
            for sub in os.listdir(split_dir):
                base = os.path.join(split_dir, sub)
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


def sample_frames_from_roots(roots, max_samples=5):
    candidates = []

    for root in roots:
        if not os.path.isdir(root):
            continue

        for vid in sorted(os.listdir(root)):
            vid_dir = os.path.join(root, vid)
            if not os.path.isdir(vid_dir):
                continue

            frames = collect_frames_from_video_dir(vid_dir)
            if len(frames) == 0:
                continue

            frame = frames[len(frames) // 2]
            candidates.append(frame)

    return candidates[:max_samples]


def make_export_dirs(dataset_name, model_keys):
    dataset_dir = os.path.join(EXPORT_ROOT, dataset_name.lower())
    ensure_dir(dataset_dir)

    for cls in ["real", "fake"]:
        ensure_dir(os.path.join(dataset_dir, "origin", cls))

    for key in model_keys:
        if key != "ours":
            for cls in ["real", "fake"]:
                ensure_dir(os.path.join(dataset_dir, key, cls))

    for cls in ["real", "fake"]:
        ensure_dir(os.path.join(dataset_dir, "ours", cls, "rgb"))
        ensure_dir(os.path.join(dataset_dir, "ours", cls, "wavelet"))

    return dataset_dir


def save_origin_image(image_path, save_path, image_size=224):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    img.save(save_path)


# =========================================================
# Grad-CAM++ for spatial feature maps
# =========================================================
class GradCAMPlusPlus:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None

        self.h_forward = target_module.register_forward_hook(self._forward_hook)
        self.h_backward = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        if isinstance(grad_output, (tuple, list)):
            self.gradients = grad_output[0]
        else:
            self.gradients = grad_output

    def generate_from_existing_graph(self):
        if self.activations is None:
            raise RuntimeError("No activations captured.")
        if self.gradients is None:
            raise RuntimeError("No gradients captured.")

        acts = self.activations[0]      # [C,H,W]
        grads = self.gradients[0]       # [C,H,W]

        if acts.ndim != 3:
            raise ValueError(f"GradCAM++ expects [C,H,W], got {acts.shape}")

        grads_2 = grads ** 2
        grads_3 = grads_2 * grads

        denom = 2.0 * grads_2 + torch.sum(acts * grads_3, dim=(1, 2), keepdim=True)
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))

        alpha = grads_2 / (denom + 1e-8)
        positive_grads = F.relu(grads)
        weights = torch.sum(alpha * positive_grads, dim=(1, 2), keepdim=True)

        cam = torch.sum(weights * acts, dim=0)
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy().astype(np.float32)
        cam = normalize_map(cam)
        return cam

    def remove(self):
        self.h_forward.remove()
        self.h_backward.remove()


# =========================================================
# Token-CAM++ for token outputs [B, N, D]
# =========================================================
class TokenImportanceCAMPlusPlus:
    """
    Grad-CAM++-style token attribution.
    """

    def __init__(self):
        self.tensor = None
        self.grad = None

    def attach(self, token_tensor):
        self.tensor = token_tensor
        self.grad = None

        def _save_grad(g):
            self.grad = g

        token_tensor.register_hook(_save_grad)

    def generate(self):
        if self.tensor is None:
            raise RuntimeError("Token tensor not attached.")
        if self.grad is None:
            raise RuntimeError("No gradients captured for token tensor.")

        tok = self.tensor[0]   # [N, D]
        grd = self.grad[0]     # [N, D]

        grd2 = grd ** 2
        grd3 = grd2 * grd

        denom = 2.0 * grd2 + tok * grd3
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))

        alpha = grd2 / (denom + 1e-8)
        positive_grads = F.relu(grd)

        weights = alpha * positive_grads
        score = torch.sum(weights * tok, dim=-1)

        score = F.relu(score)
        score = score.detach().cpu().numpy().astype(np.float32)
        score = normalize_map(score)
        return score


def tokens_to_spatial_map(token_scores, h_tokens, w_tokens):
    if len(token_scores) != h_tokens * w_tokens:
        raise ValueError(
            f"Token count mismatch: len={len(token_scores)} vs grid={h_tokens}x{w_tokens}"
        )
    m = token_scores.reshape(h_tokens, w_tokens)
    return normalize_map(m)


# =========================================================
# Xception
# =========================================================
def build_xception_model(num_classes):
    add_sys_path(RGB_ROOT)

    xception_py = os.path.join(RGB_ROOT, "Xception", "xception.py")
    if not os.path.isfile(xception_py):
        raise FileNotFoundError(f"Xception file not found: {xception_py}")

    xception_mod = load_module_from_path("xception_mod_for_export", xception_py)

    sig = inspect.signature(xception_mod.xception)
    kwargs = {}
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = num_classes
    if "use_cbam" in sig.parameters:
        kwargs["use_cbam"] = False
    if "cbam_kernel" in sig.parameters:
        kwargs["cbam_kernel"] = 7
    if "cbam_reduction" in sig.parameters:
        kwargs["cbam_reduction"] = 16

    model = xception_mod.xception(**kwargs)
    return model


def init_xception_runner(cfg):
    model = build_xception_model(cfg["num_classes"])
    model = load_checkpoint(model, cfg["checkpoint_path"], strict=False)
    model = model.to(DEVICE).eval()

    target_layer = find_last_conv(model)
    if target_layer is None:
        raise ValueError("Could not find last conv for Xception.")

    cam = GradCAMPlusPlus(model, target_layer)

    def run_one(image_path, save_path):
        x, raw_img = load_image_for_model(image_path, cfg["image_size"])
        x = x.to(DEVICE)

        model.zero_grad(set_to_none=True)
        output = model(x)
        _, score, _, _ = get_target_class_from_output(
            output, mode=cfg.get("target_class_mode", "pred")
        )
        score.backward(retain_graph=True)

        cam_map = cam.generate_from_existing_graph()
        _, overlay = overlay_map_on_image(raw_img, cam_map)
        save_rgb_image(save_path, overlay)

    def cleanup():
        cam.remove()

    return run_one, cleanup


# =========================================================
# F3Net-FAD
# =========================================================
def build_f3net_fad_model(num_classes):
    model_py = os.path.join(F3NET_ROOT, "models.py")
    if not os.path.isfile(model_py):
        raise FileNotFoundError(f"F3Net models.py not found: {model_py}")

    add_sys_path(F3NET_ROOT)
    f3net_mod = load_module_from_path("f3net_model_for_export", model_py)

    model = f3net_mod.F3Net(
        num_classes=num_classes,
        img_width=299,
        img_height=299,
        mode="FAD",
    )
    return model


def init_f3net_runner(cfg):
    model = build_f3net_fad_model(cfg["num_classes"])
    model = load_checkpoint(model, cfg["checkpoint_path"], strict=False)
    model = model.to(DEVICE).eval()

    if not hasattr(model, "FAD_xcep"):
        raise ValueError("F3Net model has no FAD_xcep.")

    target_layer = find_last_conv(model.FAD_xcep)
    if target_layer is None:
        raise ValueError("Could not find target conv in F3Net FAD_xcep.")

    cam = GradCAMPlusPlus(model, target_layer)

    def run_one(image_path, save_path):
        x, raw_img = load_image_for_model(image_path, cfg["image_size"])
        x = x.to(DEVICE)

        model.zero_grad(set_to_none=True)
        output = model(x)

        if isinstance(output, (tuple, list)) and len(output) >= 2:
            logits = output[1]
        else:
            logits = unwrap_model_output(output)

        _, score, _, _ = get_target_class_from_output(
            logits, mode=cfg.get("target_class_mode", "pred")
        )
        score.backward(retain_graph=True)

        cam_map = cam.generate_from_existing_graph()
        _, overlay = overlay_map_on_image(raw_img, cam_map)
        save_rgb_image(save_path, overlay)

    def cleanup():
        cam.remove()

    return run_one, cleanup


# =========================================================
# M2TR
# =========================================================
def load_m2tr_cfg(config_path):
    add_sys_path(M2TR_ROOT)
    from tools.utils import load_config as m2tr_load_config

    class Args:
        cfg_file = config_path
        shard_id = 0
        base_lr = None

    prev_cwd = os.getcwd()
    try:
        os.chdir(M2TR_ROOT)
        cfg = m2tr_load_config(Args)
    finally:
        os.chdir(prev_cwd)

    cfg["NUM_GPUS"] = 1
    cfg["TEST"]["ENABLE"] = True
    return cfg


def build_m2tr_model(config_path):
    add_sys_path(M2TR_ROOT)
    from M2TR.utils.build_helper import build_model as m2tr_build_model

    cfg = load_m2tr_cfg(config_path)
    model = m2tr_build_model(cfg)
    return model, cfg


def init_m2tr_runner(cfg):
    model, _ = build_m2tr_model(cfg["config_path"])
    model = load_checkpoint(model, cfg["checkpoint_path"], strict=False)
    model = model.to(DEVICE).eval()

    token_cam = TokenImportanceCAMPlusPlus()

    target_layer = None
    if hasattr(model, "backbone"):
        target_layer = find_last_conv(model.backbone)
    if target_layer is None:
        target_layer = find_last_conv(model)

    cam = GradCAMPlusPlus(model, target_layer) if target_layer is not None else None

    def run_one(image_path, save_path):
        x, raw_img = load_image_for_model(image_path, cfg["image_size"])
        x = x.to(DEVICE)

        model.zero_grad(set_to_none=True)
        out = model({"img": x})

        if isinstance(out, dict) and "final_tokens" in out:
            token_cam.attach(out["final_tokens"])
            _, score, _, _ = get_target_class_from_output(
                out["logits"], mode=cfg.get("target_class_mode", "pred")
            )
            score.backward(retain_graph=True)

            token_scores = token_cam.generate()
            n = len(token_scores)
            h = w = int(np.sqrt(n))
            spatial = tokens_to_spatial_map(token_scores, h, w)
            _, overlay = overlay_map_on_image(raw_img, spatial)
            save_rgb_image(save_path, overlay)

        else:
            _, score, _, _ = get_target_class_from_output(
                out, mode=cfg.get("target_class_mode", "pred")
            )
            score.backward(retain_graph=True)

            if cam is None:
                raise RuntimeError("No available final_tokens or conv target for M2TR.")
            cam_map = cam.generate_from_existing_graph()
            _, overlay = overlay_map_on_image(raw_img, cam_map)
            save_rgb_image(save_path, overlay)

    def cleanup():
        if cam is not None:
            cam.remove()

    return run_one, cleanup


# =========================================================
# Tri-stream (Ours)
# =========================================================
def build_tri_args_from_cfg(cfg):
    args = types.SimpleNamespace()

    args.img_size = cfg.get("img_size", 224)

    args.embed_dim = cfg.get("embed_dim", 256)
    args.hidden_dim = cfg.get("hidden_dim", 512)
    args.num_heads = cfg.get("num_heads", 8)
    args.dropout = cfg.get("dropout", 0.2)

    args.wavelet = cfg.get("wavelet", "sym4")
    args.wavelet_level = cfg.get("wavelet_level", 2)
    args.wavelet_type = cfg.get("wavelet_type", "swt")
    args.subband = cfg.get("subband", "ll_energy")
    args.wavelet_gray = cfg.get("wavelet_gray", False)
    args.no_robust_norm = cfg.get("no_robust_norm", False)

    args.dct_mode = cfg.get("dct_mode", "block")
    args.freq_in = cfg.get("freq_in", "ycbcr")
    args.block_energy = cfg.get("block_energy", "ac")

    args.clip_backbone = cfg.get("clip_backbone", "openai/clip-vit-base-patch32")
    args.finetune_clip = cfg.get("finetune_clip", False)
    args.resnet_pretrained_wavelet = cfg.get("resnet_pretrained_wavelet", False)

    args.rgb_ckpt = None
    args.wavelet_ckpt = None
    args.dct_ckpt = None
    args.freeze_rgb = False
    args.freeze_wavelet = False
    args.freeze_dct = False
    args.freeze_semantic = False

    return args


def import_tri_train_module():
    cand1 = os.path.join(TRI_STREAM_ROOT, "train.py")
    cand2 = os.path.join(TRI_STREAM_ROOT, "train_tri.py")

    if os.path.isfile(cand1):
        target = cand1
        module_name = "tri_stream_train"
    elif os.path.isfile(cand2):
        target = cand2
        module_name = "tri_stream_train"
    else:
        raise FileNotFoundError(f"Tri-stream train file not found in {TRI_STREAM_ROOT}")

    add_sys_path(TRI_STREAM_ROOT)
    add_sys_path(os.path.join(TRI_STREAM_ROOT, "models"))

    tri_train = load_module_from_path(module_name, target)
    return tri_train


def init_tri_stream_runner(cfg):
    tri_train = import_tri_train_module()
    args = build_tri_args_from_cfg(cfg)
    streams = tri_train.parse_streams(cfg["streams"])

    model = tri_train.build_model(args, streams)
    model = load_checkpoint(model, cfg["checkpoint_path"], strict=False)
    model = model.to(DEVICE).eval()

    clip_processor = None
    if "semantic" in streams:
        from transformers import CLIPImageProcessor
        clip_processor = CLIPImageProcessor.from_pretrained(args.clip_backbone)

    token_cam = TokenImportanceCAMPlusPlus()

    def make_input(image_path):
        img = Image.open(image_path).convert("RGB")
        resize = transforms.Resize((args.img_size, args.img_size))
        img_resized = resize(img)

        arr_rgb = np.array(img_resized).astype(np.float32)
        arr_rgb_uint8 = arr_rgb.astype(np.uint8)
        arr_bgr = arr_rgb[:, :, ::-1].copy()

        item = {}

        if "rgb" in streams:
            item["rgb"] = tri_train.make_rgb_input(arr_rgb_uint8).unsqueeze(0)

        if "wavelet" in streams:
            wav = tri_train.make_wavelet_input(
                arr_bgr=arr_bgr,
                wavelet=args.wavelet,
                level=args.wavelet_level,
                wavelet_type=args.wavelet_type,
                wavelet_gray=args.wavelet_gray,
                subband=args.subband,
                robust=(not args.no_robust_norm),
            )
            wav = np.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=0.0)
            item["wavelet"] = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)

        if "dct" in streams:
            dct = tri_train.make_dct_input(
                arr_bgr=arr_bgr,
                freq_in=args.freq_in,
                block_energy=args.block_energy,
            )
            dct = np.nan_to_num(dct, nan=0.0, posinf=1.0, neginf=0.0)
            item["dct"] = torch.from_numpy(dct.astype(np.float32)).unsqueeze(0)

        if "semantic" in streams:
            item["semantic"] = clip_processor(
                images=img,
                return_tensors="pt",
            )["pixel_values"]

        raw_img = np.array(img_resized).astype(np.uint8)
        return item, raw_img

    def run_one(image_path):
        batch, raw_img = make_input(image_path)
        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        model.zero_grad(set_to_none=True)
        out = model(batch)

        if "attn_out" not in out:
            raise KeyError(
                "Ours requires attn_out in model forward output. "
                "Please modify Tri_stream/train.py forward() to return attn_out."
            )

        attn_out = out["attn_out"]   # [B, Nq, D]
        token_cam.attach(attn_out)

        _, score, _, _ = get_target_class_from_output(
            out["logits"], mode=cfg.get("target_class_mode", "pred")
        )
        score.backward(retain_graph=True)

        token_scores = token_cam.generate()

        result = {}
        start = 0

        rgb_n = 0
        wavelet_n = 0

        if "rgb" in model.branches:
            rgb_feat = model.branches["rgb"](batch["rgb"])
            rgb_tok = model._make_tokens("rgb", rgb_feat)
            rgb_n = rgb_tok.shape[1]

        if "wavelet" in model.branches:
            wav_feat = model.branches["wavelet"](batch["wavelet"])
            wav_tok = model._make_tokens("wavelet", wav_feat)
            wavelet_n = wav_tok.shape[1]

        for s in out["main_streams"]:
            if s == "rgb" and rgb_n > 0:
                part = token_scores[start:start + rgb_n]
                h = w = int(np.sqrt(rgb_n))
                spatial = tokens_to_spatial_map(part, h, w)
                _, overlay = overlay_map_on_image(raw_img, spatial)
                result["rgb"] = overlay
                start += rgb_n

            elif s == "wavelet" and wavelet_n > 0:
                part = token_scores[start:start + wavelet_n]
                h = w = int(np.sqrt(wavelet_n))
                spatial = tokens_to_spatial_map(part, h, w)
                _, overlay = overlay_map_on_image(raw_img, spatial)
                result["wavelet"] = overlay
                start += wavelet_n

        return result

    def cleanup():
        pass

    return run_one, cleanup


# =========================================================
# Runner factory
# =========================================================
def build_model_runners():
    enabled_cfgs = [cfg for cfg in MODEL_CONFIGS if cfg.get("enable", True)]
    runners = []

    for cfg in enabled_cfgs:
        ckpt = cfg["checkpoint_path"]
        if not os.path.exists(ckpt):
            print(f"[SKIP] checkpoint not found: {ckpt}")
            continue

        print("\n" + "=" * 100)
        print(f"[INIT] {cfg['export_key']} ({cfg['model_type']})")
        print("=" * 100)

        if cfg["model_type"] == "single":
            run_one, cleanup = init_xception_runner(cfg)
        elif cfg["model_type"] == "f3net":
            run_one, cleanup = init_f3net_runner(cfg)
        elif cfg["model_type"] == "m2tr":
            run_one, cleanup = init_m2tr_runner(cfg)
        elif cfg["model_type"] == "tri_stream":
            run_one, cleanup = init_tri_stream_runner(cfg)
        else:
            print(f"[SKIP] unknown model_type: {cfg['model_type']}")
            continue

        runners.append({
            "cfg": cfg,
            "run_one": run_one,
            "cleanup": cleanup,
        })

    return runners


# =========================================================
# Export loop
# =========================================================
def export_dataset_samples():
    runners = build_model_runners()
    model_keys = [r["cfg"]["export_key"] for r in runners]

    print("\n" + "=" * 100)
    print(" Export start ")
    print("=" * 100)
    print(f"Device     : {DEVICE}")
    print(f"Export root: {EXPORT_ROOT}")
    print(f"Models     : {model_keys}")
    print("=" * 100)

    for ds_name, ds_cfg in TEST_DATASETS.items():
        print(f"\n=== Exporting {ds_name} ===")

        roots = collect_dataset_roots(ds_name, ds_cfg)
        dataset_dir = make_export_dirs(ds_name, model_keys)

        for cls_name in ["real", "fake"]:
            frame_list = sample_frames_from_roots(roots.get(cls_name, []), max_samples=5)

            if len(frame_list) == 0:
                print(f"[WARN] {ds_name}/{cls_name}: no samples")
                continue

            print(f"[INFO] {ds_name}/{cls_name}: {len(frame_list)} samples")

            for idx, image_path in enumerate(frame_list, start=1):
                file_name = f"img_{idx}.png"

                origin_save = os.path.join(dataset_dir, "origin", cls_name, file_name)
                try:
                    save_origin_image(image_path, origin_save, image_size=224)
                except Exception as e:
                    print(f"[ERROR] origin save failed: {image_path} -> {e}")
                    continue

                for runner in runners:
                    cfg = runner["cfg"]

                    try:
                        if cfg["export_key"] == "ours":
                            result = runner["run_one"](image_path)

                            if "rgb" in result:
                                save_path_rgb = os.path.join(dataset_dir, "ours", cls_name, "rgb", file_name)
                                save_rgb_image(save_path_rgb, result["rgb"])

                            if "wavelet" in result:
                                save_path_wavelet = os.path.join(dataset_dir, "ours", cls_name, "wavelet", file_name)
                                save_rgb_image(save_path_wavelet, result["wavelet"])

                            print(f"[DONE] {ds_name}/{cls_name}/{file_name} -> ours(post-attn++ rgb/wavelet)")

                        else:
                            save_path = os.path.join(dataset_dir, cfg["export_key"], cls_name, file_name)
                            runner["run_one"](image_path, save_path)
                            print(f"[DONE] {ds_name}/{cls_name}/{file_name} -> {cfg['export_key']}")

                    except Exception as e:
                        print(f"[ERROR] {ds_name}/{cls_name}/{file_name} | {cfg['export_key']} -> {e}")

    for runner in runners:
        try:
            runner["cleanup"]()
        except Exception:
            pass


# =========================================================
# Main
# =========================================================
def main():
    export_dataset_samples()
    print("\nDone.")


if __name__ == "__main__":
    main()