# Grad_CAM.py
import os
import csv
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
import timm


# =========================================================
# 0. 기본 설정
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "./test_images"
SAVE_ROOT = "./gradcam_results"

os.makedirs(SAVE_ROOT, exist_ok=True)


# =========================================================
# 1. 실행 설정
# =========================================================
"""
model_type:
- "single" : 단일 backbone 모델
- "fusion" : fusion 모델

single model_name:
- "xception"
- "efficientnet_b4"
- "efficientnet_b7"
- "convnext_tiny"

fusion model_name:
- "fusion_model"

target_class_mode:
- "pred" : 예측 클래스 기준 CAM
- "fake" : fake class 기준 CAM (2-class softmax에서 index=1 가정)
"""

MODEL_CONFIGS = [
    {
        "model_type": "single",
        "model_name": "xception",
        "checkpoint_path": "./checkpoints/xception_best.pth",
        "image_size": 299,
        "num_classes": 2,
        "target_class_mode": "pred",
    },
    {
        "model_type": "single",
        "model_name": "efficientnet_b4",
        "checkpoint_path": "./checkpoints/efficientnet_b4_best.pth",
        "image_size": 380,
        "num_classes": 2,
        "target_class_mode": "pred",
    },
    {
        "model_type": "single",
        "model_name": "efficientnet_b7",
        "checkpoint_path": "./checkpoints/efficientnet_b7_best.pth",
        "image_size": 600,
        "num_classes": 2,
        "target_class_mode": "pred",
    },
    {
        "model_type": "single",
        "model_name": "convnext_tiny",
        "checkpoint_path": "./checkpoints/convnext_tiny_best.pth",
        "image_size": 224,
        "num_classes": 2,
        "target_class_mode": "pred",
    },

    # fusion 모델 예시
    {
        "model_type": "fusion",
        "model_name": "fusion_model",
        "checkpoint_path": "./checkpoints/fusion_best.pth",
        "image_size": 224,
        "num_classes": 2,
        "target_class_mode": "pred",
    },
]


# =========================================================
# 2. 이미지 전처리
# =========================================================
def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image(image_path, image_size):
    pil_img = Image.open(image_path).convert("RGB")
    resized = pil_img.resize((image_size, image_size))
    raw_img = np.array(resized)

    transform = get_transform(image_size)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor, raw_img


# =========================================================
# 3. 공통 유틸
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_last_conv(module):
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def remove_module_prefix(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    return cleaned


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = remove_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[LOAD] {os.path.basename(checkpoint_path)}")
    print(f"  Missing keys   : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    return model


def get_target_class_from_output(output, mode="pred"):
    if output.ndim != 2:
        raise ValueError(f"Unsupported output shape: {output.shape}")

    if output.shape[1] > 1:
        if mode == "fake":
            class_idx = 1
        else:
            class_idx = int(output.argmax(dim=1).item())

        score = output[:, class_idx]
        prob = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(prob))
        pred_score = float(prob[pred_idx])
        return class_idx, score, pred_idx, pred_score

    else:
        class_idx = 0
        score = output[:, 0]
        sig = torch.sigmoid(output)[0].item()
        pred_idx = 1 if sig >= 0.5 else 0
        pred_score = float(sig)
        return class_idx, score, pred_idx, pred_score


def overlay_cam_on_image(img_rgb, cam, alpha=0.4):
    cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(img_rgb * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)
    return heatmap, overlay


def save_three_panel_figure(raw_img, heatmap, overlay, save_path, title_text=""):
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(raw_img)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(heatmap)
    ax2.set_title("Heatmap")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title(title_text)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# 4. Grad-CAM (CNN branch용)
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_from_existing_graph(self):
        if self.activations is None or self.gradients is None:
            raise ValueError("Activations/gradients가 없습니다.")

        act = self.activations[0]
        grad = self.gradients[0]

        if act.ndim != 3 or grad.ndim != 3:
            raise ValueError(
                f"Grad-CAM은 [C,H,W] feature가 필요합니다. got act={act.shape}, grad={grad.shape}"
            )

        weights = grad.mean(dim=(1, 2))

        cam = torch.zeros(act.shape[1:], dtype=torch.float32, device=act.device)
        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

    def remove(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


# =========================================================
# 5. ViT / CLIP semantic branch용 Token-CAM
# =========================================================
class ViTTokenCAM:
    """
    CLIP/ViT semantic branch용 token importance map
    target module output shape: [B, N, D] 가정
    일반적으로 CLS token 포함 시 [CLS + patch tokens]
    """
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.tokens = None
        self.gradients = None

        self.fwd_hook = target_module.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.tokens = output

    def _backward_hook(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if isinstance(grad, (tuple, list)):
            grad = grad[0]
        self.gradients = grad

    def generate_from_existing_graph(self, image_size):
        if self.tokens is None or self.gradients is None:
            raise ValueError("Semantic tokens/gradients가 없습니다.")

        tokens = self.tokens
        grads = self.gradients

        if tokens.ndim != 3 or grads.ndim != 3:
            raise ValueError(
                f"ViT token tensor shape expected [B,N,D], got {tokens.shape}, {grads.shape}"
            )

        tokens = tokens[0]   # [N, D]
        grads = grads[0]     # [N, D]

        # CLS token 제거 가정
        if tokens.shape[0] <= 1:
            raise ValueError("Patch token이 없습니다.")

        patch_tokens = tokens[1:]
        patch_grads = grads[1:]

        # 중요도 계산
        patch_importance = (patch_tokens * patch_grads).abs().mean(dim=1)  # [N_patch]

        num_patches = patch_importance.shape[0]
        grid_size = int(num_patches ** 0.5)

        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch token 수가 정사각 grid가 아닙니다: {num_patches}")

        cam = patch_importance.reshape(grid_size, grid_size)
        cam = cam.detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cv2.resize(cam, (image_size, image_size))
        return cam

    def remove(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


# =========================================================
# 6. 단일 backbone 모델 구성
# =========================================================
def build_single_model(model_name, num_classes):
    if model_name == "efficientnet_b4":
        return timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)

    elif model_name == "efficientnet_b7":
        return timm.create_model("efficientnet_b7", pretrained=False, num_classes=num_classes)

    elif model_name == "convnext_tiny":
        return timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes)

    elif model_name == "xception":
        try:
            return timm.create_model("xception", pretrained=False, num_classes=num_classes)
        except Exception as e:
            raise ValueError(
                "timm xception 생성 실패. 직접 사용한 Xception 클래스로 교체해야 합니다."
            ) from e

    else:
        raise ValueError(f"Unsupported single model: {model_name}")


def get_single_target_layer(model, model_name):
    if model_name in ["efficientnet_b4", "efficientnet_b7"]:
        if hasattr(model, "conv_head"):
            return model.conv_head

    elif model_name == "convnext_tiny":
        if hasattr(model, "stages"):
            last_stage = model.stages[-1]
            if hasattr(last_stage, "blocks") and hasattr(last_stage.blocks[-1], "conv_dw"):
                return last_stage.blocks[-1].conv_dw

    elif model_name == "xception":
        last_conv = find_last_conv(model)
        if last_conv is not None:
            return last_conv

    last_conv = find_last_conv(model)
    if last_conv is None:
        raise ValueError(f"{model_name}: Conv2d target layer를 찾지 못했습니다.")
    return last_conv


# =========================================================
# 7. fusion importance 추출기
# =========================================================
class FusionInspector:
    def __init__(self):
        self.saved = {}

    def save_tensor(self, name, tensor):
        if not tensor.requires_grad:
            return

        self.saved[name] = {
            "value": tensor,
            "grad": None
        }

        def hook_fn(grad):
            self.saved[name]["grad"] = grad.detach()

        tensor.register_hook(hook_fn)

    def get_scalar_importance(self, name):
        if name not in self.saved:
            return None

        value = self.saved[name]["value"]
        grad = self.saved[name]["grad"]

        if grad is None:
            return None

        return (value * grad).abs().mean().item()


# =========================================================
# 8. fusion 모델 예시
# =========================================================
"""
중요:
아래 ExampleFusionModel은 예시입니다.
네 실제 fusion 모델로 교체해야 합니다.

가정:
- rgb/frequency/wavelet branch는 CNN feature [B,C,H,W]
- semantic branch는 ViT/CLIP token [B,N,D]
"""

class ExampleSemanticViT(nn.Module):
    """
    예시용 semantic branch.
    timm vit_small_patch16_224를 사용.
    실제 CLIP/ViT branch로 교체 필요.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)

    def forward(self, x):
        # timm ViT 계열에서 forward_features 결과 활용
        tokens = self.backbone.forward_features(x)

        # timm 버전에 따라 [B,N,D] 또는 [B,D] 반환 가능성 있음
        # 여기서는 [B,N,D] 전제. [B,D]면 target module을 다른 곳에 걸어야 함
        return tokens


class ExampleFusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # CNN branches
        self.rgb_branch = timm.create_model("efficientnet_b0", pretrained=False, features_only=True, out_indices=[-1])
        self.freq_branch = timm.create_model("efficientnet_b0", pretrained=False, features_only=True, out_indices=[-1])
        self.wavelet_branch = timm.create_model("efficientnet_b0", pretrained=False, features_only=True, out_indices=[-1])

        # Semantic branch (ViT)
        self.semantic_branch = ExampleSemanticViT()

        branch_dim = 320
        sem_dim = 384  # vit_small_patch16_224 embedding dim

        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        self.freq_pool = nn.AdaptiveAvgPool2d(1)
        self.wave_pool = nn.AdaptiveAvgPool2d(1)

        self.rgb_proj = nn.Linear(branch_dim, 128)
        self.freq_proj = nn.Linear(branch_dim, 128)
        self.wave_proj = nn.Linear(branch_dim, 128)
        self.sem_proj = nn.Linear(sem_dim, 128)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.classifier = nn.Linear(256, num_classes)

        self.inspector = None

    def _extract_cnn_feat(self, branch, x):
        feat = branch(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        return feat

    def _extract_semantic_tokens(self, x):
        tokens = self.semantic_branch(x)

        # [B, N, D]면 그대로
        if tokens.ndim == 3:
            return tokens

        # [B, D]면 CLS만 나온 것이므로 patch map 불가
        if tokens.ndim == 2:
            raise ValueError(
                "semantic branch가 [B,D]만 반환했습니다. "
                "patch token [B,N,D]를 반환하도록 forward 또는 target module을 수정해야 합니다."
            )

        raise ValueError(f"Unexpected semantic token shape: {tokens.shape}")

    def forward(self, x):
        rgb_feat = self._extract_cnn_feat(self.rgb_branch, x)
        freq_feat = self._extract_cnn_feat(self.freq_branch, x)
        wave_feat = self._extract_cnn_feat(self.wavelet_branch, x)
        sem_tokens = self._extract_semantic_tokens(x)   # [B, N, D]

        # CNN vectors
        rgb_vec = self.rgb_pool(rgb_feat).flatten(1)
        freq_vec = self.freq_pool(freq_feat).flatten(1)
        wave_vec = self.wave_pool(wave_feat).flatten(1)

        # semantic CLS token 사용
        sem_vec = sem_tokens[:, 0, :]   # [B, D]

        rgb_proj = self.rgb_proj(rgb_vec)
        freq_proj = self.freq_proj(freq_vec)
        wave_proj = self.wave_proj(wave_vec)
        sem_proj = self.sem_proj(sem_vec)

        if self.inspector is not None:
            self.inspector.save_tensor("rgb_proj", rgb_proj)
            self.inspector.save_tensor("freq_proj", freq_proj)
            self.inspector.save_tensor("wave_proj", wave_proj)
            self.inspector.save_tensor("sem_proj", sem_proj)

        fusion_input = torch.cat([rgb_proj, freq_proj, wave_proj, sem_proj], dim=1)

        if self.inspector is not None:
            self.inspector.save_tensor("fusion_input", fusion_input)

        fusion_out = self.fusion_mlp(fusion_input)

        if self.inspector is not None:
            self.inspector.save_tensor("fusion_out", fusion_out)

        logits = self.classifier(fusion_out)
        return logits


def build_fusion_model(num_classes):
    """
    반드시 네 실제 fusion 모델 생성 코드로 교체할 것
    """
    return ExampleFusionModel(num_classes=num_classes)


def get_semantic_vit_target_module(semantic_branch):
    """
    semantic branch 내부에서 [B,N,D] token sequence가 나오는 모듈 반환
    반드시 실제 구조에 맞춰 직접 지정하는 것이 가장 정확함
    """

    # ExampleSemanticViT 내부 backbone 기준
    if hasattr(semantic_branch, "backbone"):
        bb = semantic_branch.backbone

        # timm ViT
        if hasattr(bb, "blocks") and len(bb.blocks) > 0:
            return bb.blocks[-1]

    # 일반 timm vit-like
    if hasattr(semantic_branch, "blocks") and len(semantic_branch.blocks) > 0:
        return semantic_branch.blocks[-1]

    # OpenAI CLIP-like
    if hasattr(semantic_branch, "transformer") and hasattr(semantic_branch.transformer, "resblocks"):
        return semantic_branch.transformer.resblocks[-1]

    raise ValueError("semantic branch에서 ViT target module을 찾지 못했습니다. 직접 지정이 필요합니다.")


def get_fusion_target_layers(model):
    conv_layers = {
        "rgb": find_last_conv(model.rgb_branch),
        "frequency": find_last_conv(model.freq_branch),
        "wavelet": find_last_conv(model.wavelet_branch),
    }

    for k, v in conv_layers.items():
        if v is None:
            raise ValueError(f"{k} branch에서 Conv2d layer를 찾지 못했습니다.")

    semantic_target = get_semantic_vit_target_module(model.semantic_branch)
    return conv_layers, semantic_target


# =========================================================
# 9. 단일 모델 실행
# =========================================================
def run_single_backbone_model(cfg, image_paths):
    model_name = cfg["model_name"]
    checkpoint_path = cfg["checkpoint_path"]
    image_size = cfg["image_size"]
    num_classes = cfg["num_classes"]
    target_class_mode = cfg.get("target_class_mode", "pred")

    print("=" * 100)
    print(f"[SINGLE MODEL] {model_name}")
    print(f"[CHECKPOINT  ] {checkpoint_path}")

    model = build_single_model(model_name, num_classes)
    model = load_checkpoint(model, checkpoint_path)
    model = model.to(DEVICE)
    model.eval()

    target_layer = get_single_target_layer(model, model_name)
    cam_extractor = GradCAM(model, target_layer)

    model_save_dir = os.path.join(SAVE_ROOT, model_name)
    ensure_dir(model_save_dir)

    csv_path = os.path.join(model_save_dir, "predictions.csv")
    csv_rows = []

    for image_path in image_paths:
        try:
            x, raw_img = load_image(image_path, image_size)
            x = x.to(DEVICE)

            model.zero_grad()
            output = model(x)

            class_idx, score, pred_idx, pred_score = get_target_class_from_output(
                output, mode=target_class_mode
            )
            score.backward(retain_graph=True)

            cam = cam_extractor.generate_from_existing_graph()
            heatmap, overlay = overlay_cam_on_image(raw_img, cam)

            image_name = os.path.splitext(os.path.basename(image_path))[0]

            cv2.imwrite(
                os.path.join(model_save_dir, f"{image_name}_heatmap.jpg"),
                cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(model_save_dir, f"{image_name}_overlay.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )

            save_three_panel_figure(
                raw_img=raw_img,
                heatmap=heatmap,
                overlay=overlay,
                save_path=os.path.join(model_save_dir, f"{image_name}_figure.jpg"),
                title_text=f"Pred={pred_idx}, Score={pred_score:.4f}, CAM class={class_idx}"
            )

            csv_rows.append([image_name, pred_idx, pred_score, class_idx])

            print(f"  [DONE] {image_name} | pred={pred_idx} | score={pred_score:.4f}")

        except Exception as e:
            print(f"  [ERROR] {os.path.basename(image_path)} -> {e}")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_idx", "pred_score", "cam_target_class"])
        writer.writerows(csv_rows)

    cam_extractor.remove()


# =========================================================
# 10. fusion 모델 실행
# =========================================================
def run_fusion_model(cfg, image_paths):
    model_name = cfg["model_name"]
    checkpoint_path = cfg["checkpoint_path"]
    image_size = cfg["image_size"]
    num_classes = cfg["num_classes"]
    target_class_mode = cfg.get("target_class_mode", "pred")

    print("=" * 100)
    print(f"[FUSION MODEL] {model_name}")
    print(f"[CHECKPOINT   ] {checkpoint_path}")

    model = build_fusion_model(num_classes)
    model = load_checkpoint(model, checkpoint_path)
    model = model.to(DEVICE)
    model.eval()

    conv_target_layers, semantic_target_module = get_fusion_target_layers(model)

    cam_extractors = {
        branch_name: GradCAM(model, layer)
        for branch_name, layer in conv_target_layers.items()
    }
    semantic_extractor = ViTTokenCAM(model, semantic_target_module)

    model_save_dir = os.path.join(SAVE_ROOT, model_name)
    ensure_dir(model_save_dir)

    csv_path = os.path.join(model_save_dir, "fusion_predictions.csv")
    csv_rows = []

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_save_dir = os.path.join(model_save_dir, image_name)
        ensure_dir(image_save_dir)

        try:
            x, raw_img = load_image(image_path, image_size)
            x = x.to(DEVICE)

            inspector = FusionInspector()
            model.inspector = inspector

            model.zero_grad()
            output = model(x)

            class_idx, score, pred_idx, pred_score = get_target_class_from_output(
                output, mode=target_class_mode
            )
            score.backward(retain_graph=True)

            # RGB / Frequency / Wavelet branch CAM
            for branch_name, cam_extractor in cam_extractors.items():
                cam = cam_extractor.generate_from_existing_graph()
                heatmap, overlay = overlay_cam_on_image(raw_img, cam)

                cv2.imwrite(
                    os.path.join(image_save_dir, f"{branch_name}_heatmap.jpg"),
                    cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    os.path.join(image_save_dir, f"{branch_name}_overlay.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )

                save_three_panel_figure(
                    raw_img=raw_img,
                    heatmap=heatmap,
                    overlay=overlay,
                    save_path=os.path.join(image_save_dir, f"{branch_name}_figure.jpg"),
                    title_text=f"{branch_name} | Pred={pred_idx}, Score={pred_score:.4f}"
                )

            # semantic branch Token-CAM
            semantic_cam = semantic_extractor.generate_from_existing_graph(image_size=image_size)
            semantic_heatmap, semantic_overlay = overlay_cam_on_image(raw_img, semantic_cam)

            cv2.imwrite(
                os.path.join(image_save_dir, "semantic_heatmap.jpg"),
                cv2.cvtColor(semantic_heatmap, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(image_save_dir, "semantic_overlay.jpg"),
                cv2.cvtColor(semantic_overlay, cv2.COLOR_RGB2BGR)
            )

            save_three_panel_figure(
                raw_img=raw_img,
                heatmap=semantic_heatmap,
                overlay=semantic_overlay,
                save_path=os.path.join(image_save_dir, "semantic_figure.jpg"),
                title_text=f"semantic(vit) | Pred={pred_idx}, Score={pred_score:.4f}"
            )

            # fusion importance
            fusion_scores = {
                "rgb_proj": inspector.get_scalar_importance("rgb_proj"),
                "freq_proj": inspector.get_scalar_importance("freq_proj"),
                "wave_proj": inspector.get_scalar_importance("wave_proj"),
                "sem_proj": inspector.get_scalar_importance("sem_proj"),
                "fusion_input": inspector.get_scalar_importance("fusion_input"),
                "fusion_out": inspector.get_scalar_importance("fusion_out"),
            }

            names = list(fusion_scores.keys())
            values = [0.0 if fusion_scores[k] is None else fusion_scores[k] for k in names]

            plt.figure(figsize=(8, 4))
            plt.bar(names, values)
            plt.xticks(rotation=20)
            plt.title(f"Fusion Importance | pred={pred_idx}, score={pred_score:.4f}")
            plt.tight_layout()
            plt.savefig(os.path.join(image_save_dir, "fusion_importance_bar.jpg"))
            plt.close()

            # branch comparison
            comparison_items = []
            for branch_name in ["rgb", "frequency", "wavelet", "semantic"]:
                overlay_path = os.path.join(image_save_dir, f"{branch_name}_overlay.jpg")
                if os.path.exists(overlay_path):
                    overlay_img = cv2.cvtColor(cv2.imread(overlay_path), cv2.COLOR_BGR2RGB)
                    comparison_items.append((branch_name, overlay_img))

            cols = 3
            rows = math.ceil((len(comparison_items) + 1) / cols)
            fig = plt.figure(figsize=(5 * cols, 4 * rows))

            ax0 = fig.add_subplot(rows, cols, 1)
            ax0.imshow(raw_img)
            ax0.set_title("Original")
            ax0.axis("off")

            idx = 2
            for branch_name, overlay_img in comparison_items:
                ax = fig.add_subplot(rows, cols, idx)
                ax.imshow(overlay_img)
                ax.set_title(branch_name)
                ax.axis("off")
                idx += 1

            plt.tight_layout()
            plt.savefig(os.path.join(image_save_dir, "branch_comparison.jpg"))
            plt.close()

            csv_rows.append([
                image_name,
                pred_idx,
                pred_score,
                class_idx,
                fusion_scores["rgb_proj"],
                fusion_scores["freq_proj"],
                fusion_scores["wave_proj"],
                fusion_scores["sem_proj"],
                fusion_scores["fusion_input"],
                fusion_scores["fusion_out"],
            ])

            print(f"  [DONE] {image_name} | pred={pred_idx} | score={pred_score:.4f}")

        except Exception as e:
            print(f"  [ERROR] {os.path.basename(image_path)} -> {e}")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name", "pred_idx", "pred_score", "cam_target_class",
            "rgb_proj_importance", "freq_proj_importance",
            "wave_proj_importance", "sem_proj_importance",
            "fusion_input_importance", "fusion_out_importance"
        ])
        writer.writerows(csv_rows)

    for cam_extractor in cam_extractors.values():
        cam_extractor.remove()
    semantic_extractor.remove()


# =========================================================
# 11. main
# =========================================================
def main():
    image_paths = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    if len(image_paths) == 0:
        print(f"[ERROR] 이미지가 없습니다: {IMAGE_DIR}")
        return

    for cfg in MODEL_CONFIGS:
        checkpoint_path = cfg["checkpoint_path"]

        if not os.path.exists(checkpoint_path):
            print(f"[SKIP] 체크포인트 없음: {checkpoint_path}")
            continue

        try:
            if cfg["model_type"] == "single":
                run_single_backbone_model(cfg, image_paths)

            elif cfg["model_type"] == "fusion":
                run_fusion_model(cfg, image_paths)

            else:
                print(f"[SKIP] 알 수 없는 model_type: {cfg['model_type']}")

        except Exception as e:
            print(f"[MODEL ERROR] {cfg['model_name']} -> {e}")


if __name__ == "__main__":
    main()