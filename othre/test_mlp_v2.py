#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------
# Meta-MLP 평가 스크립트 (학습 파이프라인과 완전 동일화 버전)
# - RGB EfficientNet-B7 (num_classes=2)
# - Freq RepLKNet31B (in_channels=4, num_classes=2)
# - penultimate feature 훅 + Δlogit(2개) -> concat -> FeatMLP
# - FFT/DCT 둘 다 지원 (학습 때 사용한 방식과 동일해야 함)
# - 온도 스케일링(T_rgb, T_freq) 지원 (temps-json 또는 인자)
# -----------------------------------------------------------
import os, sys, glob, argparse, cv2, json, torch, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from contextlib import nullcontext
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import torch.nn as nn

base = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base, "Frequency_step2"))
sys.path.append(os.path.join(base, "RGBsparial_step1"))

# ---------- FFT / DCT ----------
def extract_fft(bgr: np.ndarray) -> np.ndarray:
    """채널별 DFT → magnitude → log scale."""
    chans = []
    for ch in cv2.split(bgr):
        dft   = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        shift = np.fft.fftshift(dft)
        mag   = cv2.magnitude(shift[:, :, 0], shift[:, :, 1])
        chans.append((20 * np.log(mag + 1)).astype(np.float32))
    return np.stack(chans, axis=2)  # H×W×3

def extract_dct(bgr: np.ndarray) -> np.ndarray:
    """BGR → gray(0~1) → DCT."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    return cv2.dct(gray).astype(np.float32)  # H×W

# ---------- Dataset ----------
class FrameDS(Dataset):
    def __init__(self, frame_paths, resize, freq_method='dct'):
        self.frames, self.resize = frame_paths, resize
        self.freq_method = freq_method
        self.norm = T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

    def __len__(self): return len(self.frames)

    def __getitem__(self, idx):
        img_pil = self.resize(Image.open(self.frames[idx]).convert('RGB'))

        # RGB (EfficientNet 입력)
        rgb = self.norm(T.ToTensor()(img_pil))  # 3×224×224

        # FREQ (RepLKNet 입력, 4채널)
        arr = np.asarray(img_pil)               # RGB (H×W×3)
        bgr = arr[:, :, ::-1]                   # BGR uint8
        if self.freq_method == 'fft':
            fft3  = extract_fft(bgr)                                # H×W×3
            freq1 = np.mean(fft3, axis=2, keepdims=True).astype(np.float32)  # H×W×1
            bgr_f = bgr.astype(np.float32) / 255.0
            freq_in = np.concatenate([bgr_f, freq1], axis=2)        # H×W×4
        else:
            # DCT: 학습과 동일하게 0~1 BGR + DCT(1채널)
            bgr_f = bgr.astype(np.float32) / 255.0
            dctm  = extract_dct(bgr)                                # H×W
            freq1 = dctm[..., None]
            freq_in = np.concatenate([bgr_f, freq1], axis=2)        # H×W×4

        freq = torch.from_numpy(freq_in.transpose(2, 0, 1)).float() # 4×224×224
        return rgb, freq

# ---------- Hook (penultimate) ----------
def attach_penultimate_hook(model: nn.Module):
    """마지막 nn.Linear 입력(=penultimate feature)을 캡처."""
    store = {'feat': None}
    last_linear = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("No nn.Linear layer found for penultimate hook.")
    def hook(mod, inp, out):
        store['feat'] = inp[0].detach()  # (B, D)
    handle = last_linear.register_forward_hook(hook)
    return store, handle

# ---------- MLP (학습 때와 동일) ----------
class FeatMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=2, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def load_mlp_from_ckpt(path, device):
    """
    저장된 FeatMLP state_dict로부터 in_dim/hidden 자동 추론 후 로드.
    - 우선 'net.1.weight' 사용 시도 (Linear(in_dim->hidden))
    - 없으면 2D weight 중 'second dim'이 가장 큰 것을 입력층으로 추정
    """
    sd = torch.load(path, map_location=device)
    if not isinstance(sd, dict):
        raise RuntimeError("MLP ckpt is not a state_dict-like dict.")

    # 1) 선호 키
    if 'net.1.weight' in sd:
        w1 = sd['net.1.weight']
        in_dim, hidden = w1.shape[1], w1.shape[0]
    else:
        # 2) 폴백: 2D weight 중 두 번째 차원이 가장 큰 weight를 입력층으로 추정
        cand = [(k, v) for k, v in sd.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
        if not cand:
            raise RuntimeError("No 2D weights found in MLP state_dict.")
        k, w = max(cand, key=lambda kv: kv[1].shape[1])
        in_dim, hidden = w.shape[1], w.shape[0]

    mlp = FeatMLP(in_dim=in_dim, hidden=hidden, out_dim=2, p=0.0).to(device).eval()
    mlp.load_state_dict(sd, strict=True)
    return mlp, in_dim

# ---------- 모델 가중치 로더 ----------
def load_weights(path, device):
    """
    다양한 형태의 .pth 파일에서 state_dict를 꺼냄.
    우선순위: model_state > state_dict > net > weights > 그대로
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for k in ('model_state', 'state_dict', 'net', 'weights'):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

# ---------- 온도 스케일링 ----------
def maybe_load_temps(json_path, default_rgb=1.0, default_freq=1.0):
    T_rgb, T_freq = default_rgb, default_freq
    if json_path and Path(json_path).exists():
        with open(json_path, 'r') as f:
            cfg = json.load(f)
        T_rgb  = float(cfg.get('T_rgb',  T_rgb))
        T_freq = float(cfg.get('T_freq', T_freq))
        print(f"[*] Loaded temperatures: T_rgb={T_rgb}, T_freq={T_freq}")
    return T_rgb, T_freq

# ---------- inference (비디오 단위) ----------
@torch.no_grad()
def infer_video(rgb_model, freq_model, mlp, loader, device, T_rgb=1.0, T_freq=1.0):
    # 훅 등록
    rgb_store, rgb_hook   = attach_penultimate_hook(rgb_model)
    freq_store, freq_hook = attach_penultimate_hook(freq_model)

    probs = []
    amp_ctx = autocast if device.type == 'cuda' else nullcontext

    for rgb, freq in loader:
        rgb, freq = rgb.to(device), freq.to(device)
        with amp_ctx():
            pr_logits = rgb_model(rgb) / T_rgb     # (B,2)
            pf_logits = freq_model(freq) / T_freq  # (B,2)

            fr = rgb_store['feat']                 # (B, D_r)
            ff = freq_store['feat']                # (B, D_f)
            if fr is None or ff is None:
                # 이론상 발생하지 않지만 방어적으로 한 번 더 전파
                pr_logits = rgb_model(rgb) / T_rgb
                pf_logits = freq_model(freq) / T_freq
                fr = rgb_store['feat']
                ff = freq_store['feat']

            d_rgb  = (pr_logits[:, 1] - pr_logits[:, 0]).unsqueeze(1)   # (B,1)
            d_freq = (pf_logits[:, 1] - pf_logits[:, 0]).unsqueeze(1)   # (B,1)
            fusion = torch.cat([fr, ff, d_rgb, d_freq], dim=1)          # (B, D_r+D_f+2)

            p_fake = torch.softmax(mlp(fusion), dim=1)[:, 1]            # (B,)
        probs.append(p_fake.cpu().numpy())

    rgb_hook.remove(); freq_hook.remove()
    return float(np.concatenate(probs).mean())

# ---------- main ----------
def main():
    par = argparse.ArgumentParser()
    par.add_argument('--gpu', default='0')
    par.add_argument('--rgb-ckpt',   required=True)
    par.add_argument('--freq-ckpt',  required=True)
    par.add_argument('--mlp-ckpt',   required=True)

    par.add_argument('--freq-method', choices=['fft', 'dct'], default='fft')
    par.add_argument('--temps-json', type=str, default='')
    par.add_argument('--T-rgb',      type=float, default=1.0)
    par.add_argument('--T-freq',     type=float, default=1.0)

    par.add_argument('--batch', type=int, default=4)
    par.add_argument('--num-workers', type=int, default=0)
    par.add_argument('--thr',   type=float, default=0.5)
    par.add_argument('--out',   default='/home/sujin/psj2003/deepfake/code/result/mlp/mlp_eval_results1.csv')
    args = par.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ─ 백본 & MLP ───────────────────────────────────────
    from efficientnet_pytorch import EfficientNet
    from Frequency_step2.models.replknet import create_RepLKNet31B

    rgb_model  = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)
    freq_model = create_RepLKNet31B(in_channels=4, num_classes=2, use_cbam=False)

    rgb_sd = load_weights(args.rgb_ckpt, device)
    rgb_model.load_state_dict(rgb_sd, strict=False)

    freq_sd = load_weights(args.freq_ckpt, device)
    freq_model.load_state_dict(freq_sd, strict=False)

    rgb_model.eval().to(device)
    freq_model.eval().to(device)

    mlp, in_dim = load_mlp_from_ckpt(args.mlp_ckpt, device)
    print(f"[*] Loaded Meta-MLP from {args.mlp_ckpt} (in_dim={in_dim})")

    T_rgb, T_freq = maybe_load_temps(args.temps_json, args.T_rgb, args.T_freq)

    resize = T.Resize((224, 224))

    # ─ 평가 대상 데이터셋 경로 정의 ─────────────────────
    DATASETS = {
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
            "root":   "/home/oem/deepfake/hdd_5TB/WildDeepfake",
            "splits": ["train","test"],
        },
    }

    records, y_all, p_all = [], [], []

    # ---------- 데이터셋 루프 ----------
    for ds_name, cfg in DATASETS.items():

        # --- real / fake 폴더 수집 ---
        if ds_name == "WildDeepfake":
            real_roots, fake_roots = [], []
            for split in cfg['splits']:
                sd = os.path.join(cfg['root'], split)
                if not os.path.isdir(sd): continue
                for subj in os.listdir(sd):
                    base = os.path.join(sd, subj)
                    r, f = os.path.join(base, "real"), os.path.join(base, "fake")
                    if os.path.isdir(r): real_roots.append(r)
                    if os.path.isdir(f): fake_roots.append(f)
            ds_paths = {"real": real_roots, "fake": fake_roots}

        elif ds_name == "DeepfakeTIMIT":
            fake_roots = []
            for qroot in cfg["fake"]:
                if not os.path.isdir(qroot): continue
                for spk in os.listdir(qroot):
                    spk_path = os.path.join(qroot, spk)
                    if os.path.isdir(spk_path):
                        fake_roots.append(spk_path)
            ds_paths = {"real": [], "fake": fake_roots}

        else:
            ds_paths = cfg  # Celeb, DFD

        # --- 평가 ---
        for label_name, label_val in [("real", 0), ("fake", 1)]:
            for root in ds_paths.get(label_name, []):
                if not os.path.isdir(root):
                    print(f"[Skip] {root}")
                    continue

                for vid in tqdm(sorted(os.listdir(root)),
                                 desc=f"{ds_name}-{label_name}", leave=False):
                    vid_dir = os.path.join(root, vid)
                    if not os.path.isdir(vid_dir): continue

                    frames = sorted(glob.glob(os.path.join(vid_dir, "*.png")) +
                                    glob.glob(os.path.join(vid_dir, "*.jpg")))
                    if not frames: continue

                    loader = DataLoader(
                        FrameDS(frames, resize, freq_method=args.freq_method),
                        batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=='cuda')
                    )

                    prob_fake = infer_video(
                        rgb_model, freq_model, mlp,
                        loader, device, T_rgb=T_rgb, T_freq=T_freq
                    )
                    pred = int(prob_fake >= args.thr)

                    y_all.append(label_val)
                    p_all.append(pred)
                    records.append({
                        "dataset": ds_name,
                        "video":   vid_dir,
                        "label":   label_val,
                        "prob_fake": prob_fake
                    })

    # ---------- 전체 지표 ----------
    metrics_records = []
    if y_all:
        acc  = accuracy_score(y_all, p_all)
        prec = precision_score(y_all, p_all, zero_division=0)
        rec  = recall_score(y_all, p_all, zero_division=0)
        f1_m = f1_score(y_all, p_all, average='macro')
        f1_b = f1_score(y_all, p_all, average='binary')
        print(f"\n=== Overall ===  Acc {acc:.4f}  "
              f"Prec {prec:.4f}  Rec {rec:.4f}  F1-Macro {f1_m:.4f}  F1-Binary {f1_b:.4f}")
        metrics_records.append({
            "dataset": "Overall",
            "Acc": acc,
            "Prec": prec,
            "Rec": rec,
            "F1-Macro": f1_m,
            "F1-Binary": f1_b
        })

    # ---------- 데이터셋별 지표 ----------
    for ds_name in DATASETS.keys():
        ds_labels = [r["label"] for r in records if r["dataset"] == ds_name]
        ds_preds  = [int(r["prob_fake"] >= args.thr) for r in records if r["dataset"] == ds_name]
        if not ds_labels: 
            continue
        acc  = accuracy_score(ds_labels, ds_preds)
        prec = precision_score(ds_labels, ds_preds, zero_division=0)
        rec  = recall_score(ds_labels, ds_preds, zero_division=0)
        f1_m = f1_score(ds_labels, ds_preds, average='macro')
        f1_b = f1_score(ds_labels, ds_preds, average='binary')
        print(f"=== {ds_name} ===  Acc {acc:.4f}  "
              f"Prec {prec:.4f}  Rec {rec:.4f}  F1-Macro {f1_m:.4f}  F1-Binary {f1_b:.4f}")
        metrics_records.append({
            "dataset": ds_name,
            "Acc": acc,
            "Prec": prec,
            "Rec": rec,
            "F1-Macro": f1_m,
            "F1-Binary": f1_b
        })

    # ---------- CSV 저장 ----------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved per-video results → {out_path}")

    # ---------- 지표 CSV 저장 ----------
    metrics_out = out_path.with_name(out_path.stem + "_metrics_v2.csv")
    pd.DataFrame(metrics_records).to_csv(metrics_out, index=False)
    print(f"Saved metrics results → {metrics_out}")

if __name__ == '__main__':
    main()
