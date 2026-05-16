#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFPPFrameDataset 기반 학습 → 즉시 테스트(다중 데이터셋) 통합 스크립트
- 학습: FF++ 프레임 단위(FFT/비FFT 쌍)로 Meta-MLP만 학습 (백본·교차어텐션 고정)
- 테스트: 교차어텐션→Concat 벡터의 프레임 평균으로 비디오 스코어 산출
- RGB 백본 : HorNet (RGBsparial_step1.hornet.hornet_base_gf)
- FREQ 백본: ConvNeXt-V2 (Frequency_step2.models.convnextv2)

사용 예:
python train_then_eval_ffpp.py \
  --gpu 0 \
  --rgb-ckpt /path/hornet_base_ddp_best.pth \
  --freq-ckpt /path/convnext_fft_1_best.pth \
  --ffpp-root /PATH/FaceForensics++ \
  --compression c23 \
  --use-fft \
  --epochs 80 --lr 1e-3 --bs 64 --patience 10 \
  --save-mlp /path/meta_mlp_crossattn.pth \
  --csv-dir /path/results
"""

import os, sys, glob, argparse, cv2, numpy as np, torch, torch.nn as nn, pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

# ─── 로컬 모듈 경로 ──────────────────────────────────────
base = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base, "Frequency_step2"))
sys.path.append(os.path.join(base, "RGBsparial_step1"))
from Frequency_step2.models.convnextv2 import convnextv2_large
from RGBsparial_step1.hornet.hornet import hornet_base_gf

# ─── 고정 테스트 데이터셋 루트 (필요시 수정) ─────────────
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
        "real": ["/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences"],
        "fake": ["/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences"],
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
        "splits": ["train","test"],
    },
}

# ─── FFT / DCT ───────────────────────────────────────────
def extract_fft(bgr: np.ndarray) -> np.ndarray:
    chans = []
    for ch in cv2.split(bgr):
        dft   = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        shift = np.fft.fftshift(dft)
        mag   = cv2.magnitude(shift[:,:,0], shift[:,:,1])
        chans.append((20 * np.log(mag + 1)).astype(np.float32))
    return np.stack(chans, axis=2)

def extract_dct(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    return cv2.dct(gray).astype(np.float32)

# ─── FFPP 프레임 데이터셋(제공 클래스) ───────────────────
class FFPPFrameDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, compression='c23', use_fft=False, transform=None):
        self.use_fft   = use_fft
        self.transform = transform
        self.samples   = []
        roots = [
            os.path.join(root_dir, 'original_sequences'),
            os.path.join(root_dir, 'manipulated_sequences')
        ]
        for label, base in enumerate(roots):
            if not os.path.isdir(base): continue
            for method in os.listdir(base):
                full_dir = os.path.join(base, method, compression, 'mtcnn')
                if not os.path.isdir(full_dir): continue
                for subdir, _, files in os.walk(full_dir):
                    for fname in files:
                        if fname.lower().endswith(('.jpg','.png','jpeg')):
                            self.samples.append((os.path.join(subdir, fname), label))
        print(f"[Dataset] use_fft={self.use_fft}, 총 샘플: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            img = self.transform(img)  # PIL 유지(Resize 등)
        arr = np.array(img)[:, :, ::-1].astype(np.float32) / 255.0  # BGR [0,1]
        if self.use_fft:
            fft_map = extract_fft(arr)              # H W 3
            x_np = fft_map.transpose(2, 0, 1)      # 3 x H x W
        else:
            x_np = arr.transpose(2, 0, 1)          # 3 x H x W (BGR)
        return torch.from_numpy(x_np), torch.tensor(label, dtype=torch.long)

# ─── RGB/FFT 쌍 데이터셋 ─────────────────────────────────
class FFPPPairedDataset(Dataset):
    """동일 순서로 RGB(BGR-텐서)와 FFT 텐서를 페어링."""
    def __init__(self, root_dir, compression='c23', transform=None):
        self.ds_rgb = FFPPFrameDataset(root_dir, compression, use_fft=False, transform=transform)
        self.ds_fft = FFPPFrameDataset(root_dir, compression, use_fft=True,  transform=transform)
        assert len(self.ds_rgb) == len(self.ds_fft), "RGB/FFT 샘플 수가 다릅니다."
    def __len__(self):
        return len(self.ds_rgb)
    def __getitem__(self, idx):
        x_bgr, y1 = self.ds_rgb[idx]   # 3xHxW, BGR [0,1]
        x_fft, y2 = self.ds_fft[idx]   # 3xHxW, FFT
        assert y1.item() == y2.item()
        return x_bgr, x_fft, y1

# ─── 모델 로드 & 훅 ─────────────────────────────────────
def load_rgb_model(ckpt: str, device):
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    sd = {k.replace('model.',''): v for k,v in sd.items()}
    m = hornet_base_gf(num_classes=2)
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()

def load_freq_model(ckpt: str, device):
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    sd = {k.replace('model.',''): v for k,v in sd.items()}
    in_ch = 3
    for k,v in sd.items():
        if k.endswith('downsample_layers.0.0.weight') and v.dim()==4:
            in_ch = v.shape[1]; break
    m = convnextv2_large(in_chans=in_ch, num_classes=2, use_cbam=False)
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval(), in_ch

def attach_penultimate_hook(model: nn.Module):
    store = {'feat': None}
    last_linear = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_linear = mod
    if last_linear is None:
        raise RuntimeError("No nn.Linear to hook.")
    def hook(module, inputs, output):
        store['feat'] = inputs[0].detach()  # (B, D)
    h = last_linear.register_forward_hook(hook)
    return store, h

# ─── 교차 어텐션 → Concat 융합 ─────────────────────────
class CrossAttnFusion(nn.Module):
    def __init__(self, dim_r, dim_f, d_model=256, n_heads=4, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.q_r = nn.Linear(dim_r, d_model); self.k_f = nn.Linear(dim_f, d_model); self.v_f = nn.Linear(dim_f, d_model)
        self.attn_rf = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_rf   = nn.LayerNorm(d_model); self.res_r  = nn.Linear(dim_r, d_model)
        if bidirectional:
            self.q_f = nn.Linear(dim_f, d_model); self.k_r = nn.Linear(dim_r, d_model); self.v_r = nn.Linear(dim_r, d_model)
            self.attn_fr = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.ln_fr   = nn.LayerNorm(d_model); self.res_f  = nn.Linear(dim_f, d_model)
    def forward(self, f_rgb, f_freq):
        q = self.q_r(f_rgb).unsqueeze(1); k = self.k_f(f_freq).unsqueeze(1); v = self.v_f(f_freq).unsqueeze(1)
        out,_ = self.attn_rf(q,k,v)
        f_rgb_p = self.ln_rf(out.squeeze(1) + self.res_r(f_rgb))
        if self.bidirectional:
            q2 = self.q_f(f_freq).unsqueeze(1); k2 = self.k_r(f_rgb).unsqueeze(1); v2 = self.v_r(f_rgb).unsqueeze(1)
            out2,_ = self.attn_fr(q2,k2,v2)
            f_freq_p = self.ln_fr(out2.squeeze(1) + self.res_f(f_freq))
        else:
            f_freq_p = None
        return f_rgb_p, f_freq_p

# ─── Meta-MLP ───────────────────────────────────────────
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
    def forward(self, x): return self.net(x)

# ─── 유틸: 전처리/수집 ──────────────────────────────────
RESIZE_PIL = transforms.Resize((224,224))
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def normalize_imagenet_from_bgr(x_bgr: torch.Tensor, device):
    """x_bgr: (B,3,H,W) [0,1] BGR → RGB 정규화"""
    x_rgb = x_bgr[:, [2,1,0], :, :]
    mean = IMNET_MEAN.to(device); std = IMNET_STD.to(device)
    return (x_rgb - mean) / std

def collect_video_dirs(roots):
    vids = []
    for root in roots:
        if not os.path.isdir(root): continue
        for vid in sorted(os.listdir(root)):
            vd = os.path.join(root, vid)
            if os.path.isdir(vd):
                frames = sorted(glob.glob(os.path.join(vd, "*.png")))
                frames += sorted(glob.glob(os.path.join(vd, "*.jpg")))
                if frames: vids.append(frames)
    return vids

def expand_wilddeepfake(cfg):
    real_roots, fake_roots = [], []
    root = cfg['root']
    for split in cfg['splits']:
        sd = os.path.join(root, split)
        if not os.path.isdir(sd): continue
        for m in os.listdir(sd):
            base = os.path.join(sd, m)
            r,f  = os.path.join(base,"real"), os.path.join(base,"fake")
            if os.path.isdir(r): real_roots.append(r)
            if os.path.isdir(f): fake_roots.append(f)
    return {'real': real_roots, 'fake': fake_roots}

# ─── 프레임 → (fr, ff) 배치 추출 ───────────────────────
@torch.no_grad()
def batch_backbone_feats(x_bgr, x_fft, device, rgb_model, freq_model, freq_in_ch,
                         rgb_store, freq_store):
    """
    x_bgr: (B,3,H,W) BGR[0,1], x_fft: (B,3,H,W) FFT
    반환: fr(B,Dr), ff(B,Df)
    """
    x_rgb_norm = normalize_imagenet_from_bgr(x_bgr, device)
    _ = rgb_model(x_rgb_norm)
    fr = rgb_store['feat']  # (B, D_r)

    # freq 입력 채널 구성
    if freq_in_ch == 6:
        freq_in = torch.cat([x_bgr, x_fft], dim=1)  # BGR(3)+FFT(3)
    elif freq_in_ch == 3:
        freq_in = x_fft
    elif freq_in_ch == 1:
        freq_in = x_fft.mean(dim=1, keepdim=True)
    else:
        # 예외: 알 수 없는 채널 → FFT 3채널 사용
        freq_in = x_fft

    _ = freq_model(freq_in.to(device))
    ff = freq_store['feat']  # (B, D_f)
    return fr, ff

# ─── MLP 학습(프레임 단위) ─────────────────────────────
def train_meta_mlp_ffpp(root_dir, compression, device, rgb_model, freq_model, freq_in_ch,
                        d_model=256, n_heads=4, bidirectional=True,
                        epochs=100, lr=1e-3, wd=1e-4, bs=64, num_workers=4,
                        val_ratio=0.1, patience=10):
    # 데이터셋/로더
    ds = FFPPPairedDataset(root_dir, compression, transform=RESIZE_PIL)
    val_len = max(1, int(len(ds)*val_ratio))
    train_len = len(ds) - val_len
    tr_ds, va_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 백본 훅(한 번만 등록)
    rgb_store, rgb_hook   = attach_penultimate_hook(rgb_model)
    freq_store, freq_hook = attach_penultimate_hook(freq_model)

    # 차원 파악(더미 1배치)
    xb0, xf0, _ = next(iter(tr_dl))
    xb0, xf0 = xb0.to(device), xf0.to(device)
    with torch.no_grad():
        fr0, ff0 = batch_backbone_feats(xb0, xf0, device, rgb_model, freq_model, freq_in_ch, rgb_store, freq_store)
    D_r, D_f = fr0.shape[1], ff0.shape[1]
    fusion   = CrossAttnFusion(D_r, D_f, d_model=d_model, n_heads=n_heads, bidirectional=bidirectional).to(device).eval()
    in_dim   = d_model + (d_model if bidirectional else 0) + D_r + D_f
    mlp      = FeatMLP(in_dim=in_dim, hidden=256, out_dim=2, p=0.2).to(device)
    opt      = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd)
    ce       = nn.CrossEntropyLoss()

    # 백본/융합 고정
    for p in rgb_model.parameters():  p.requires_grad_(False)
    for p in freq_model.parameters(): p.requires_grad_(False)
    for p in fusion.parameters():     p.requires_grad_(False)

    best_loss, best_state, wait = 1e9, None, 0
    for ep in range(1, epochs+1):
        mlp.train()
        tr_loss = 0.0
        for x_bgr, x_fft, y in tqdm(tr_dl, desc=f"[Train] ep{ep}"):
            x_bgr, x_fft, y = x_bgr.to(device), x_fft.to(device), y.to(device)

            with torch.no_grad():
                fr, ff = batch_backbone_feats(x_bgr, x_fft, device, rgb_model, freq_model, freq_in_ch,
                                              rgb_store, freq_store)
                f_rgb_p, f_freq_p = fusion(fr, ff)
                fused = torch.cat([f_rgb_p, f_freq_p, fr, ff], dim=1) if bidirectional \
                        else torch.cat([f_rgb_p, fr, ff], dim=1)

            logits = mlp(fused)
            loss   = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * y.size(0)

        tr_loss /= len(tr_dl.dataset)

        # validation
        mlp.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x_bgr, x_fft, y in va_dl:
                x_bgr, x_fft, y = x_bgr.to(device), x_fft.to(device), y.to(device)
                fr, ff = batch_backbone_feats(x_bgr, x_fft, device, rgb_model, freq_model, freq_in_ch,
                                              rgb_store, freq_store)
                f_rgb_p, f_freq_p = fusion(fr, ff)
                fused = torch.cat([f_rgb_p, f_freq_p, fr, ff], dim=1) if bidirectional \
                        else torch.cat([f_rgb_p, fr, ff], dim=1)
                va_loss += ce(mlp(fused), y).item() * y.size(0)
        va_loss /= len(va_dl.dataset)

        print(f"[MLP][{ep:03d}] train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best_loss - 1e-4:
            best_loss, best_state, wait = va_loss, {k:v.detach().cpu() for k,v in mlp.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    if best_state is not None:
        mlp.load_state_dict(best_state, strict=True)

    # 훅 해제
    rgb_hook.remove(); freq_hook.remove()
    return fusion, mlp, in_dim

# ─── 비디오 특징(프레임 평균) ──────────────────────────
@torch.no_grad()
def build_video_feature(frames, device, rgb_model, freq_model, fusion, freq_in_ch, num_workers=0):
    # 훅 1회 등록
    rgb_store, rgb_hook   = attach_penultimate_hook(rgb_model)
    freq_store, freq_hook = attach_penultimate_hook(freq_model)

    vecs = []
    for p in frames:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        img = RESIZE_PIL(img)
        arr = np.array(img).astype(np.float32) / 255.0
        bgr = torch.from_numpy(arr[:, :, ::-1].transpose(2,0,1)).unsqueeze(0).to(device)  # 1x3xHxW, BGR

        # dummy FFT (테스트 시 FFT 사용 가정: freq_in_ch에 맞춰 구성)
        fft_np = extract_fft(arr[:, :, ::-1])
        x_fft  = torch.from_numpy(fft_np.transpose(2,0,1)).unsqueeze(0).float().to(device)

        fr, ff = batch_backbone_feats(bgr, x_fft, device, rgb_model, freq_model, freq_in_ch,
                                      rgb_store, freq_store)
        f_rgb_p, f_freq_p = fusion(fr, ff)
        fused = torch.cat([f_rgb_p, f_freq_p, fr, ff], dim=1) if fusion.bidirectional \
                else torch.cat([f_rgb_p, fr, ff], dim=1)
        vecs.append(fused[0].cpu().numpy())
        torch.cuda.empty_cache()

    rgb_hook.remove(); freq_hook.remove()
    if not vecs: return None
    return np.mean(np.stack(vecs,0), axis=0).astype(np.float32)

# ─── 평가 루프 ──────────────────────────────────────────
def evaluate_datasets(rgb_model, freq_model, fusion, mlp, device, csv_dir, threshold=0.5):
    os.makedirs(csv_dir, exist_ok=True)
    results, all_true, all_pred = [], [], []

    # freq in_chans 파악
    freq_in_ch = 3
    for n, p in freq_model.named_parameters():
        if 'downsample_layers.0.0.weight' in n and p.dim()==4:
            freq_in_ch = p.shape[1]; break

    for ds_name, cfg in TEST_DATASETS.items():
        if ds_name == "WildDeepfake":
            paths = expand_wilddeepfake(cfg)
        elif ds_name == "DeepfakeTIMIT":
            real_roots, fake_roots = [], []
            for q in cfg.get('real', []):
                for grp in os.listdir(q):
                    gp = os.path.join(q, grp)
                    if os.path.isdir(gp):
                        for vid in os.listdir(gp):
                            p = os.path.join(gp, vid)
                            if os.path.isdir(p): real_roots.append(p)
            for q in cfg.get('fake', []):
                for grp in os.listdir(q):
                    gp = os.path.join(q, grp)
                    if os.path.isdir(gp):
                        for vid in os.listdir(gp):
                            p = os.path.join(gp, vid)
                            if os.path.isdir(p): fake_roots.append(p)
            paths = {'real': real_roots, 'fake': fake_roots}
        else:
            paths = cfg

        def collect_video_dirs(roots):
            vids = []
            for root in roots:
                if not os.path.isdir(root): continue
                for vid in sorted(os.listdir(root)):
                    vd = os.path.join(root, vid)
                    if os.path.isdir(vd):
                        frames = sorted(glob.glob(os.path.join(vd, "*.png")))
                        frames += sorted(glob.glob(os.path.join(vd, "*.jpg")))
                        if frames: vids.append(frames)
            return vids

        real_vids = collect_video_dirs(paths.get('real', []))
        fake_vids = collect_video_dirs(paths.get('fake', []))

        y_t, y_p = [], []
        for frames in tqdm(real_vids, desc=f"[0] {ds_name} real"):
            feat = build_video_feature(frames, device, rgb_model, freq_model, fusion, freq_in_ch)
            if feat is None: continue
            with torch.no_grad():
                prob = torch.softmax(mlp(torch.from_numpy(feat)[None].to(device)),1)[0,1].item()
            y_t.append(0); y_p.append(1 if prob >= threshold else 0)
        for frames in tqdm(fake_vids, desc=f"[1] {ds_name} fake"):
            feat = build_video_feature(frames, device, rgb_model, freq_model, fusion, freq_in_ch)
            if feat is None: continue
            with torch.no_grad():
                prob = torch.softmax(mlp(torch.from_numpy(feat)[None].to(device)),1)[0,1].item()
            y_t.append(1); y_p.append(1 if prob >= threshold else 0)

        if y_t:
            acc  = accuracy_score(y_t, y_p)
            prec = precision_score(y_t, y_p, zero_division=0)
            rec  = recall_score(y_t, y_p, zero_division=0)
            f1   = f1_score(y_t, y_p, average='macro')
            print(f"[{ds_name}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
            results.append({'dataset': ds_name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_macro': f1})
            all_true.extend(y_t); all_pred.extend(y_p)

    if all_true:
        oa  = accuracy_score(all_true, all_pred)
        op  = precision_score(all_true, all_pred, zero_division=0)
        or_ = recall_score(all_true, all_pred, zero_division=0)
        of1 = f1_score(all_true, all_pred, average='macro')
        print("\n=== Overall Metrics ===")
        print(f"Acc={oa:.4f}  Prec={op:.4f}  Rec={or_:.4f}  F1={of1:.4f}")
        results.append({'dataset':'Overall','accuracy':oa,'precision':op,'recall':or_,'f1_macro':of1})

    out_csv = os.path.join(csv_dir, "crossattn_metaMLP_ffpp_metrics.csv")
    pd.DataFrame(results, columns=['dataset','accuracy','precision','recall','f1_macro']).to_csv(out_csv, index=False)
    print(f"\n▶ Saved metrics to {out_csv}")

# ─── 메인 ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--rgb-ckpt',  type=str, required=True)
    ap.add_argument('--freq-ckpt', type=str, required=True)
    ap.add_argument('--ffpp-root', type=str, required=True, help='FF++ 루트( original_sequences / manipulated_sequences 포함 )')
    ap.add_argument('--compression', type=str, default='raw', choices=['raw','c23','c40'])
    ap.add_argument('--use-fft', action='store_true', help='(참고) 학습은 항상 RGB+FFT를 사용합니다.')
    ap.add_argument('--d-model',  type=int, default=256)
    ap.add_argument('--n-heads',  type=int, default=4)
    ap.add_argument('--oneway',   action='store_true', help='단방향(RGB→Freq)만 사용')
    ap.add_argument('--epochs',   type=int, default=100)
    ap.add_argument('--lr',       type=float, default=1e-3)
    ap.add_argument('--wd',       type=float, default=1e-4)
    ap.add_argument('--bs',       type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--csv-dir',   type=str, default="/home/oem/deepfake/Ourmethod/results")
    ap.add_argument('--save-mlp',  type=str, default="", help='학습된 Meta-MLP 저장경로(.pth)')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"▶ Device: {device}")

    # 백본 로드(고정)
    rgb_model  = load_rgb_model(args.rgb-ckpt, device)
    freq_model, freq_in_ch = load_freq_model(args.freq-ckpt, device)
    print(f"▶ Loaded RGB from {args.rgb-ckpt}")
    print(f"▶ Loaded FREQ from {args.freq-ckpt} (in_chans={freq_in_ch})")

    # Meta-MLP 학습 (FFPP 프레임 쌍 기반)
    fusion, mlp, in_dim = train_meta_mlp_ffpp(
        root_dir=args.ffpp_root, compression=args.compression, device=device,
        rgb_model=rgb_model, freq_model=freq_model, freq_in_ch=freq_in_ch,
        d_model=args.d_model, n_heads=args.n_heads, bidirectional=not args.oneway,
        epochs=args.epochs, lr=args.lr, wd=args.wd, bs=args.bs, num_workers=args.num_workers,
        val_ratio=args.val_ratio, patience=args.patience
    )
    if args.save_mlp:
        torch.save(mlp.state_dict(), args.save_mlp)
        print(f"💾 Saved Meta-MLP to {args.save_mlp}")

    # 학습 직후 다중 테스트셋 평가
    evaluate_datasets(rgb_model, freq_model, fusion, mlp, device,
                      csv_dir=args.csv_dir, threshold=args.threshold)

if __name__ == "__main__":
    main()
