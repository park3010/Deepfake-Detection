#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, UnidentifiedImageError

# ------------------------- 기본 설정 -------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- 로컬 임포트 ------------------------
base = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base, "Frequency_step2"))
sys.path.append(os.path.join(base, "RGBsparial_step1"))
from Frequency_step2.models.convnextv2 import convnextv2_large
from RGBsparial_step1.hornet.hornet import hornet_base_gf

# ---------------------- FFT / DCT 유틸 ----------------------
def extract_fft(bgr: np.ndarray) -> np.ndarray:
    chans = []
    for ch in cv2.split(bgr):
        dft   = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        shift = np.fft.fftshift(dft)
        mag   = cv2.magnitude(shift[:, :, 0], shift[:, :, 1])
        chans.append((20 * np.log(mag + 1)).astype(np.float32))
    return np.stack(chans, axis=2)

def extract_dct(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    return cv2.dct(gray).astype(np.float32)

# --------------------- 백본 로드/훅 유틸 --------------------
def load_base(kind, ckpt_path):
    sd = torch.load(ckpt_path, map_location=DEVICE)
    sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
    sd = {k.replace('model.', ''): v for k, v in sd.items()}  # DDP 호환

    if kind == 'hornet':
        m = hornet_base_gf(num_classes=2)
    elif kind == 'freq':
        stem_w = next(v for k, v in sd.items() if k.endswith('downsample_layers.0.0.weight'))
        m = convnextv2_large(in_chans=stem_w.shape[1], num_classes=2, use_cbam=False)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE).eval()

def attach_penultimate_hook(model: nn.Module):
    """마지막 nn.Linear의 입력 텐서를 캡처."""
    storage = {'feat': None}
    last_linear = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_linear = mod
    if last_linear is None:
        raise RuntimeError("No nn.Linear layer found for penultimate hook.")

    def hook(module, inputs, output):
        storage['feat'] = inputs[0].detach()  # (B, D)

    handle = last_linear.register_forward_hook(hook)
    return storage, handle

# ----------------------- FF++ 학습 데이터셋 -----------------------
class FFPPFrameDataset(Dataset):
    """
    FF++ 루트에서 original/manipulated의 mtcnn 프레임을 스캔하여
    (이미지 경로, 라벨=0/1)을 수집. __getitem__에서 PIL 이미지를 반환.
    """
    def __init__(self, root_dir, compression='c23', transform=None):
        self.transform = transform
        self.samples   = []
        roots = [
            os.path.join(root_dir, 'original_sequences'),
            os.path.join(root_dir, 'manipulated_sequences')
        ]
        for label, base in enumerate(roots):
            if not os.path.isdir(base):
                continue
            for method in os.listdir(base):
                full_dir = os.path.join(base, method, compression, 'mtcnn')
                if not os.path.isdir(full_dir): 
                    continue
                for subdir, _, files in os.walk(full_dir):
                    for fname in files:
                        if fname.lower().endswith(('.jpg','.png','jpeg')):
                            self.samples.append((os.path.join(subdir, fname), label))
        print(f"[FFPP] 총 샘플: {len(self.samples)}")

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
        return img, torch.tensor(label, dtype=torch.long)

# --------------------- Meta-MLP (특징 concat+로짓) --------------------
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

# --------------------- 온도 스케일링 로드(선택) ----------------------
def maybe_load_temps(json_path, default_rgb=1.0, default_freq=1.0):
    T_rgb, T_freq = default_rgb, default_freq
    if json_path and Path(json_path).exists():
        with open(json_path, "r") as f:
            cfg = json.load(f)
        T_rgb  = float(cfg.get("T_rgb",  T_rgb))
        T_freq = float(cfg.get("T_freq", T_freq))
        print(f"[*] Loaded temperatures: T_RGB={T_rgb}, T_FREQ={T_freq}")
    return T_rgb, T_freq

# --------------------------- 학습 루틴 ---------------------------
def train_meta_mlp(args, rgb_model, freq_model, T_RGB, T_FREQ):
    """
    FF++ 프레임 데이터셋으로 Meta-MLP 학습
    입력: [fr, ff, logit_rgb, logit_freq]
    백본/훅: 고정(eval), MLP만 학습
    """
    resize_tf = T.Resize((224, 224))
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = FFPPFrameDataset(args.ffpp_root, args.compression, transform=resize_tf)
    val_len = max(1, int(len(ds)*args.val_ratio))
    tr_len  = len(ds) - val_len
    tr_ds, va_ds = random_split(ds, [tr_len, val_len], generator=torch.Generator().manual_seed(42))

    tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 훅 한 번만 등록
    rgb_store,  rgb_hook  = attach_penultimate_hook(rgb_model)
    freq_store, freq_hook = attach_penultimate_hook(freq_model)
    softmax = nn.Softmax(dim=1)

    # in_dim 파악(더미 1배치)
    xb0, y0 = next(iter(tr_dl))
    with torch.no_grad():
        # RGB forward
        pr_logits = rgb_model(to_tensor(xb0[0]).unsqueeze(0).to(DEVICE)) / T_RGB
        fr = rgb_store['feat']; assert fr is not None
        D_r = fr.shape[1]
        # FREQ forward
        arr = np.array(xb0[0]).astype(np.float32) / 255.0
        bgr = arr[:, :, ::-1]
        if args.freq_method == 'fft':
            freq_map = extract_fft(bgr)
        else:
            bgr_uint8 = (bgr * 255).astype(np.uint8)
            dctm      = extract_dct(bgr_uint8)
            freq_map  = np.stack([dctm]*3, axis=2)
        freq_in = np.concatenate([bgr, freq_map], axis=2)   # H×W×6 (체크포인트와 동일 가정)
        inp_f   = torch.from_numpy(freq_in.transpose(2,0,1)[None]).float().to(DEVICE)
        _ = freq_model(inp_f) / T_FREQ
        ff = freq_store['feat']; assert ff is not None
        D_f = ff.shape[1]
        in_dim = int(D_r + D_f + 2)

    mlp = FeatMLP(in_dim=in_dim, hidden=args.mlp_hidden, out_dim=2, p=args.dropout).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.wd)
    ce  = nn.CrossEntropyLoss()

    best_val, best_state, wait = 1e9, None, 0
    for ep in range(1, args.epochs+1):
        mlp.train()
        tr_loss = 0.0
        for imgs, y in tqdm(tr_dl, desc=f"[Train] ep{ep}"):
            B = len(imgs)
            y = y.to(DEVICE)

            # RGB forward (배치)
            rgb_in = torch.stack([to_tensor(img) for img in imgs], dim=0).to(DEVICE)  # (B,3,224,224)
            with torch.no_grad():
                pr_logits = rgb_model(rgb_in) / T_RGB     # (B,2)
                fr = rgb_store['feat']                    # (B, D_r)

            # FREQ forward (배치)
            fused_feats = []
            with torch.no_grad():
                for img in imgs:
                    arr = np.array(img).astype(np.float32) / 255.0
                    bgr = arr[:, :, ::-1]
                    if args.freq_method == 'fft':
                        freq_map = extract_fft(bgr)
                    else:
                        bgr_uint8 = (bgr * 255).astype(np.uint8)
                        dctm      = extract_dct(bgr_uint8)
                        freq_map  = np.stack([dctm]*3, axis=2)
                    freq_in = np.concatenate([bgr, freq_map], axis=2)   # H×W×6
                    inp_f   = torch.from_numpy(freq_in.transpose(2,0,1)[None]).float().to(DEVICE)
                    pf_logits = freq_model(inp_f) / T_FREQ              # (1,2)
                    ff = freq_store['feat']                             # (1, D_f)
                    # 로짓 스칼라 (fake-real)
                    l_rgb  = (pr_logits[:,1] - pr_logits[:,0]) if isinstance(pr_logits, torch.Tensor) else None
                    l_freq = (pf_logits[0,1] - pf_logits[0,0]).unsqueeze(0)  # (1,)
                    # 해당 프레임의 인덱스에 맞는 fr 추출
                    # 주의: 훅은 마지막 forward 입력이 저장됨 -> 배치 루프에서 RGB는 B개, FREQ는 1개씩 처리
                    # 간단화를 위해 RGB 로짓 스칼라/특징은 배치 기준 한 번만 가져오고, FREQ는 루프 내에서 추출
                    # 여기서는 각 img 단위로 fr을 재사용하지 않고, B=1 시나리오로 정합
                    # 안정적 동작을 위해 RGB도 한 장씩 돌리는 방식으로 맞추는 것이 안전
                    pass
            # 위 배치/훅 정합 문제 때문에 RGB도 프레임 단위로 처리(안전).
            fused_list = []
            with torch.no_grad():
                for i, img in enumerate(imgs):
                    # RGB
                    pr = rgb_model(to_tensor(img).unsqueeze(0).to(DEVICE)) / T_RGB
                    fr = rgb_store['feat']  # (1,D_r)
                    # FREQ
                    arr = np.array(img).astype(np.float32) / 255.0
                    bgr = arr[:, :, ::-1]
                    if args.freq_method == 'fft':
                        freq_map = extract_fft(bgr)
                    else:
                        bgr_uint8 = (bgr * 255).astype(np.uint8)
                        dctm      = extract_dct(bgr_uint8)
                        freq_map  = np.stack([dctm]*3, axis=2)
                    freq_in = np.concatenate([bgr, freq_map], axis=2)
                    pf = freq_model(torch.from_numpy(freq_in.transpose(2,0,1)[None]).float().to(DEVICE)) / T_FREQ
                    ff = freq_store['feat']  # (1,D_f)
                    # 로짓 스칼라
                    l_rgb  = (pr[0,1] - pr[0,0]).item()
                    l_freq = (pf[0,1] - pf[0,0]).item()
                    fused  = torch.from_numpy(
                        np.concatenate([fr[0].cpu().numpy(), ff[0].cpu().numpy(),
                                        np.array([l_rgb, l_freq], np.float32)], axis=0)
                    ).float().to(DEVICE)  # (D_r + D_f + 2,)
                    fused_list.append(fused)
            fused_batch = torch.stack(fused_list, dim=0)  # (B, in_dim)

            logits = mlp(fused_batch)
            loss   = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * B

            torch.cuda.empty_cache()

        tr_loss /= len(tr_dl.dataset)

        # Validation
        mlp.eval()
        va_loss = 0.0
        with torch.no_grad():
            for imgs, y in va_dl:
                y = y.to(DEVICE)
                fused_list = []
                for img in imgs:
                    # RGB
                    pr = rgb_model(to_tensor(img).unsqueeze(0).to(DEVICE)) / T_RGB
                    fr = rgb_store['feat']
                    # FREQ
                    arr = np.array(img).astype(np.float32) / 255.0
                    bgr = arr[:, :, ::-1]
                    if args.freq_method == 'fft':
                        freq_map = extract_fft(bgr)
                    else:
                        bgr_uint8 = (bgr * 255).astype(np.uint8)
                        dctm      = extract_dct(bgr_uint8)
                        freq_map  = np.stack([dctm]*3, axis=2)
                    freq_in = np.concatenate([bgr, freq_map], axis=2)
                    pf = freq_model(torch.from_numpy(freq_in.transpose(2,0,1)[None]).float().to(DEVICE)) / T_FREQ
                    ff = freq_store['feat']
                    l_rgb  = (pr[0,1] - pr[0,0]).item()
                    l_freq = (pf[0,1] - pf[0,0]).item()
                    fused  = torch.from_numpy(
                        np.concatenate([fr[0].cpu().numpy(), ff[0].cpu().numpy(),
                                        np.array([l_rgb, l_freq], np.float32)], axis=0)
                    ).float().to(DEVICE)
                    fused_list.append(fused)
                fused_batch = torch.stack(fused_list, dim=0)
                va_loss += ce(mlp(fused_batch), y).item() * len(y)
        va_loss /= len(va_dl.dataset)

        print(f"[MLP][{ep:03d}] train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best_val - 1e-4:
            best_val, best_state, wait = va_loss, {k:v.detach().cpu() for k,v in mlp.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at epoch {ep}")
                break

    if best_state is not None:
        mlp.load_state_dict(best_state, strict=True)

    # 훅 해제
    rgb_hook.remove(); freq_hook.remove()

    if args.save_mlp:
        Path(args.save_mlp).parent.mkdir(parents=True, exist_ok=True)
        torch.save(mlp.state_dict(), args.save_mlp)
        print(f"💾 Saved Meta-MLP to {args.save_mlp}")

    return mlp, in_dim

# --------------------------- 테스트 루틴 ---------------------------
def run_test(args, rgb_model, freq_model, mlp, T_RGB, T_FREQ):
    resize_tf = T.Resize((224, 224))
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    softmax = nn.Softmax(dim=1)

    # 훅 등록(테스트)
    rgb_store,  rgb_hook  = attach_penultimate_hook(rgb_model)
    freq_store, freq_hook = attach_penultimate_hook(freq_model)

    results = []
    test_dir = Path(args.test_dir)
    for vid_dir in tqdm(sorted(test_dir.iterdir()), desc="Test videos"):
        if not vid_dir.is_dir():
            continue
        vid = vid_dir.name + ".mp4"
        frame_scores = []

        for img_path in sorted(vid_dir.iterdir()):
            if img_path.suffix.lower() not in [".png",".jpg",".jpeg"]:
                continue
            img = Image.open(img_path).convert("RGB")
            img = resize_tf(img)

            # — RGB (로짓 + 특징)
            with torch.no_grad():
                inp_r = to_tensor(img).unsqueeze(0).to(DEVICE)
                pr_logits = rgb_model(inp_r) / T_RGB
                fr = rgb_store['feat']                          # (1, D_r)
            if fr is None:
                continue
            fr = fr[0].cpu().numpy()

            # — 주파수 입력
            arr = np.array(img).astype(np.float32) / 255.0
            bgr = arr[:, :, ::-1]
            if args.freq_method == 'fft':
                freq_map = extract_fft(bgr)
            else:
                bgr_uint8 = (bgr * 255).astype(np.uint8)
                dctm      = extract_dct(bgr_uint8)
                freq_map  = np.stack([dctm]*3, axis=2)
            freq_in = np.concatenate([bgr, freq_map], axis=2)   # H×W×6
            inp_f   = torch.from_numpy(freq_in.transpose(2,0,1)[None]).float().to(DEVICE)
            with torch.no_grad():
                pf_logits = freq_model(inp_f) / T_FREQ
                ff = freq_store['feat']                         # (1, D_f)
            if ff is None:
                continue
            ff = ff[0].cpu().numpy()

            # — 로짓 스칼라(temperature scaling 반영)
            logit_rgb  = (pr_logits[0,1] - pr_logits[0,0]).item()
            logit_freq = (pf_logits[0,1] - pf_logits[0,0]).item()

            # — 특징 concat (+ 로짓 2개)
            fused_feat = np.concatenate(
                [fr, ff, np.array([logit_rgb, logit_freq], dtype=np.float32)],
                axis=0
            )
            with torch.no_grad():
                logits = mlp(torch.from_numpy(fused_feat).float().unsqueeze(0).to(DEVICE))
                prob   = softmax(logits)[0].cpu().numpy()   # [p_real, p_fake]
            frame_scores.append(float(prob[1]))             # fake 확률

            torch.cuda.empty_cache()

        video_score = float(np.mean(frame_scores)) if frame_scores else 0.0
        results.append((vid, int(video_score > args.threshold)))

    # 훅 해제
    rgb_hook.remove(); freq_hook.remove()

    # CSV 저장
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(results, columns=["ID", "label"]).to_csv(out_csv, index=False)
    print("✅ Test submission written to", out_csv)

# ----------------------------- 메인 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    # 경로/체크포인트
    ap.add_argument('--rgb-ckpt',  type=str, required=True)
    ap.add_argument('--freq-ckpt', type=str, required=True)
    ap.add_argument('--mlp-ckpt',  type=str, default="", help='(선택) 기존 학습된 MLP를 불러옵니다.')
    ap.add_argument('--save-mlp',  type=str, default="", help='학습된 MLP 저장 경로(.pth)')

    # 온도 스케일링
    ap.add_argument('--temps-json', type=str, default="", help='{"T_rgb":1.2,"T_freq":1.1}')
    ap.add_argument('--T-rgb',      type=float, default=1.0)
    ap.add_argument('--T-freq',     type=float, default=1.0)

    # 주파수 방법/테스트 경로
    ap.add_argument('--freq-method', choices=['fft','dct'], default='fft')
    ap.add_argument('--test-dir', type=str, required=True)
    ap.add_argument('--out-csv',  type=str, required=True)
    ap.add_argument('--threshold', type=float, default=0.5)

    # 학습 데이터(FF++)
    ap.add_argument('--ffpp-root', type=str, required=False, help='FF++ 루트 경로')
    ap.add_argument('--compression', type=str, default='raw', choices=['raw','c23','c40'])

    # 학습 하이퍼
    ap.add_argument('--epochs',  type=int,   default=50)
    ap.add_argument('--lr',      type=float, default=1e-3)
    ap.add_argument('--wd',      type=float, default=1e-4)
    ap.add_argument('--bs',      type=int,   default=32)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--patience',  type=int,   default=8)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--mlp-hidden',  type=int, default=256)
    ap.add_argument('--dropout',     type=float, default=0.2)

    args = ap.parse_args()

    # 백본 로드
    rgb_model  = load_base('hornet', args.rgb_ckpt)
    freq_model = load_base('freq',   args.freq_ckpt)
    print(f"▶ Loaded RGB from {args.rgb_ckpt}")
    print(f"▶ Loaded FREQ from {args.freq_ckpt}")

    # 온도 스케일링 설정
    T_RGB, T_FREQ = maybe_load_temps(args.temps_json, args.T_rgb, args.T_freq)

    # MLP를 불러오거나(있으면), 없으면 학습 후 사용
    mlp = None
    if args.mlp_ckpt and Path(args.mlp_ckpt).exists():
        # in_dim 검증을 위해 더미 한 프레임에서 특징 차원 계산
        resize_tf = T.Resize((224, 224))
        to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        rgb_store,  rgb_hook  = attach_penultimate_hook(rgb_model)
        freq_store, freq_hook = attach_penultimate_hook(freq_model)
        # TEST_DIR에서 임의 프레임 하나
        any_img = None
        for vd in sorted(Path(args.test_dir).iterdir()):
            if vd.is_dir():
                imgs = list(sorted(vd.glob("*.png"))) + list(sorted(vd.glob("*.jpg")))
                if imgs:
                    any_img = Image.open(imgs[0]).convert("RGB")
                    any_img = resize_tf(any_img)
                    break
        if any_img is None:
            raise RuntimeError("테스트 디렉토리에서 임의 프레임을 찾지 못했습니다.")
        with torch.no_grad():
            pr = rgb_model(to_tensor(any_img).unsqueeze(0).to(DEVICE)) / T_RGB
            fr = rgb_store['feat']; assert fr is not None
            arr = np.array(any_img).astype(np.float32) / 255.0
            bgr = arr[:, :, ::-1]
            if args.freq_method == 'fft':
                freq_map = extract_fft(bgr)
            else:
                bgr_uint8 = (bgr * 255).astype(np.uint8)
                dctm      = extract_dct(bgr_uint8)
                freq_map  = np.stack([dctm]*3, axis=2)
            freq_in = np.concatenate([bgr, freq_map], axis=2)
            pf = freq_model(torch.from_numpy(freq_in.transpose(2,0,1)[None]).float().to(DEVICE)) / T_FREQ
            ff = freq_store['feat']; assert ff is not None
            in_dim = int(fr.shape[1] + ff.shape[1] + 2)

        mlp = FeatMLP(in_dim=in_dim, hidden=args.mlp_hidden, out_dim=2, p=args.dropout).to(DEVICE).eval()
        mlp.load_state_dict(torch.load(args.mlp_ckpt, map_location=DEVICE), strict=True)
        print(f"[*] Loaded Meta-MLP from {args.mlp_ckpt}")
        rgb_hook.remove(); freq_hook.remove()
    else:
        if not args.ffpp_root:
            raise RuntimeError("MLP 체크포인트가 없으므로, --ffpp-root 를 지정해 학습 후 테스트해야 합니다.")
        mlp, _ = train_meta_mlp(args, rgb_model, freq_model, T_RGB, T_FREQ)

    # 학습(또는 로드)된 MLP로 즉시 테스트
    run_test(args, rgb_model, freq_model, mlp, T_RGB, T_FREQ)

if __name__ == '__main__':
    main()
