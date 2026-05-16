#!/usr/bin/env python3
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from netcal.metrics import ECE
import timm

# local imports
import sys
sys.path.append('/home/oem/deepfake/Ourmethod/Frequency_step2')
from Frequency_step2.models.convnextv2 import ConvNeXtV2
from Frequency_step2.models.replknet import create_RepLKNet31B
from RGBsparial_step1.hornet.hornet import hornet_tiny_gf

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

DATA_DIR   = Path("/home/oem/deepfake/hdd_5TB/FF++")
TEST_DIR   = Path("/home/oem/deepfake/hdd/test_sample_frames_5")
OUT_CSV    = Path("/home/oem/deepfake/Ourmethod/ensemble_results")
RGB_CKPT   = Path("/home/oem/deepfake/Ourmethod/RGBsparial_step1/checkpoints/hornet_base_ddp_best.pth")
FREQ_CKPT  = Path("/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoints/convnex_fft_no_data/convnext_fft_6_best.pth")

def normalize_probs(p, method):
    q = p.copy()
    if method=='minmax':
        mn, mx = q.min(0,keepdims=True), q.max(0,keepdims=True)
        q = (q-mn)/(mx-mn+1e-12)
    elif method=='zscore':
        mu, sd = q.mean(0,keepdims=True), q.std(0,keepdims=True)
        q = (q-mu)/(sd+1e-12)
    q -= q.min(1,keepdims=True)
    q /= (q.sum(1,keepdims=True)+1e-12)
    return q

def calibrate_temperature(p, y, Ts, bins):
    best_T, best_ece = None, np.inf
    meter = ECE(bins=bins)
    for T in Ts:
        lp = np.log(p+1e-12)/T
        exp = np.exp(lp - lp.max(1,keepdims=True))
        pt = exp/exp.sum(1,keepdims=True)
        e = meter.measure(pt, y)
        if e<best_ece:
            best_ece, best_T = e, T
    return best_T

def soft_voting_equal(ps):
    ens = np.mean(ps,0)
    return ens.argmax(1), ens

def soft_voting_weighted(ps, ws):
    w = np.array(ws)
    ens = sum(wi*p for wi,p in zip(w,ps))
    return ens.argmax(1), ens

def print_metrics(y, preds, ens=None, name='Ensemble', bins=15):
    print(f"--- {name} ---")
    print("Accuracy :", accuracy_score(y,preds))
    print("F1 score :", f1_score(y,preds,average='macro'))
    print("Precision:", precision_score(y,preds,average='macro',zero_division=0))
    print("Recall   :", recall_score(y,preds,average='macro',zero_division=0))
    if ens is not None:
        ece = ECE(bins=bins).measure(ens, y)
        print(f"ECE ({bins} bins): {ece:.4f}")
    print()

def extract_fft(image_bgr: np.ndarray) -> np.ndarray:
    channels = []
    for ch in cv2.split(image_bgr):
        dft   = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        shift = np.fft.fftshift(dft)
        mag   = cv2.magnitude(shift[:,:,0], shift[:,:,1])
        channels.append((20 * np.log(mag + 1)).astype(np.float32))
    return np.stack(channels, axis=2)

def load_model(kind, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    sd = {k.replace('model.', ''): v for k,v in sd.items()}

    if kind == 'convnext':
        stem_w = next(v for k,v in sd.items() if k.endswith('downsample_layers.0.0.weight'))
        in_chans = stem_w.shape[1]
        m = ConvNeXtV2(in_chans=in_chans, num_classes=2, use_cbam=False)
    elif kind == 'xception':
        m = timm.create_model('xception', pretrained=False, num_classes=2)
    elif kind == 'hornet':
        m = hornet_tiny_gf(num_classes=2)
    else:
        m = create_RepLKNet31B(num_classes=2, in_channels=3, use_cbam=False)

    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    torch.cuda.empty_cache()  
    return m

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
            img = self.transform(img)
        arr = np.array(img)[:, :, ::-1].astype(np.float32) / 255.0
        if self.use_fft:
            fft_map = extract_fft(arr)
            x_np = fft_map.transpose(2, 0, 1)
        else:
            x_np = arr.transpose(2, 0, 1)
        return torch.from_numpy(x_np), torch.tensor(label, dtype=torch.long)

def inference(model, loader, device):
    allp, ally = [], []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Infer {model.__class__.__name__}", leave=False):
            x = x.to(device)
            out = model(x)
            p = softmax(out).cpu().numpy()
            allp.append(p)
            ally.append(y.numpy())
    result_p = np.vstack(allp)
    result_y = np.concatenate(ally)
    torch.cuda.empty_cache()
    return result_p, result_y

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rgb-kind',   choices=['xception','hornet','maxvit','coanet'], default='xception')
    p.add_argument('--freq-kind',  choices=['convnext'], default='convnext')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--method',     choices=['equal','weighted'], default='equal')
    p.add_argument('--weights',    default='0.5,0.5')
    p.add_argument('--grid-search',action='store_true')
    p.add_argument('--norm',       choices=['none','minmax','zscore'], default='none')
    p.add_argument('--temp-scale', action='store_true')
    p.add_argument('--T-range',    default='1,5,0.1')
    p.add_argument('--bins',       type=int, default=15)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # transforms
    resize_tf   = T.Resize((224,224))
    to_tensor_tf= T.Compose([T.ToTensor(),
                             T.Normalize([0.485,0.456,0.406],
                                         [0.229,0.224,0.225])])

    # loaders
    ds_rgb  = FFPPFrameDataset(DATA_DIR, use_fft=False, transform=resize_tf)
    loader_rgb  = DataLoader(ds_rgb, batch_size=args.batch_size,
                             shuffle=False, num_workers=1, pin_memory=False)
    ds_freq = FFPPFrameDataset(DATA_DIR, use_fft=True,  transform=resize_tf)
    loader_freq = DataLoader(ds_freq, batch_size=args.batch_size,
                             shuffle=False, num_workers=1, pin_memory=False)

    # load models
    rgb_model  = load_model(args.rgb_kind,  RGB_CKPT,  device)
    freq_model = load_model(args.freq_kind, FREQ_CKPT, device)

    # inference on validation
    rgb_probs, y_true  = inference(rgb_model,  loader_rgb,  device)
    freq_probs, _      = inference(freq_model, loader_freq, device)

    # normalization
    if args.norm!='none':
        rgb_probs  = normalize_probs(rgb_probs,  args.norm)
        freq_probs = normalize_probs(freq_probs, args.norm)

    # temperature scaling
    if args.temp_scale:
        start,end,step = map(float, args.T_range.split(','))
        Ts = np.arange(start,end+1e-9,step)
        Tr = calibrate_temperature(rgb_probs,  y_true, Ts, args.bins)
        Tf = calibrate_temperature(freq_probs,y_true, Ts, args.bins)
        def apply_T(p,T):
            lp = np.log(p+1e-12)/T
            e  = np.exp(lp - lp.max(1,keepdims=True))
            return e / e.sum(1,keepdims=True)
        rgb_probs  = apply_T(rgb_probs,  Tr)
        freq_probs = apply_T(freq_probs, Tf)

    # ensemble on val
    if args.method=='equal':
        preds, ens = soft_voting_equal([rgb_probs, freq_probs])
        print_metrics(y_true, preds, ens, name="Equal Voting", bins=args.bins)
    else:
        if args.grid_search:
            best_f1,best_w = -1,0
            for w in np.linspace(0,1,101):
                p = w*rgb_probs + (1-w)*freq_probs
                f = f1_score(y_true, p.argmax(1), average='macro')
                if f>best_f1:
                    best_f1,best_w = f,w
            ws = [best_w,1-best_w]
        else:
            ws = list(map(float,args.weights.split(',')))
        preds, ens = soft_voting_weighted([rgb_probs, freq_probs], ws)
        print_metrics(y_true, preds, ens, name=f"Weighted Voting {ws}", bins=args.bins)

    # test inference & ensemble
    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    test_results = []
    for vid_dir in tqdm(sorted(TEST_DIR.iterdir()), desc="Test videos"):
        if not vid_dir.is_dir(): continue
        vid = vid_dir.name

        frame_probs_rgb, frame_probs_freq = [], []
        for img_path in sorted(vid_dir.iterdir()):
            pil = Image.open(img_path).convert("RGB")
            pil = resize_tf(pil)
            inp = to_tensor_tf(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                p_rgb  = torch.softmax(rgb_model(inp),  dim=1)[0,1].item()
                p_freq = torch.softmax(freq_model(inp), dim=1)[0,1].item()
            frame_probs_rgb.append(p_rgb)
            frame_probs_freq.append(p_freq)

        mean_rgb  = np.mean(frame_probs_rgb)  if frame_probs_rgb  else 0.0
        mean_freq = np.mean(frame_probs_freq) if frame_probs_freq else 0.0
        if args.method=='equal':
            combined = 0.5*mean_rgb + 0.5*mean_freq
        else:
            w1,w2 = ws
            combined = w1*mean_rgb + w2*mean_freq

        label = 1 if combined>0.5 else 0
        test_results.append((vid, label))

    import pandas as pd
    df_sub = pd.DataFrame(test_results, columns=["ID","label"])
    df_sub.to_csv(OUT_CSV, index=False)
    print(f"✅ Test submission written to {OUT_CSV}")

if __name__ == '__main__':
    main()
