import sys
from pathlib import Path
import cv2, torch, numpy as np, pandas as pd
from tqdm import tqdm

sys.path.append('/home/oem/deepfake/Ourmethod/Frequency_step2')
from models.convnextv2 import convnextv2_large
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# 1) load checkpoint
ckpt = torch.load(
    '/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoints/convnext_fft_1_best.pth',
    map_location=DEVICE
)
state = {k.replace('model.', ''): v for k,v in ckpt.items()}

# 2) FFT extractor (no per-image norm)
def extract_fft(image_bgr: np.ndarray) -> np.ndarray:
    channels = []
    for ch in cv2.split(image_bgr):
        dft       = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        shift     = np.fft.fftshift(dft)
        mag       = cv2.magnitude(shift[:,:,0], shift[:,:,1])
        channels.append((20 * np.log(mag + 1)).astype(np.float32))
    return np.stack(channels, axis=2)  # H×W×3

# 3) infer in_chans from checkpoint
stem_w   = next(v for k,v in state.items() if k.endswith('downsample_layers.0.0.weight'))
in_chans = stem_w.shape[1]
print(f"⚙️  Model expects {in_chans} input channels")

# 4) build & load model
model    = convnextv2_large(in_chans=in_chans, num_classes=2, use_cbam=False)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Loaded ckpt: missing={missing}, unexpected={unexpected}")
model.to(DEVICE).eval()

# 5) 테스트
TEST_DIR = Path('/home/oem/deepfake/hdd/test_sample_frames_5')
OUT_CSV  = Path('/home/oem/deepfake/Ourmethod/Frequency_step2/test_csv/convnext_fft_all_data_frame5.csv')
results  = []

for vid_dir in tqdm(sorted(TEST_DIR.iterdir()), desc="Videos"):
    if not vid_dir.is_dir(): continue
    vid   = vid_dir.name + '.mp4'  # 비디오 ID
    probs = []

    for img_path in sorted(vid_dir.iterdir()):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: 
            continue
        img_bgr = cv2.resize(img_bgr, (224,224), interpolation=cv2.INTER_LANCZOS4)
        bgr_f   = img_bgr.astype(np.float32) / 255.0    # H×W×3
        
        fft_map = extract_fft(bgr_f)                   # H×W×3 raw log-mag
        # concat → 6×224×224
        inp_np  = np.concatenate([
            bgr_f.transpose(2,0,1),
            fft_map.transpose(2,0,1),
        ], axis=0)
        inp     = torch.from_numpy(inp_np[None]).to(DEVICE)

        with torch.no_grad():
            out = model(inp)
            probs.append(torch.softmax(out,1)[0,1].item())

    avg_p = float(np.mean(probs)) if probs else 0.0
    label = 1 if avg_p > 0.5 else 0
    results.append((vid, label))

pd.DataFrame(results, columns=['ID','label']).to_csv(OUT_CSV, index=False)
print("✅ Written fixed submission:", OUT_CSV)
