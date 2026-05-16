# viz_freq_transforms.py
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt

# --- DCT: scipy 필요 ---
try:
    from scipy.fftpack import dct
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def _require_scipy():
    if not SCIPY_OK:
        raise RuntimeError("DCT 시각화를 위해 scipy가 필요합니다. pip install scipy")

def to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    m = x.max()
    if m > 1e-12:
        x /= m
    return (x * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

def dct2(img2d: np.ndarray) -> np.ndarray:
    _require_scipy()
    x = img2d.astype(np.float32)
    # 2D DCT-II (ortho)
    y = dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')
    return y

def dct_visualize(img2d: np.ndarray, suppress_dc=True, robust=True,
                  p_low=1.0, p_high=99.0, gamma=1.0) -> np.ndarray:
    """
    반환: 0..1 float 시각화 맵(좌상단이 저주파)
    """
    Y = dct2(img2d)
    if suppress_dc:
        Y = Y.copy()
        Y[0, 0] = 0.0
    mag = np.log1p(np.abs(Y))
    if robust:
        lo, hi = np.percentile(mag, [p_low, p_high])
        mag = np.clip(mag, lo, hi)
    mag -= mag.min()
    mag /= (mag.max() + 1e-12)
    if gamma != 1.0:
        mag = np.power(mag, gamma)
    return mag

def fft_visualize(img2d: np.ndarray, robust=True, p_low=1.0, p_high=99.0, gamma=1.0) -> np.ndarray:
    """
    2D FFT: 중심 이동 + log-magnitude → 0..1 float
    """
    x = img2d.astype(np.float32)
    f = np.fft.fft2(x)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    if robust:
        lo, hi = np.percentile(mag, [p_low, p_high])
        mag = np.clip(mag, lo, hi)
    mag -= mag.min()
    mag /= (mag.max() + 1e-12)
    if gamma != 1.0:
        mag = np.power(mag, gamma)
    return mag

def wavelet_dwt_grid(img2d: np.ndarray, wavelet='db2', level=2, mode='symmetric') -> np.ndarray:
    """
    Label-Down(Decimated) DWT를 타일 모자이크로 시각화 → uint8 2D
    """
    coeffs = pywt.wavedec2(img2d.astype(np.float32), wavelet=wavelet, level=level, mode=mode)
    arr, _ = pywt.coeffs_to_array(coeffs)
    return to_uint8(np.abs(arr))

def stack_rgb(ch_list_01: list) -> np.ndarray:
    """
    0..1 float 채널 3개 → (H,W,3) uint8
    """
    ch8 = [(c * 255.0 + 0.5).clip(0, 255).astype(np.uint8) for c in ch_list_01]
    return np.stack(ch8, axis=-1)

def save_with_ext(fig, out_path: str):
    root, ext = os.path.splitext(out_path)
    if not ext:
        out_path = root + ".png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print("[OK] saved →", out_path)

def main():
    ap = argparse.ArgumentParser(description="RGB → DCT/FFT/Wavelet 시각화(1×4)")
    ap.add_argument("--input", required=True, help="입력 RGB 이미지 경로")
    ap.add_argument("--output", default="freq_viz.png", help="출력 파일(.png 권장, 미지정시 .png 자동)")
    ap.add_argument("--wavelet", default="db2", help="Wavelet (예: haar, db2, sym4, coif1, bior4.4)")
    ap.add_argument("--level", type=int, default=2, help="DWT 분해 레벨")
    ap.add_argument("--mode", default="symmetric", help="DWT 경계 모드")
    ap.add_argument("--max_width", type=int, default=640, help="가로 리사이즈 상한")
    # DCT/FFT 시각화 튜닝
    ap.add_argument("--suppress_dc", type=str, default="true", help="DCT DC 억제 여부")
    ap.add_argument("--robust", type=str, default="true", help="백분위 클리핑 사용")
    ap.add_argument("--plow", type=float, default=1.0, help="클리핑 하위 백분위")
    ap.add_argument("--phigh", type=float, default=99.0, help="클리핑 상위 백분위")
    ap.add_argument("--gamma", type=float, default=1.0, help="감마 보정(1.0=없음)")
    args = ap.parse_args()

    suppress_dc = args.suppress_dc.lower() in ("1","true","yes","y")
    robust = args.robust.lower() in ("1","true","yes","y")

    # 1) 입력 로드
    img = Image.open(args.input).convert("RGB")
    W, H = img.size
    if W > args.max_width:
        scale = args.max_width / float(W)
        img = img.resize((args.max_width, int(H * scale)), Image.BICUBIC)
    rgb = np.array(img, dtype=np.float32)  # (H,W,3)

    # 2) 채널별 변환
    dct_rgb_01 = []
    fft_rgb_01 = []
    wav_rgb_u8 = []
    for c in range(3):
        ch = rgb[..., c]
        dct_vis = dct_visualize(ch, suppress_dc=suppress_dc, robust=robust,
                                p_low=args.plow, p_high=args.phigh, gamma=args.gamma)
        fft_vis = fft_visualize(ch, robust=robust,
                                p_low=args.plow, p_high=args.phigh, gamma=args.gamma)
        wav_vis = wavelet_dwt_grid(ch, wavelet=args.wavelet, level=args.level, mode=args.mode)

        dct_rgb_01.append(dct_vis)
        fft_rgb_01.append(fft_vis)
        wav_rgb_u8.append(wav_vis)

    dct_vis_rgb = stack_rgb(dct_rgb_01)
    fft_vis_rgb = stack_rgb(fft_rgb_01)
    wav_vis_rgb = np.stack(wav_rgb_u8, axis=-1)  # 이미 uint8

    # 3) 패널 구성
    titles = [
        "RGB",
        f"DCT",
        f"FFT",
        f"Wavelet(label-down)"
    ]
    images = [rgb.astype(np.uint8), dct_vis_rgb, fft_vis_rgb, wav_vis_rgb]

    fig = plt.figure(figsize=(16, 4.6))
    for i, (im, t) in enumerate(zip(images, titles), start=1):
        ax = plt.subplot(1, 4, i)
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    plt.tight_layout()

    # 4) 저장
    save_with_ext(fig, args.output)

if __name__ == "__main__":
    main()
