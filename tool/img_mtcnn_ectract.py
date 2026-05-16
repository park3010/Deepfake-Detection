#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5-프레임마다 얼굴을 추출해 PNG 저장
  • 입력  : <ROOT>/<branch>/<method>/raw/videos/<clip>.mp4
  • 출력  : <ROOT>/<branch>/<method>/raw/mtcnn/<clip>/<연속번호>.png
실행 예
    CUDA_VISIBLE_DEVICES=1 python extract_face_every5.py
"""

import os, cv2, torch, warnings, shutil, subprocess, tempfile
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import multiprocessing as mp

# ───────── 사용자 설정 ──────────────────────────────────────────
ROOT        = Path('/home/oem/deepfake/hdd_5TB/FF++')
BRANCHES    = ['manipulated_sequences']
METHODS     = ['NeuralTextures']
COMPR       = 'c40'

SKIP        = 5                 # n-프레임마다 1장
FRAME_METH  = 'ffmpeg'          # 'ffmpeg' | 'cv2'
JOBS        = mp.cpu_count()    # ffmpeg 병렬 스레드 when FRAME_METH='ffmpeg'

OUT_SIZE    = 224               # 얼굴 해상도
GPU_ID      = '3'               # 사용할 GPU ID
FMT         = 'PNG'             # 저장 포맷 (‘PNG’ | ‘JPEG’)
PNG_COMP    = 3                 # PNG compress_level (0~9)
# ───────────────────────────────────────────────────────────────

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[INFO] device = {device}')

# ─── 얼굴 검출기 ────────────────────────────────────────────────
mtcnn = MTCNN(
    image_size     = OUT_SIZE,
    margin         = 0,
    select_largest = True,
    post_process   = False,     # float Tensor 반환
    device         = device
)

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}

# ─── 헬퍼 ───────────────────────────────────────────────────────
def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.dtype != torch.uint8:           # 0~255 float or other → uint8
        t = t.clamp(0, 255).byte()
    arr = t.permute(1, 2, 0).cpu().numpy()   # C,H,W → H,W,C
    return Image.fromarray(arr, mode='RGB')

def save_face(pil_img: Image.Image, path: Path):
    if FMT.upper() == 'PNG':
        pil_img.save(path.with_suffix('.png'), format='PNG', compress_level=PNG_COMP)
    else:
        pil_img.save(path.with_suffix('.jpg'), format='JPEG', quality=90)

# ─── 프레임 추출 함수들 ─────────────────────────────────────────
def extract_frames_ffmpeg(vpath: Path, tmp_dir: Path, skip: int):
    cmd = ['ffmpeg', '-y', '-v', 'quiet',
           '-threads', str(max(1, JOBS // 2)),
           '-i', str(vpath),
           '-vf', f"select='not(mod(n\\,{skip}))'",
           '-vsync', 'vfr',
           str(tmp_dir / '%06d.png')]
    subprocess.run(cmd, check=True)

def extract_frames_cv2(vpath: Path, tmp_dir: Path, skip: int):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        warnings.warn(f'Cannot open {vpath}')
        return
    idx, saved = 0, 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break
        if idx % skip == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            cv2.imwrite(str(tmp_dir / f'{idx:06d}.png'), frame)
            saved += 1
        idx += 1
    cap.release()

# ─── 비디오 하나 처리 ───────────────────────────────────────────
def process_video(vpath: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = max((int(p.stem) for p in out_dir.glob(f'*.{FMT.lower()}')), default=0) + 1

    # 1) 임시 폴더에 프레임 추출
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            if FRAME_METH.lower() == 'ffmpeg':
                extract_frames_ffmpeg(vpath, tmp_dir, SKIP)
            else:
                extract_frames_cv2(vpath, tmp_dir, SKIP)
        except subprocess.CalledProcessError as e:
            warnings.warn(f'ffmpeg error on {vpath.name}: {e}')
            return

        frame_paths = sorted(tmp_dir.glob('*.png'))
        if not frame_paths:
            warnings.warn(f'No frames extracted from {vpath.name}')
            return

        # 2) 얼굴 검출 & 저장
        p_faces = tqdm(frame_paths, desc=f'{vpath.stem} faces', leave=False, ncols=80)
        saved = 0
        for fpath in p_faces:
            pil_in = Image.open(fpath).convert('RGB')
            try:
                face = mtcnn(pil_in)
            except RuntimeError:
                torch.cuda.empty_cache()
                face = mtcnn(pil_in, device='cpu')

            if face is None:
                continue

            face_pil = tensor_to_pil(face)
            save_face(face_pil, out_dir / f'{seq:04d}.{FMT.lower()}')
            seq += 1
            saved += 1
            p_faces.set_postfix(saved=saved)

    print(f'  ↳ {vpath.name}: saved {seq-1} faces')

# ─── 전체 루프 ───────────────────────────────────────────────────
def main():
    for branch in BRANCHES:
        for method in METHODS:
            video_root = ROOT / branch / method / COMPR / 'videos'
            if not video_root.exists():
                print(f'[SKIP] {video_root} not found'); continue

            videos = [p for p in video_root.rglob('*') if p.suffix.lower() in VIDEO_EXT]
            if not videos:
                print(f'[SKIP] no videos in {video_root}'); continue

            for vp in tqdm(sorted(videos), desc=f'{method}', ncols=80):
                rel     = vp.relative_to(video_root).with_suffix('')
                out_dir = ROOT / branch / method / COMPR / 'mtcnn' / rel
                process_video(vp, out_dir)

    print('✓ 모든 얼굴 추출 완료')

if __name__ == '__main__':
    main()
