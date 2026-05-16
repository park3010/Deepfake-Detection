#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비디오에서 일정 간격으로 프레임을 추출하고 리사이즈하는 스크립트
Usage: 코드 상단의 INPUT_DIR, OUTPUT_DIR, SKIP, RESIZE 등을 설정하고 실행하세요.
"""

import subprocess
import cv2
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp

# ─────────────────────────────────────────────────────────────────────────────
# 설정
INPUT_DIR   = Path('/home/oem/deepfake/data/DeepfakeTIMIT')    # 비디오 입력 폴더
OUTPUT_DIR  = Path('/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT')  # 추출된 프레임 저장 폴더
SKIP        = 5               # 몇 프레임마다 1프레임 저장할지
RESIZE      = (224, 224)      # 저장 전 리사이즈할 크기 (width, height). None으로 두면 리사이즈 안 함
METHOD      = 'ffmpeg'        # 'ffmpeg' 또는 'cv2'
JOBS        = mp.cpu_count()  # 병렬 작업 수
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}

def extract_ffmpeg(video_path: Path, out_dir: Path, skip: int, threads: int, resize: tuple):
    out_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg -vf "select='not(mod(n\,SKIP))',scale=WIDTH:HEIGHT"
    vf_filters = [f"select='not(mod(n\\,{skip}))'"]
    if resize:
        w, h = resize
        vf_filters.append(f"scale={w}:{h}")
    vf = ",".join(vf_filters)

    cmd = [
        'ffmpeg', '-y',
        '-v', 'quiet',
        '-threads', str(threads),
        '-i', str(video_path),
        '-vf', vf,
        '-vsync', 'vfr',
        str(out_dir / '%04d.png')
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] FFmpeg failed for {video_path.name}: {e}")

def extract_cv2(video_path: Path, out_dir: Path, skip: int, resize: tuple):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_num = 0
    save_idx  = 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break
        if frame_num % skip == 0:
            success, frame = cap.retrieve()
            if not success:
                break
            # 리사이즈 적용
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{save_idx:04d}.png"), frame)
            save_idx += 1
        frame_num += 1
    cap.release()

def process_one(video_path: Path):
    rel = video_path.relative_to(INPUT_DIR)
    out_subdir = OUTPUT_DIR / rel.with_suffix('')
    if METHOD == 'ffmpeg':
        extract_ffmpeg(video_path, out_subdir, SKIP, max(1, JOBS//2), RESIZE)
    else:
        extract_cv2(video_path, out_subdir, SKIP, RESIZE)

def main():
    # 1) 비디오 파일 목록 수집
    videos = [p for p in INPUT_DIR.rglob('*') if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print("입력 디렉터리에 비디오 파일이 없습니다:", INPUT_DIR)
        return

    # 2) 병렬 처리
    Parallel(n_jobs=JOBS)(
        delayed(process_one)(vp)
        for vp in tqdm(videos, desc="Extracting frames", ncols=80)
    )

    print("모두 완료되었습니다.")

if __name__ == '__main__':
    main()
