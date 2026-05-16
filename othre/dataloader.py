import os
import cv2
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import random

from torch.utils.data import Dataset, DataLoader
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import random_split

class FFPPFrameDataset(Dataset):
    """
    FaceForensics++ full_images 폴더를 재귀 탐색하여
    original_sequences/*/{compression}/full_images/** 에 있는 이미지는 label=0 (real),
    manipulated_sequences/*/{compression}/full_images/** 에 있는 이미지는 label=1 (fake)
    로 취급합니다.
    """
    def __init__(self, root_dir, compression='c23', transform=None):
        self.samples = []
        self.transform = transform

        # real (original_sequences)
        real_root = os.path.join(root_dir, 'original_sequences')
        for method in os.listdir(real_root):
            full_dir = os.path.join(real_root, method, compression, 'faces')
            if not os.path.isdir(full_dir):
                continue
            # full_images 이하 모든 서브폴더 재귀 탐색
            for subdir, _, files in os.walk(full_dir):
                for fname in files:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(subdir, fname), 0))

        # fake (manipulated_sequences)
        fake_root = os.path.join(root_dir, 'manipulated_sequences')
        for method in os.listdir(fake_root):
            full_dir = os.path.join(fake_root, method, compression, 'faces')
            if not os.path.isdir(full_dir):
                continue
            for subdir, _, files in os.walk(full_dir):
                for fname in files:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(subdir, fname), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def extract_dct(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = np.float32(image_gray) / 255.0
    dct = cv2.dct(image_gray)
    return dct


def extract_fft(image_path: str) -> np.ndarray:
    """
    주어진 컬러 이미지를 읽어서 각 채널별 2D FFT의 magnitude spectrum을 계산한 뒤,
    3채널 배열로 합쳐서 반환합니다.
    
    Args:
        image_path (str): 입력 이미지 파일 경로.
    
    Returns:
        np.ndarray: shape=(H, W, 3), dtype=float32. 각 채널의 로그 스케일 magnitude spectrum.
    """
    # 1. 이미지 로드 및 BGR→Gray 채널 분리
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)
    
    # 2. 각 채널별 DFT→shift→magnitude(log) 계산
    def channel_fft(ch: np.ndarray) -> np.ndarray:
        dft = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        # 로그 스케일 변환, +1로 로그(0) 방지
        return 20 * np.log(mag + 1)
    
    mag_b = channel_fft(b)
    mag_g = channel_fft(g)
    mag_r = channel_fft(r)
    
    # 3. 3채널로 병합 (PIL 사용)
    im_b = Image.fromarray(np.uint8(np.clip(mag_b, 0, 255))).convert('L')
    im_g = Image.fromarray(np.uint8(np.clip(mag_g, 0, 255))).convert('L')
    im_r = Image.fromarray(np.uint8(np.clip(mag_r, 0, 255))).convert('L')
    merged = Image.merge("RGB", (im_b, im_g, im_r))
    
    # 4. numpy array로 반환 (H, W, 3)
    return np.array(merged)