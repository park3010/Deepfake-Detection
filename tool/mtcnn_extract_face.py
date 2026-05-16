"""
Extracts faces from selected datasets using MTCNN detect→crop approach,
optimized for maximum throughput with GPU acceleration fallback to CPU,
resume support, clip-level skipping, and specific GPU selection.

Key optimizations:
 - Dynamic chunked batches to avoid OOM
 - DataLoader with high num_workers for parallel I/O
 - Mixed-precision inference (torch.amp.autocast)
 - Clip + frame-level skipping with START_CLIP
 - Safe CPU fallback without unexpected args
 - Bounding box validity checks to avoid empty crops

Usage:
  - Ensure ROOT and OUTPUT_ROOT are correctly set.
  - Outputs always saved in 'mtcnn' folder under each clip.

Author: Enhanced error handling for empty crops (2025-06-17)
"""
import os
import cv2
import torch
from torch.amp import autocast
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from facenet_pytorch import MTCNN
from tqdm import tqdm

# Prevent CUDA fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Settings
ROOT = Path('/home/oem/deepfake/hdd_5TB/FF++')
COMPRESSION = 'c40'
OUTPUT_ROOT = Path('/home/oem/deepfake/hdd_5TB/FF++')
DATASET_PATHS = {
    # "NeuralTextures": "manipulated_sequences/NeuralTextures",
    # "FaceSwap":         "manipulated_sequences/FaceSwap",
    # "FaceShifter":         "manipulated_sequences/FaceShifter",
    # "Face2Face":         "manipulated_sequences/Face2Face",
    # "Deepfakes":         "manipulated_sequences/Deepfakes",
    "DeepFakeDetection":         "manipulated_sequences/DeepFakeDetection",
}

# Batch and processing settings
BATCH_SIZE = 64       # frames per batch for detection
CHUNK_SIZE = 16       # chunk size within batch to avoid OOM
GPU_ID = 0            # GPU index to use

DEVICE = f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Initialize MTCNN with safe fallback
def init_mtcnn(device):
    try:
        return MTCNN(image_size=224, margin=0, keep_all=False,
                     select_largest=True, device=device)
    except RuntimeError:
        torch.cuda.empty_cache()
        return MTCNN(image_size=224, margin=0, keep_all=False,
                     select_largest=True, device='cpu')

mtcnn = init_mtcnn(DEVICE)

class FrameDataset(IterableDataset):
    def __init__(self, paths): self.paths = paths
    def __iter__(self):
        yield from self.paths

# Process a single clip with detailed progress
def process_clip(clip_dir, out_clip_dir,
                 batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE,
                 num_workers=None):
    if num_workers is None:
        num_workers = max(1, os.cpu_count() // 2)
    out_clip_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted([p for p in clip_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    done = {f.name for f in out_clip_dir.iterdir() if f.is_file()}
    todo = [f for f in frames if f.name not in done]
    if not todo:
        return

    mtcnn_cpu = None
    loader = DataLoader(
        FrameDataset(todo), batch_size=batch_size,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda x: x
    )

    # Batch-level progress
    for paths in tqdm(loader, desc=f"Clip {clip_dir.name}", leave=False):
        imgs = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not imgs:
            continue

        # Chunk-level processing with safe fallback
        for i in range(0, len(imgs), chunk_size):
            chunk = imgs[i:i+chunk_size]
            try:
                with autocast(device_type='cuda', enabled=DEVICE.startswith('cuda')):
                    boxes_list, _ = mtcnn.detect(chunk)
            except RuntimeError:
                torch.cuda.empty_cache()
                if mtcnn_cpu is None:
                    mtcnn_cpu = init_mtcnn('cpu')
                boxes_list, _ = mtcnn_cpu.detect(chunk)

            for j, boxes in enumerate(boxes_list):
                if boxes is None or len(boxes) == 0:
                    continue
                x1, y1, x2, y2 = boxes[0].astype(int)
                # Validate bounding box
                h = y2 - y1
                w = x2 - x1
                if h <= 0 or w <= 0:
                    continue
                crop = chunk[j][y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                face = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                save_path = out_clip_dir / paths[i+j].name
                cv2.imwrite(str(save_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

# Main pipeline with dataset-level progress
if __name__ == '__main__':
    num_workers = max(1, os.cpu_count() // 2)
    START_CLIP = '28_16__walking_outside_cafe_disgusted__NAZP864W'
    started = False
    for key, rel in tqdm(DATASET_PATHS.items(), desc="Datasets", leave=True):
        base = ROOT / rel / COMPRESSION / 'full_images'
        if not base.exists():
            continue
        clips = sorted([d for d in base.iterdir() if d.is_dir()])
        for clip in tqdm(clips, desc=f"Clips {key}", leave=False):
            if START_CLIP and not started:
                if clip.name != START_CLIP:
                    continue
                started = True
                START_CLIP = None
            out_dir = OUTPUT_ROOT / rel / COMPRESSION / 'mtcnn' / clip.name
            process_clip(clip, out_dir, batch_size=BATCH_SIZE,
                         chunk_size=CHUNK_SIZE, num_workers=num_workers)
    print("Done.")
