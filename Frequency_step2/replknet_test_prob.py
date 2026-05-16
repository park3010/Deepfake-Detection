import sys
from pathlib import Path

import torch
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from tqdm import tqdm

# 로컬 모듈 import 경로
sys.path.append('/home/oem/deepfake/Ourmethod/Frequency_step2')
from models.replknet import create_RepLKNet31B

# GPU 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 1) 모델 초기화
model = create_RepLKNet31B(
    num_classes=2,
    in_channels=3,
    use_cbam=False
)

# 2) 체크포인트 불러오기
ckpt_path = '/home/oem/deepfake/Ourmethod/Frequency_step2/checkpoints/replknet31b_best.pth'
state = torch.load(ckpt_path, map_location=device)
new_state = {k.replace('model.', ''): v for k, v in state.items()}

# stem 채널 mismatch(4→3) 처리
stem_w = new_state.get('stem.0.0.0.weight')
if stem_w is not None and stem_w.shape[1] == 4:
    new_state['stem.0.0.0.weight'] = stem_w[:, :3, :, :].clone()
    print("⚙️  Adjusted stem.0.0.0.weight from 4→3 channels")

# 3) state_dict 로드
model.load_state_dict(new_state, strict=False)
model.to(device)
model.eval()

# 4) 전처리 정의
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# 5) 테스트 프레임 폴더
TEST_DIR = Path('/home/oem/deepfake/hdd/test_sample_frames_10')
print(f"Processing videos in: {TEST_DIR}")

results = []
# 전체 비디오 진행률 표시
for video_dir in tqdm(sorted(TEST_DIR.iterdir()), desc="Videos"):
    if not video_dir.is_dir():
        continue
    vid = video_dir.name + '.mp4'  # 비디오 ID

    # 폴더 내 이미지 리스트
    img_paths = sorted([
        p for p in video_dir.iterdir()
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ])
    probs = []
    # 프레임별 진행률 표시
    for img_path in tqdm(img_paths, desc=f"  {vid}", leave=False, ncols=80):
        img = Image.open(img_path).convert('RGB')
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            prob1 = torch.softmax(out, dim=1)[0, 1].item()
        probs.append(prob1)

    avg_prob = float(sum(probs) / len(probs)) if probs else 0.0
    label = 0 if avg_prob > 0.5 else 1
    results.append((vid, label))

# 6) CSV 저장
out_csv = Path('/home/oem/deepfake/Ourmethod/Frequency_step2/test_csv/replknet_frame10.csv')
out_csv.parent.mkdir(exist_ok=True, parents=True)
df = pd.DataFrame(results, columns=['ID', 'label'])
df.to_csv(out_csv, index=False)
print(f"✅ Saved submission to {out_csv}")
