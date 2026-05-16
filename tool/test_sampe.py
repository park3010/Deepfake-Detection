import os
import glob
import pandas as pd

DATASETS = {
    "Celeb-DF": {
        "real": [
            "/mnt/server1_test/Celeb/Celeb-real",
            "/mnt/server1_test/Celeb/YouTube-real",
        ],
        "fake": [
            "/mnt/server1_test/Celeb/Celeb-synthesis",
        ],
    },
    "DFD": {
        "real": [
            "/mnt/server1_test/DFD/DFD_original_sequences",
        ],
        "fake": [
            "/mnt/server1_test/DFD/DFD_manipulated_sequences",
        ],
    },
    "DeepfakeTIMIT": {
        "real": [],
        "fake": [
            "/mnt/server1_test/DeepfakeTIMIT/higher_quality",
            "/mnt/server1_test/DeepfakeTIMIT/lower_quality",
        ],
    },
    "WildDeepfake": {
        "root": "/mnt/server1_test/WildDeepfake",
        "splits": ["train", "test"],
    },
}

def count_videos_and_frames(roots):
    video_count = 0
    frame_count = 0

    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] Missing path: {root}")
            continue

        for vid in sorted(os.listdir(root)):
            vid_dir = os.path.join(root, vid)
            if not os.path.isdir(vid_dir):
                continue

            frames = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                frames.extend(glob.glob(os.path.join(vid_dir, ext)))

            if len(frames) > 0:
                video_count += 1
                frame_count += len(frames)

    return video_count, frame_count

rows = []

for name, cfg in DATASETS.items():
    if name == "WildDeepfake":
        real_roots, fake_roots = [], []

        for split in cfg["splits"]:
            split_dir = os.path.join(cfg["root"], split)
            if not os.path.isdir(split_dir):
                continue

            for method in os.listdir(split_dir):
                base = os.path.join(split_dir, method)
                r = os.path.join(base, "real")
                f = os.path.join(base, "fake")

                if os.path.isdir(r):
                    real_roots.append(r)
                if os.path.isdir(f):
                    fake_roots.append(f)

    elif name == "DeepfakeTIMIT":
        real_roots = []
        fake_roots = []

        for quality_root in cfg["fake"]:
            if not os.path.isdir(quality_root):
                continue

            for speaker in os.listdir(quality_root):
                sp_path = os.path.join(quality_root, speaker)
                if os.path.isdir(sp_path):
                    fake_roots.append(sp_path)

    else:
        real_roots = cfg["real"]
        fake_roots = cfg["fake"]

    real_v, real_f = count_videos_and_frames(real_roots)
    fake_v, fake_f = count_videos_and_frames(fake_roots)

    rows.append({
        "dataset": name,
        "real_videos": real_v,
        "fake_videos": fake_v,
        "total_videos": real_v + fake_v,
        "real_frames": real_f,
        "fake_frames": fake_f,
        "total_frames": real_f + fake_f,
    })

df = pd.DataFrame(rows)
print(df)
df.to_csv("test_dataset_statistics.csv", index=False)