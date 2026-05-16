import os
import glob
import pandas as pd

FFPP_ROOT = "/home/oem/deepfake/hdd"  # 또는 실제 FF++ root
COMPRESSION = "raw"

def count_ffpp(root, compression="raw"):
    rows = []

    for split_name, label in [
        ("original_sequences", 0),
        ("manipulated_sequences", 1),
    ]:
        base = os.path.join(root, split_name)

        if not os.path.isdir(base):
            continue

        for method in sorted(os.listdir(base)):
            method_dir = os.path.join(base, method, compression, "mtcnn")
            if not os.path.isdir(method_dir):
                continue

            video_count = 0
            frame_count = 0

            for vid in os.listdir(method_dir):
                vid_dir = os.path.join(method_dir, vid)
                if not os.path.isdir(vid_dir):
                    continue

                frames = []
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    frames.extend(glob.glob(os.path.join(vid_dir, ext)))

                if frames:
                    video_count += 1
                    frame_count += len(frames)

            rows.append({
                "split": split_name,
                "method": method,
                "label": "real" if label == 0 else "fake",
                "videos": video_count,
                "frames": frame_count,
            })

    df = pd.DataFrame(rows)
    print(df)
    print("\n[Summary]")
    print(df.groupby("label")[["videos", "frames"]].sum())
    df.to_csv("ffpp_train_dataset_statistics.csv", index=False)

count_ffpp(FFPP_ROOT, COMPRESSION)