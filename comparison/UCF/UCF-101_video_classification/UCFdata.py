import os
import glob
import random
import numpy as np
import pandas as pd
from processor import process_image
from keras.utils import np_utils

# -----------------------------
# FF++ train dataset
# -----------------------------
FFPP_ROOT = "/home/oem/deepfake/FF++"   # 네 환경에 맞게 수정
FFPP_COMPRESSION = "c23"
FFPP_REAL_SUBDIR = "original_sequences"
FFPP_FAKE_SUBDIR = "manipulated_sequences"
FFPP_FACE_DIRNAME = "faces"   # 네 코드 기준 faces 사용

# -----------------------------
# test-only datasets
# -----------------------------
TEST_DATASETS = {
    "Celeb": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-real",
            "/home/oem/deepfake/hdd_5TB/Celeb/YouTube-real"
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-synthesis"
        ],
    },
    "DFD": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences"
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences"
        ],
    },
    "DeepfakeTIMIT": {
        "real": [],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality"
        ],
    },
    "WildDeepfake": {
        "root": "/home/oem/deepfake/hdd_5TB/WildDeepfake",
        "splits": ["train", "test"],   # 여기서는 둘 다 test 취급할 수도 있음
    },
}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _dir_has_images(path):
    if not os.path.isdir(path):
        return False
    return any(f.lower().endswith(IMG_EXTS) for f in os.listdir(path))


def _find_video_dirs_recursively(root):
    vids = []
    for cur, _dirs, files in os.walk(root):
        if any(f.lower().endswith(IMG_EXTS) for f in files):
            vids.append(cur)
    return sorted(list(set(vids)))


def _count_frames_in_dir(vid_dir):
    frames = []
    for ext in IMG_EXTS:
        frames += glob.glob(os.path.join(vid_dir, f"*{ext}"))
    return len(frames)


class DataSet():

    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = './data/sequences/'
        self.max_frames = 300

        self.data = self.get_data()
        self.classes = self.get_classes()
        self.data = self.clean_data()
        self.image_shape = image_shape

    @staticmethod
    def get_data():
        data = []

        # =========================================================
        # 1) TRAIN: FF++ only
        # =========================================================
        # real
        ffpp_real_root = os.path.join(FFPP_ROOT, FFPP_REAL_SUBDIR)
        if os.path.isdir(ffpp_real_root):
            for method in os.listdir(ffpp_real_root):
                face_root = os.path.join(
                    ffpp_real_root, method, FFPP_COMPRESSION, FFPP_FACE_DIRNAME
                )
                if not os.path.isdir(face_root):
                    continue

                for vid_dir in _find_video_dirs_recursively(face_root):
                    n_frames = _count_frames_in_dir(vid_dir)
                    if n_frames > 0:
                        data.append([
                            "train",                      # split
                            "real",                       # class
                            os.path.basename(vid_dir),    # video id
                            n_frames,                     # number of frames
                            vid_dir                       # actual directory path
                        ])

        # fake
        ffpp_fake_root = os.path.join(FFPP_ROOT, FFPP_FAKE_SUBDIR)
        if os.path.isdir(ffpp_fake_root):
            for method in os.listdir(ffpp_fake_root):
                face_root = os.path.join(
                    ffpp_fake_root, method, FFPP_COMPRESSION, FFPP_FACE_DIRNAME
                )
                if not os.path.isdir(face_root):
                    continue

                for vid_dir in _find_video_dirs_recursively(face_root):
                    n_frames = _count_frames_in_dir(vid_dir)
                    if n_frames > 0:
                        data.append([
                            "train",
                            "fake",
                            os.path.basename(vid_dir),
                            n_frames,
                            vid_dir
                        ])

        # =========================================================
        # 2) TEST: external datasets only
        # =========================================================
        for ds_name, cfg in TEST_DATASETS.items():

            if ds_name == "WildDeepfake":
                root = cfg["root"]
                for split in cfg["splits"]:
                    split_root = os.path.join(root, split)
                    if not os.path.isdir(split_root):
                        continue

                    for method in os.listdir(split_root):
                        base = os.path.join(split_root, method)

                        for cls in ["real", "fake"]:
                            cls_root = os.path.join(base, cls)
                            if not os.path.isdir(cls_root):
                                continue

                            for vid_dir in _find_video_dirs_recursively(cls_root):
                                n_frames = _count_frames_in_dir(vid_dir)
                                if n_frames > 0:
                                    data.append([
                                        "test",
                                        cls,
                                        os.path.basename(vid_dir),
                                        n_frames,
                                        vid_dir
                                    ])
            else:
                for cls in ["real", "fake"]:
                    for root in cfg.get(cls, []):
                        if not os.path.isdir(root):
                            continue

                        subdirs = [
                            os.path.join(root, d)
                            for d in os.listdir(root)
                            if os.path.isdir(os.path.join(root, d))
                        ]

                        vid_dirs = [p for p in subdirs if _dir_has_images(p)]
                        if not vid_dirs:
                            vid_dirs = _find_video_dirs_recursively(root)

                        for vid_dir in vid_dirs:
                            n_frames = _count_frames_in_dir(vid_dir)
                            if n_frames > 0:
                                data.append([
                                    "test",
                                    cls,
                                    os.path.basename(vid_dir),
                                    n_frames,
                                    vid_dir
                                ])

        return data

    def clean_data(self):
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)
        return data_clean

    def get_classes(self):
        return ['real', 'fake']

    def get_class_one_hot(self, class_str):
        label_encoded = self.classes.index(class_str)
        label_hot = np_utils.to_categorical(label_encoded, len(self.classes))
        return label_hot[0]

    def split_train_test(self):
        train = [item for item in self.data if item[0] == 'train']
        test = [item for item in self.data if item[0] == 'test']
        return train, test

    def build_image_sequence(self, frames):
        return [process_image(x, self.image_shape) for x in frames]

    def frame_generator(self, batch_size, train_test, data_type, concat=False):
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print(f"Creating {train_test} generator with {len(data)} samples.")

        while 1:
            X, y = [], []
            for _ in range(batch_size):
                sample = random.choice(data)

                if data_type == "images":
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)
                    sequence = self.build_image_sequence(frames)
                else:
                    raise NotImplementedError("features mode not used for this dataset")

                if concat:
                    sequence = np.concatenate(sequence).ravel()

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    @staticmethod
    def get_frames_for_sample(sample):
        vid_dir = sample[4]
        images = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            images += glob.glob(os.path.join(vid_dir, ext))
        return sorted(images)

    @staticmethod
    def rescale_list(input_list, size):
        assert len(input_list) >= size

        skip = len(input_list) // size
        output = [input_list[i] for i in range(0, len(input_list), skip)]
        return output[:size]