import os
import glob
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

import mytransforms as mtfs


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# -----------------------------
# FF++ train/val dataset
# -----------------------------
FFPP_ROOT = "/home/oem/deepfake/hdd"   # 수정 필요
FFPP_COMPRESSION = "raw"
FFPP_FACE_DIRNAME = "mtcnn"             # full_images면 바꾸기

# -----------------------------
# external test datasets
# -----------------------------
TEST_DATASETS = {
    "Celeb": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-real",
            "/home/oem/deepfake/hdd_5TB/Celeb/YouTube-real",
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-synthesis",
        ],
    },
    "DFD": {
        "real": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences",
        ],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences",
        ],
    },
    "DeepfakeTIMIT": {
        "real": [],
        "fake": [
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality",
        ],
    },
    "WildDeepfake": {
        "root": "/home/oem/deepfake/hdd_5TB/WildDeepfake",
        "splits": ["train", "test"],
    },
}


def flatten_list(list_of_lists):
    out = []
    for x in list_of_lists:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def _is_image_file(fname):
    return fname.lower().endswith(IMG_EXTS)


def _find_all_image_files(root):
    imgs = []
    for cur, _, files in os.walk(root):
        for f in files:
            if _is_image_file(f):
                imgs.append(os.path.join(cur, f))
    return sorted(imgs)


def build_external_test_dict():
    """
    TestDataset1용 test_dict 생성
    key 이름에 'real' 포함 -> label 0
    아니면 -> label 1
    """
    test_dict = {
        "Celeb_real": {"test": []},
        "Celeb_fake": {"test": []},
        "DFD_real": {"test": []},
        "DFD_fake": {"test": []},
        "DeepfakeTIMIT_fake": {"test": []},
        "WildDeepfake_real": {"test": []},
        "WildDeepfake_fake": {"test": []},
    }

    datasets = {
        "Celeb_real": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-real",
            "/home/oem/deepfake/hdd_5TB/Celeb/YouTube-real",
        ],
        "Celeb_fake": [
            "/home/oem/deepfake/hdd_5TB/Celeb/Celeb-synthesis",
        ],
        "DFD_real": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_original_sequences",
        ],
        "DFD_fake": [
            "/home/oem/deepfake/hdd_5TB/DFD/DFD_manipulated_sequences",
        ],
        "DeepfakeTIMIT_fake": [
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/higher_quality",
            "/home/oem/deepfake/hdd_5TB/DeepfakeTIMIT/lower_quality",
        ],
    }

    for key, roots in datasets.items():
        for root in roots:
            if not os.path.isdir(root):
                continue
            imgs = _find_all_image_files(root)
            if imgs:
                test_dict[key]["test"].append(imgs)

    wild_root = TEST_DATASETS["WildDeepfake"]["root"]
    for split in TEST_DATASETS["WildDeepfake"]["splits"]:
        split_root = os.path.join(wild_root, split)
        if not os.path.isdir(split_root):
            continue

        for method in os.listdir(split_root):
            base = os.path.join(split_root, method)
            for cls in ["real", "fake"]:
                cls_root = os.path.join(base, cls)
                if not os.path.isdir(cls_root):
                    continue

                imgs = _find_all_image_files(cls_root)
                if imgs:
                    key = f"WildDeepfake_{cls}"
                    test_dict[key]["test"].append(imgs)

    return test_dict


class TrainDataset1(Dataset):
    """
    FF++ only for training/validation
    real=0, fake=1
    """
    def __init__(self, mode='train', balanced=False, transform=None, normalize=None):
        super().__init__()
        self.real_data_list = []
        self.fake_data_list = []
        self.data_list = []
        self.label_list = []

        self.transform = transform if transform is not None else mtfs.DefaultTransform()
        self.normalize = normalize

        self.ffpp_root = FFPP_ROOT
        self.compression = FFPP_COMPRESSION
        self.face_dirname = FFPP_FACE_DIRNAME

        self.load_ffpp_data(mode=mode, balanced=balanced)

    def load_ffpp_data(self, mode='train', balanced=False):
        real_root = os.path.join(self.ffpp_root, "original_sequences")
        fake_root = os.path.join(self.ffpp_root, "manipulated_sequences")

        all_real = []
        all_fake = []

        if os.path.isdir(real_root):
            for method in os.listdir(real_root):
                face_root = os.path.join(real_root, method, self.compression, self.face_dirname)
                if not os.path.isdir(face_root):
                    continue
                all_real.extend(_find_all_image_files(face_root))

        if os.path.isdir(fake_root):
            for method in os.listdir(fake_root):
                face_root = os.path.join(fake_root, method, self.compression, self.face_dirname)
                if not os.path.isdir(face_root):
                    continue
                all_fake.extend(_find_all_image_files(face_root))

        random.seed(42)
        random.shuffle(all_real)
        random.shuffle(all_fake)

        r_split = int(len(all_real) * 0.8)
        f_split = int(len(all_fake) * 0.8)

        if mode == 'train':
            self.real_data_list = all_real[:r_split]
            self.fake_data_list = all_fake[:f_split]
        elif mode == 'val':
            self.real_data_list = all_real[r_split:]
            self.fake_data_list = all_fake[f_split:]
        else:
            raise ValueError("mode should be 'train' or 'val'")

        if balanced:
            n = min(len(self.real_data_list), len(self.fake_data_list))
            self.real_data_list = self.real_data_list[:n]
            self.fake_data_list = self.fake_data_list[:n]

        self.data_list = self.real_data_list + self.fake_data_list
        self.label_list = [0] * len(self.real_data_list) + [1] * len(self.fake_data_list)

        print(
            f"TrainDataset1[{mode}] "
            f"real={len(self.real_data_list)} fake={len(self.fake_data_list)} total={len(self.data_list)}"
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.normalize is not None:
            img = self.normalize(img)

        label = torch.tensor(self.label_list[index], dtype=torch.long)
        return img, label


class TrainDatasetVedioID(Dataset):
    """
    FF++ only for training/validation, returns video_id
    """
    def __init__(self, mode='train', transform=None, normalize=None):
        super().__init__()
        self.data_list = []
        self.label_list = []

        self.transform = transform if transform is not None else mtfs.DefaultTransform()
        self.normalize = normalize

        self.ffpp_root = FFPP_ROOT
        self.compression = FFPP_COMPRESSION
        self.face_dirname = FFPP_FACE_DIRNAME

        self.load_ffpp_data(mode=mode)

    def load_ffpp_data(self, mode='train'):
        real_list = []
        fake_list = []

        real_root = os.path.join(self.ffpp_root, "original_sequences")
        fake_root = os.path.join(self.ffpp_root, "manipulated_sequences")

        if os.path.isdir(real_root):
            for method in os.listdir(real_root):
                face_root = os.path.join(real_root, method, self.compression, self.face_dirname)
                if not os.path.isdir(face_root):
                    continue
                real_list.extend(_find_all_image_files(face_root))

        if os.path.isdir(fake_root):
            for method in os.listdir(fake_root):
                face_root = os.path.join(fake_root, method, self.compression, self.face_dirname)
                if not os.path.isdir(face_root):
                    continue
                fake_list.extend(_find_all_image_files(face_root))

        random.seed(42)
        random.shuffle(real_list)
        random.shuffle(fake_list)

        r_split = int(len(real_list) * 0.8)
        f_split = int(len(fake_list) * 0.8)

        if mode == 'train':
            self.data_list = real_list[:r_split] + fake_list[:f_split]
            self.label_list = [0] * len(real_list[:r_split]) + [1] * len(fake_list[:f_split])
        elif mode == 'val':
            self.data_list = real_list[r_split:] + fake_list[f_split:]
            self.label_list = [0] * len(real_list[r_split:]) + [1] * len(fake_list[f_split:])
        else:
            raise ValueError("mode should be 'train' or 'val'")

        print(
            f"TrainDatasetVedioID[{mode}] "
            f"real={sum(1 for x in self.label_list if x == 0)} "
            f"fake={sum(1 for x in self.label_list if x == 1)} "
            f"total={len(self.data_list)}"
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.normalize is not None:
            img = self.normalize(img)

        label = torch.tensor(self.label_list[index], dtype=torch.long)

        video_str = Path(img_path).parts[-2]
        try:
            video_id = torch.tensor(int(video_str[:3]), dtype=torch.long)
        except Exception:
            video_id = torch.tensor(index, dtype=torch.long)

        return img, label, video_id


class TestDataset1(Dataset):
    """
    External test dataset
    real=0, fake=1
    """
    def __init__(self, test_dict=None, transform=None, normalize=None):
        super().__init__()
        self.data_list = []
        self.label_list = []

        self.transform = transform if transform is not None else mtfs.DefaultTransform()
        self.normalize = normalize

        if test_dict is None:
            test_dict = build_external_test_dict()

        self.load_data(test_dict)

    def load_data(self, data_dict):
        for key, value in data_dict.items():
            flat_list = flatten_list(value['test'])
            self.data_list += flat_list
            label = 0 if 'real' in key.lower() else 1
            self.label_list += [label] * len(flat_list)

        print(
            f"TestDataset1 total={len(self.data_list)} "
            f"real={sum(1 for x in self.label_list if x == 0)} "
            f"fake={sum(1 for x in self.label_list if x == 1)}"
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.normalize is not None:
            img = self.normalize(img)

        label = torch.tensor(self.label_list[index], dtype=torch.long)
        return img, label