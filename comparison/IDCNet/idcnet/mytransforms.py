import albumentations as albu
import numpy as np
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DefaultTransform:
    """
    val / test 용 기본 transform
    """
    def __init__(self):
        self.trans = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, sample):
        return self.trans(sample)


class Transform2:
    """
    Albumentations 기반 augmentation
    """
    def __init__(self):
        self.trans = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.Rotate(limit=15, p=0.5),
            albu.GaussianBlur(blur_limit=(3, 5), p=0.3),
            albu.OneOf([
                albu.RandomBrightnessContrast(p=1.0),
                albu.HueSaturationValue(p=1.0),
            ], p=0.3),
            albu.ImageCompression(quality_lower=40, quality_upper=100, p=0.3),
            albu.Resize(256, 256),
            albu.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.to_tensor = T.ToTensor()

    def __call__(self, sample):
        img = np.array(sample)
        img = self.trans(image=img)["image"]
        img = self.to_tensor(img)
        return img


class Transform3:
    """
    torchvision 기반 augmentation
    """
    def __init__(self):
        self.trans = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, sample):
        return self.trans(sample)


class Transform4:
    """
    가장 간단한 train augmentation
    """
    def __init__(self):
        self.trans = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, sample):
        return self.trans(sample)