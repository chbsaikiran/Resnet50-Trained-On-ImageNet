# datamodule.py

import math
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import cv2
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------------ #
#  Training augmentations (OpenCV + NumPy only)                       #
# ------------------------------------------------------------------ #

def _random_resized_crop(img, crop_size=224,
                         scale=(0.08, 1.0),
                         ratio=(3.0/4.0, 4.0/3.0)):
    """Randomly crop a region with random scale & aspect ratio, then resize."""
    h, w, _ = img.shape
    area = h * w

    for _ in range(10):
        target_area = np.random.uniform(scale[0], scale[1]) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect = math.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))

        if 0 < crop_w <= w and 0 < crop_h <= h:
            x = np.random.randint(0, w - crop_w + 1)
            y = np.random.randint(0, h - crop_h + 1)
            img = img[y:y+crop_h, x:x+crop_w]
            return cv2.resize(img, (crop_size, crop_size),
                              interpolation=cv2.INTER_LINEAR)

    # Fallback: center crop at the largest inscribed rectangle
    in_ratio = w / h
    if in_ratio < ratio[0]:
        crop_w = w
        crop_h = int(round(w / ratio[0]))
    elif in_ratio > ratio[1]:
        crop_h = h
        crop_w = int(round(h * ratio[1]))
    else:
        crop_w, crop_h = w, h

    x = (w - crop_w) // 2
    y = (h - crop_h) // 2
    img = img[y:y+crop_h, x:x+crop_w]
    return cv2.resize(img, (crop_size, crop_size),
                      interpolation=cv2.INTER_LINEAR)


def _random_horizontal_flip(img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :].copy()
    return img


def _color_jitter(img, p=0.5, brightness=0.4, contrast=0.4,
                  saturation=0.4, hue=0.1):
    """Randomly perturb brightness, contrast, saturation, and hue."""
    if np.random.random() >= p:
        return img

    ops = [0, 1, 2, 3]
    np.random.shuffle(ops)

    for op in ops:
        if op == 0 and brightness > 0:
            factor = np.random.uniform(max(0, 1 - brightness),
                                       1 + brightness)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        elif op == 1 and contrast > 0:
            factor = np.random.uniform(max(0, 1 - contrast),
                                       1 + contrast)
            mean = img.mean()
            img = np.clip((img.astype(np.float32) - mean) * factor + mean,
                          0, 255).astype(np.uint8)

        elif op == 2 and saturation > 0:
            factor = np.random.uniform(max(0, 1 - saturation),
                                       1 + saturation)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        elif op == 3 and hue > 0:
            shift = np.random.uniform(-hue, hue) * 180
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return img


def _random_lighting(img, p=0.5, alpha_std=0.1):
    """PCA-based lighting noise (AlexNet-style). Operates on float [0,1] image."""
    if np.random.random() >= p:
        return img

    eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
    eigvec = np.array([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ], dtype=np.float32)

    alpha = np.random.normal(0, alpha_std, size=3).astype(np.float32)
    noise = eigvec @ (eigval * alpha)
    img = img + noise[np.newaxis, np.newaxis, :]
    return np.clip(img, 0.0, 1.0)


def _random_erasing(img, p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    """Randomly erase a rectangle and fill with per-channel mean."""
    if np.random.random() > p:
        return img

    c, h, w = img.shape
    area = h * w

    for _ in range(10):
        target_area = np.random.uniform(scale[0], scale[1]) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect = math.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

        eh = int(round(math.sqrt(target_area / aspect)))
        ew = int(round(math.sqrt(target_area * aspect)))

        if eh < h and ew < w:
            y = np.random.randint(0, h - eh)
            x = np.random.randint(0, w - ew)
            for ch in range(c):
                img[ch, y:y+eh, x:x+ew] = img[ch].mean()
            return img

    return img


# ------------------------------------------------------------------ #
#  Train / Val preprocessing pipelines                                #
# ------------------------------------------------------------------ #

def train_preprocess_opencv(img, crop_size=224):
    img = np.array(img)

    img = _color_jitter(img)
    img = _random_resized_crop(img, crop_size)
    img = _random_horizontal_flip(img)

    img = img.astype(np.float32) / 255.0
    img = _random_lighting(img)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1)).copy()

    img = _random_erasing(img)

    return torch.from_numpy(img)


def val_preprocess_opencv(img, resize_size=256, crop_size=224):
    img = np.array(img)
    h, w, _ = img.shape

    if h < w:
        new_h = resize_size
        new_w = int(w * resize_size / h)
    else:
        new_w = resize_size
        new_h = int(h * resize_size / w)

    img = cv2.resize(img, (new_w, new_h))
    start_x = (new_w - crop_size) // 2
    start_y = (new_h - crop_size) // 2

    img = img[start_y:start_y + crop_size,
              start_x:start_x + crop_size]

    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))

    return torch.from_numpy(img)


class ImageNet10(ImageFolder):
    def __init__(self, root, wnid_map, transform=None):
        super().__init__(root, transform=transform)

        self.samples = [
            (path, wnid_map[self.classes[label]])
            for path, label in self.samples
            if self.classes[label] in wnid_map
        ]


class ImageNetDataModule(pl.LightningDataModule):

    def __init__(self, train_dir, val_dir, wnid_map, batch_size=64):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.wnid_map = wnid_map
        self.batch_size = batch_size

    def setup(self, stage=None):

        self.train_dataset = ImageNet10(
            self.train_dir,
            self.wnid_map,
            transform=train_preprocess_opencv
        )

        self.val_dataset = ImageNet10(
            self.val_dir,
            self.wnid_map,
            transform=val_preprocess_opencv
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)
