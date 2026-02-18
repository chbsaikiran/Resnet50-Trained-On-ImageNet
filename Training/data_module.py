# datamodule.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import cv2
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_opencv(img, resize_size=256, crop_size=224):
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
            transform=preprocess_opencv
        )

        self.val_dataset = ImageNet10(
            self.val_dir,
            self.wnid_map,
            transform=preprocess_opencv
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
