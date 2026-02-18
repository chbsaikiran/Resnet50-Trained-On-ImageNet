# transforms.py
import cv2
import torch
import numpy as np

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

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (new_w - crop_size) // 2
    start_y = (new_h - crop_size) // 2

    img = img[start_y:start_y+crop_size,
              start_x:start_x+crop_size]

    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img = np.transpose(img, (2, 0, 1))

    return torch.from_numpy(img)
