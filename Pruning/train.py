# train.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import *
from datamodule import ImageNetDataModule
from lightning_module import PruningDistillationModule
from model import BasicBlock,Bottleneck,ResNet
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def find_lr(model, train_loader, criterion, optimizer, device):
    model.train()
    lrs = []
    losses = []

    min_lr = 1e-6
    max_lr = 1
    num_steps = len(train_loader)

    for i, (inputs, labels) in enumerate(train_loader):

        if i >= num_steps:
            break

        # Exponential LR increase
        lr = min_lr * (max_lr / min_lr) ** (i / num_steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())

    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('LR Finder')
    plt.show()

    return lrs, losses

def load_full_model(path, device):
    model = ResNet(
        block=BasicBlock,
        layers=[2,2,2,2],
        num_classes=10
    )

    #model = torch.load(path, map_location='cpu')
    state_dict = torch.load(path, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    #model.load_state_dict(state_dict)

    model.to(device)
    return model

def load_any_model(path, device):

    ckpt = torch.load(path, map_location="cpu")

    model_class = ckpt["model_class"]
    model_kwargs = ckpt["model_kwargs"]

    model = model_class(**model_kwargs)
    model.load_state_dict(ckpt["state_dict"])

    model.to(device)
    return model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = load_full_model(TEACHER_PATH, device)
    student = load_full_model(TEACHER_PATH, device)

    wnid_map = {
        "n02077923": 0,  # sea lion
        "n02058221": 1,  # albatross
        "n02051845": 2,  # pelican
        "n02037110": 3,  # oystercatcher
        "n02028035": 4,  # redshank
        "n01440764": 5,  # tench
        "n01443537": 6,  # goldfish
        "n01484850": 7,  # great white shark
        "n01491361": 8,  # tiger shark
        "n01494475": 9,  # hammerhead
    }

    datamodule = ImageNetDataModule(
        train_dir="./../imagenet10/train.X",
        val_dir="./../imagenet10/val.X",
        wnid_map=wnid_map,
        batch_size=BATCH_SIZE
    )

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(
    #     student.parameters(),
    #     lr=LR
    # )
    # train_dataset = ImageNet10(
    #                             "./../imagenet10/train.X",
    #                             wnid_map=wnid_map,
    #                             transform=preprocess_opencv
    #                         )
    # train_loader = DataLoader(train_dataset,
    #                             batch_size=BATCH_SIZE,
    #                             shuffle=True,
    #                             num_workers=4,
    #                             pin_memory=True
    #                         )
    # lrs, losses = find_lr(student, train_loader, criterion, optimizer, device)

    # best_lr = lrs[losses.index(min(losses))]
    # print(f"Optimal LR: {best_lr}")

    # student = load_full_model(TEACHER_PATH, device)
    best_lr = 0.012252798573828652
    model = PruningDistillationModule(
        teacher=teacher,
        student=student,
        epochs=EPOCHS,
        lr=best_lr,
        pruning_ratio=PRUNING_RATIO,
        len_train_loader = 204
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        #precision="16-mixed",  # AMP
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=datamodule)

    torch.save(student,"./../temp.pth")


if __name__ == "__main__":
    main()
