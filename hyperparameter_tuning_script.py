import os
import time
import copy
import random
import numpy as np
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau
)
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
import cv2
import torchvision.transforms as T
from PIL import Image
import time
import numpy as np
import json
import csv
from model import BasicBlock,Bottleneck,ResNet

# ImageNet normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_opencv(
    img,
    resize_size=256,
    crop_size=224
):
    # img is PIL Image → convert to numpy (RGB)
    img = np.array(img)

    # --------------------------------------------------
    # 2. Resize: shorter side = 256 (keep aspect ratio)
    # --------------------------------------------------
    h, w, _ = img.shape

    if h < w:
        new_h = resize_size
        new_w = int(w * resize_size / h)
    else:
        new_w = resize_size
        new_h = int(h * resize_size / w)

    img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------
    # 3. Center crop 224 × 224
    # --------------------------------------------------
    start_x = (new_w - crop_size) // 2
    start_y = (new_h - crop_size) // 2

    img = img[
        start_y:start_y + crop_size,
        start_x:start_x + crop_size
    ]

    # --------------------------------------------------
    # 4. Normalize
    # --------------------------------------------------
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # --------------------------------------------------
    # 5. HWC → CHW
    # --------------------------------------------------
    img = np.transpose(img, (2, 0, 1))

    return torch.from_numpy(img)

train_transform = preprocess_opencv

val_transform = preprocess_opencv

BATCH_SIZE = [16,32,64]
LR = [1e-3, 1e-4, 1e-5]

WNID_TO_LABEL = {
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

class ImageNet10(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        self.samples = [
            (path, WNID_TO_LABEL[self.classes[label]])
            for path, label in self.samples
            if self.classes[label] in WNID_TO_LABEL
        ]

train_dataset = ImageNet10(
    root="./imagenet10/train.X",
    transform=train_transform
)

val_dataset = ImageNet10(
    root="./imagenet10/val.X",
    transform=val_transform
)

# ------------------------------
# Utility Functions
# ------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_full_model(pth_path: str, device: torch.device):
    model = torch.load(pth_path, map_location=device)
    model.to(device)
    return model

# ------------------------------
# Train Function
# ------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,

    # Core hyperparameters
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adam",   # adam, sgd, adamw
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,

    # Scheduler
    scheduler_name: Optional[str] = None,  # step, multistep, cosine, exp, plateau
    step_size: int = 10,
    gamma: float = 0.1,
    milestones: list = None,
    T_max: int = 50,
    patience_lr: int = 5,

    # Regularization
    l1_lambda: float = 0.0,
    label_smoothing: float = 0.0,

    # Gradient
    gradient_clip: Optional[float] = None,
    accumulation_steps: int = 1,

    # Mixed precision
    use_amp: bool = False,

    # Early stopping
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 0.0,

    # Checkpointing
    save_best_only: bool = True,
    checkpoint_dir: str = "./checkpoints",
    resume: bool = False,

    # Misc
    log_interval: int = 50,
    seed: int = 42,
    class_weights: Optional[torch.Tensor] = None,
    best_model_name: str = "best_model.pth"
):

    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )

    # Optimizer
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError("Unsupported optimizer")

    # Scheduler
    scheduler = None
    if scheduler_name:
        if scheduler_name == "step":
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "multistep":
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_name == "exp":
            scheduler = ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, patience=patience_lr)

    scaler = GradScaler() if use_amp else None

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # L1 regularization
                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss += l1_lambda * l1_norm

                loss = loss / accumulation_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:

                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip
                    )

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % log_interval == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        val_loss, val_acc = None, None
        if val_loader:
            val_loss, val_acc = test(model, val_loader, device, criterion)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        if val_loader:
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.2f} sec\n")

        # Scheduler step
        if scheduler:
            if scheduler_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping
        if val_loader:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                early_stop_counter = 0
                if save_best_only:
                    torch.save(model, os.path.join(checkpoint_dir, best_model_name))
            else:
                early_stop_counter += 1
                if early_stopping and early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

    print("Training complete.")
    return val_loss, val_acc


# ------------------------------
# Test Function
# ------------------------------

def test(model: nn.Module,
         dataloader: DataLoader,
         device: torch.device,
         criterion: Optional[nn.Module] = None):

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader) if criterion else None

    return avg_loss, accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_file = "hyperparameter_results.csv"

# Create file and write header
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["batch_size", "learning_rate", "val_acc", "val_loss"])

for batch_size in BATCH_SIZE:
    for lr in LR:
        print(f"Training with batch size={batch_size} and learning rate={lr}")
        best_model_name = f"best_model_bs{batch_size}_lr{lr}.pth"
        model = ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=1000
        )

        state_dict = torch.load("resnet18.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.to(device)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        val_loss, val_acc = train(
            model=model,
            batch_size=batch_size,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=5,
            learning_rate=lr,
            optimizer_name="adamw",
            scheduler_name="cosine",
            early_stopping=True,
            patience=10,
            use_amp=True,
            gradient_clip=1.0,
            best_model_name=best_model_name
        )
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([batch_size, lr, val_acc, val_loss])
