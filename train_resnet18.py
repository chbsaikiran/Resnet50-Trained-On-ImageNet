# from torchvision.models import resnet18, ResNet18_Weights
# from torchvision.io import decode_image
# from model import BasicBlock,Bottleneck,ResNet
# import torch

# import json

# with open("imagenet_index_to_wnid.json", "r") as f:
#     imagenet_index_map = json.load(f)


# img = decode_image("test_image.jpeg")

# weights = ResNet18_Weights.IMAGENET1K_V1
# model = resnet18(weights=weights)
# model.eval()

# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()

# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)

# # Step 4: Use the model and print the predicted category
# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()

# # Lookup using JSON mapping
# class_info = imagenet_index_map[str(class_id)]
# wnid = class_info["wnid"]
# label = class_info["label"]
# description = class_info["description"]

# print(
#     f"with existing model {label} ({wnid}): "
#     f"{100 * score:.1f}%"
# )

# torch.save(model, "resnet18_full.pth")
# print("Saved PyTorch Model State to resnet18_full.pth")

# model_new = ResNet(
#     block=BasicBlock,
#     layers=[2, 2, 2, 2],
#     num_classes=1000
# )

# state_dict = torch.load("resnet18.pth", map_location="cpu")
# model_new.load_state_dict(state_dict)

# total_params = sum(p.numel() for p in model_new.parameters())
# print("Total parameters:", total_params)

# trainable_params = sum(p.numel() for p in model_new.parameters() if p.requires_grad)
# print("Trainable parameters:", trainable_params)

# model_new.eval()

# Step 2: Initialize the inference transforms
#preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
#batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
#prediction = model(batch).squeeze(0).softmax(0)
#class_id = prediction.argmax().item()
#score = prediction[class_id].item()
#category_name = weights.meta["categories"][class_id]
#print(f"with new model {category_name}: {100 * score:.1f}%")



from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import BasicBlock,Bottleneck,ResNet
import torch.optim as optim
import cv2
import torchvision.transforms as T
#from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import time
import numpy as np
import json

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

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

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


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total

epochs = 5

for epoch in range(epochs):
    loss = train_one_epoch(model, train_loader)
    acc = evaluate(model, val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={acc:.2f}%")

torch.save(model, "resnet18_imagenet10_full.pth")

#state_dict = torch.load("resnet18_imagenet10.pth", map_location="cpu")
#model.load_state_dict(state_dict)




