import torch
import json
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model import BasicBlock,Bottleneck,ResNet
import cv2
import numpy as np

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



with open("Labels.json", "r") as f:
    imagenet_index_map = json.load(f)

# index (int) -> wnid (str)
index_to_wnid = {
    int(k): v["wnid"]
    for k, v in imagenet_index_map.items()
}

# wnid (str) -> index (int)
wnid_to_index = {
    v["wnid"]: int(k)
    for k, v in imagenet_index_map.items()
}

val_dir = "./imagenet10/val.X"   # <-- change if needed

preprocess = transforms.Compose([
    transforms.Resize(
        size=256,
        interpolation=InterpolationMode.BILINEAR
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # scales to [0.0, 1.0]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_dataset = datasets.ImageFolder(
    root=val_dir,
    transform=preprocess_opencv
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ImageFolder gives: class_name -> local_index
# We remap local_index -> imagenet_index
local_to_imagenet_index = {}

for wnid, local_idx in val_dataset.class_to_idx.items():
    local_to_imagenet_index[local_idx] = wnid_to_index[wnid]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet(
    block=BasicBlock,
    layers=[2, 2, 2, 2],
    num_classes=10
)

state_dict = torch.load("resnet18_imagenet10.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, local_labels in val_loader:
        images = images.to(device)

        # Convert ImageFolder labels -> ImageNet labels
        imagenet_labels = torch.tensor(
            [local_to_imagenet_index[l.item()] for l in local_labels],
            device=device
        )

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == imagenet_labels).sum().item()
        total += imagenet_labels.size(0)

accuracy = 100.0 * correct / total
print(f"Top-1 Validation Accuracy: {accuracy:.2f}%")


