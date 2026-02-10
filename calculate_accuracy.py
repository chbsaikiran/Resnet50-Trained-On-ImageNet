import torch
import json
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

with open("imagenet_index_to_wnid.json", "r") as f:
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
    transform=preprocess
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
model = torch.load("resnet18_full.pth", map_location=device)
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


