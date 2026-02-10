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

train_transform = transforms.Compose([
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

val_transform = transforms.Compose([
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

torch.save(model.state_dict(), "resnet18_imagenet10.pth")

#state_dict = torch.load("resnet18_imagenet10.pth", map_location="cpu")
#model.load_state_dict(state_dict)




