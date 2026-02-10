from torchvision.models import resnet18, ResNet18_Weights
from torchvision.io import decode_image
from model import BasicBlock,Bottleneck,ResNet
import torch

import json

with open("imagenet_index_to_wnid.json", "r") as f:
    imagenet_index_map = json.load(f)


img = decode_image("test_image.jpeg")

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()

# Lookup using JSON mapping
class_info = imagenet_index_map[str(class_id)]
wnid = class_info["wnid"]
label = class_info["label"]
description = class_info["description"]

print(
    f"with existing model {label} ({wnid}): "
    f"{100 * score:.1f}%"
)

torch.save(model, "resnet18_full.pth")
print("Saved PyTorch Model State to resnet18_full.pth")

model_new = ResNet(
    block=BasicBlock,
    layers=[2, 2, 2, 2],
    num_classes=1000
)

state_dict = torch.load("resnet18.pth", map_location="cpu")
model_new.load_state_dict(state_dict)

total_params = sum(p.numel() for p in model_new.parameters())
print("Total parameters:", total_params)

trainable_params = sum(p.numel() for p in model_new.parameters() if p.requires_grad)
print("Trainable parameters:", trainable_params)

model_new.eval()

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
