import json

# Load Labels.json (wnid -> description)
with open("./imagenet100/Labels.json", "r") as f:
    wnid_to_label = json.load(f)

# Load imagenet_classes.txt (index -> label)
index_to_label = {}
with open("imagenet_classes.txt", "r") as f:
    for line in f:
        idx, name = line.strip().split(":", 1)
        index_to_label[int(idx)] = name.strip()

def normalize(s):
    return s.lower().split(",")[0].strip()

label_to_wnid = {
    normalize(v): k
    for k, v in wnid_to_label.items()
}

index_to_wnid = {}

for idx, label in index_to_label.items():
    key = normalize(label)
    wnid = label_to_wnid.get(key)
    index_to_wnid[idx] = wnid

imagenet_index_map = {}

for idx in index_to_label:
    wnid = index_to_wnid[idx]
    imagenet_index_map[idx] = {
        "wnid": wnid,
        "label": index_to_label[idx],
        "description": wnid_to_label.get(wnid)
    }

with open("imagenet_index_to_wnid.json", "w") as f:
    json.dump(imagenet_index_map, f, indent=2)

