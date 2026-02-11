import cv2
import torch
import torchvision.transforms as T
#from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import time
import numpy as np
import json


device = "cuda"   # change to "cuda" if you have GPU

model = torch.load("resnet18_imagenet10_full.pth", map_location=device)
model.to(device)
model.eval()

with open("Labels.json", "r") as f:
    imagenet_index_map = json.load(f)

# Create ordered label list (index 0 → 999)
labels = [imagenet_index_map[str(i)]["label"] for i in range(len(imagenet_index_map))]


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

cap = cv2.VideoCapture("imagenet10_val_video.mp4")  # or 0 for webcam

prev_time = 0
frame_id = 0
# Timing statistics
inference_times = []
max_inference_time = 0.0
total_inference_time = 0.0
total_frames = 0

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        # if frame_id % 3 != 0:
        #    continue

        # OpenCV → PIL
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Preprocess
        input_tensor = preprocess_opencv(pil_img).unsqueeze(0).to(device)

        start_time = time.perf_counter()
        # Inference
        outputs = model(input_tensor)

        end_time = time.perf_counter()
        inference_time = end_time - start_time  # seconds
        # Store statistics
        inference_times.append(inference_time)
        total_inference_time += inference_time
        max_inference_time = max(max_inference_time, inference_time)
        total_frames += 1

        pred_idx = outputs.argmax(dim=1).item()
        label = labels[pred_idx]

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Overlay text
        text = f"{label} | FPS: {fps:.1f}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Display
        cv2.imshow("ResNet18 Video Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
if total_frames > 0:
    avg_time = total_inference_time / total_frames
    print("\n====== Inference Timing Results ======")
    print(f"Total Frames Processed : {total_frames}")
    print(f"Average Inference Time : {avg_time * 1000:.3f} ms")
    print(f"Maximum Inference Time : {max_inference_time * 1000:.3f} ms")
    print(f"Min Inference Time     : {min(inference_times) * 1000:.3f} ms")
    print("======================================")

# cpu inference times:
# ====== Inference Timing Results ======
# Total Frames Processed : 3900
# Average Inference Time : 24.958 ms
# Maximum Inference Time : 57.123 ms
# Min Inference Time     : 15.376 ms
# ======================================

# cuda inference times:
# ====== Inference Timing Results ======
# Total Frames Processed : 3900
# Average Inference Time : 2.379 ms
# Maximum Inference Time : 190.333 ms
# Min Inference Time     : 2.059 ms
# ======================================

