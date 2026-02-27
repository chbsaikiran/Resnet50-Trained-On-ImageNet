# ResNet on ImageNet-10

Training, fine-tuning, pruning, and deploying ResNet18 on a 10-class subset of ImageNet.

## Classes

| Index | Label             | WordNet ID |
|-------|-------------------|------------|
| 0     | sea lion          | n02077923  |
| 1     | albatross         | n02058221  |
| 2     | pelican           | n02051845  |
| 3     | oystercatcher     | n02037110  |
| 4     | redshank          | n02028035  |
| 5     | tench             | n01440764  |
| 6     | goldfish          | n01443537  |
| 7     | great white shark | n01484850  |
| 8     | tiger shark       | n01491361  |
| 9     | hammerhead        | n01494475  |

## Project Structure

```
.
├── model.py                        # ResNet architecture (BasicBlock, Bottleneck, ResNet)
├── train.py                        # Initial training on FashionMNIST (ResNet50)
├── train_resnet18.py               # Fine-tune pretrained ResNet18 on ImageNet-10
├── model_module.py                 # PyTorch Lightning module for ResNet18
├── hyperparameter_tuning_script.py # Grid search over batch size and learning rate
├── pruning_script.py               # Pruning + knowledge distillation (standalone)
├── calculate_accuracy.py           # Evaluate top-1 accuracy on validation set
├── video_inferencing.py            # Real-time video inference with timing stats
├── onnx_dump.py                    # Export trained model to ONNX format
├── mapping_indexes_wnids.py        # Generate ImageNet class index mappings
│
├── Training/                       # PyTorch Lightning training pipeline
│   ├── train.py                    # Training entrypoint
│   ├── model.py                    # ResNet architecture (copy)
│   ├── model_module.py             # LightningModule with OneCycleLR
│   ├── data_module.py              # LightningDataModule for ImageNet-10
│   └── transforms.py               # OpenCV-based preprocessing
│
├── Pruning/                        # PyTorch Lightning pruning pipeline
│   ├── train.py                    # Pruning + distillation entrypoint
│   ├── config.py                   # Hyperparameters (batch size, LR, pruning ratio)
│   ├── pruning.py                  # Magnitude and Taylor importance pruning
│   ├── lightning_module.py         # LightningModule for pruning + distillation
│   ├── datamodule.py               # LightningDataModule for ImageNet-10
│   ├── losses.py                   # Distillation loss (CE + KL divergence)
│   ├── model.py                    # ResNet architecture (copy)
│   └── find_lr.py                  # Learning rate finder
│
├── CPP_RELATED/                    # C++ deployment via ONNX Runtime
│   ├── video_infer_x86.cpp         # Video inference in C++
│   └── run.sh                      # Build and run script
│
├── imagenet10/                     # Dataset (10-class ImageNet subset)
│   ├── train.X/                    # Training images organized by wnid
│   └── val.X/                      # Validation images organized by wnid
│
├── Labels.json                     # Class index to wnid/label mapping
├── imagenet_classes.txt            # ImageNet-1000 class names
└── imagenet_index_to_wnid.json    # ImageNet-1000 index to wnid mapping
```

## Prerequisites

- Python 3.10+
- PyTorch
- torchvision
- pytorch-lightning
- torch-pruning
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:

```bash
pip install torch torchvision pytorch-lightning torch-pruning opencv-python numpy
```

For C++ inference, you also need:
- ONNX Runtime (C++ SDK)
- OpenCV 4.x (system install with `pkg-config` support)

## Dataset Setup

Place the ImageNet-10 subset under `imagenet10/` with this layout:

```
imagenet10/
├── train.X/
│   ├── n01440764/
│   ├── n01443537/
│   ├── n01484850/
│   ├── n01491361/
│   ├── n01494475/
│   ├── n02028035/
│   ├── n02037110/
│   ├── n02051845/
│   ├── n02058221/
│   └── n02077923/
└── val.X/
    └── (same folder structure)
```

Each subfolder is named by its WordNet ID and contains the JPEG images for that class.

## Preprocessing

All scripts use the same OpenCV-based preprocessing pipeline:

1. Resize the shorter side to 256 while preserving aspect ratio
2. Center crop to 224x224
3. Normalize to `[0, 1]`, then apply ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
4. Convert HWC to CHW tensor format

## Usage

### 1. Fine-Tune ResNet18 on ImageNet-10

This loads a pretrained ResNet18 (ImageNet-1000 weights), replaces the final FC layer with 10 outputs, and trains on ImageNet-10.

**Prerequisite:** Download or place pretrained ResNet18 weights as `resnet18.pth` in the project root.

```bash
python train_resnet18.py
```

Outputs:
- `resnet18_imagenet10_full.pth` -- full model (architecture + weights)
- `resnet18_imagenet10.pth` -- state dict only

### 2. Train with PyTorch Lightning

A cleaner, modular version of the training pipeline:

```bash
cd Training
python train.py
```

Configurable parameters in `Training/train.py`:
- `batch_size` (default: 64)
- `lr` (learning rate for OneCycleLR)
- `max_epochs` (default: 10)
- `pretrained_path` (path to pretrained weights)

Outputs are saved as `resnet18_imagenet10_full.pth` and `resnet18_imagenet10.pth` inside `Training/`.

### 3. Hyperparameter Tuning

Runs a grid search over batch sizes and learning rates using transfer learning (frozen backbone, only FC layer trained):

```bash
python hyperparameter_tuning_script.py
```

Default search space:
- Batch sizes: `[16, 32, 64]`
- Learning rates: `[1e-3, 1e-4, 1e-5]`

Edit the `BATCH_SIZE` and `LR` lists at the top of the script to customize. Results are written to `hyperparameter_results.csv`. Best checkpoints are saved per configuration in `checkpoints/` as `best_model_bs{batch_size}_lr{lr}.pth`.

### 4. Pruning with Knowledge Distillation

Structurally prune the trained model using Taylor importance and recover accuracy with knowledge distillation from the unpruned teacher.

**Standalone script:**

```bash
python pruning_script.py
```

**PyTorch Lightning version (recommended):**

```bash
cd Pruning
python train.py
```

Configure pruning in `Pruning/config.py`:

```python
BATCH_SIZE = 64
LR = 1e-6
EPOCHS = 10
NUM_CLASSES = 10
PRUNING_RATIO = 0.1        # fraction of channels to prune
CHECKPOINT_DIR = "./../checkpoints_pruning"
TEACHER_PATH = "./../Training/resnet18_imagenet10.pth"
```

The distillation loss is a weighted combination of:
- Cross-entropy against ground-truth labels
- KL divergence between student and teacher logits (with temperature scaling, T=4)

### 5. Evaluate Accuracy

Compute top-1 validation accuracy on ImageNet-10:

```bash
python calculate_accuracy.py
```

This loads `resnet18_imagenet10_full.pth`, runs inference on `imagenet10/val.X/`, and prints accuracy along with total/trainable parameter counts.

### 6. Video Inference (Python)

Run frame-by-frame inference on a video file with real-time display:

```bash
python video_inferencing.py
```

By default it reads `imagenet10_val_video.mp4` and runs on CPU. To use GPU, change the `device` variable at the top of the script to `"cuda"`.

Output:
- OpenCV window showing predictions and FPS overlay
- `video_predictions_golden.txt` -- per-frame predictions (`frame_id,label`)
- Timing summary printed at the end (avg/max/min inference time)

Reference timings:
| Device | Avg    | Min    | Max     |
|--------|--------|--------|---------|
| CPU    | 25.0ms | 15.4ms | 57.1ms  |
| CUDA   | 2.4ms  | 2.1ms  | 190.3ms |

Press `q` to quit early.

### 7. Export to ONNX

Convert the trained PyTorch model to ONNX format:

```bash
python onnx_dump.py
```

Exports `resnet18_imagenet10_full.pth` to `resnet18_imagenet10.onnx` (opset 13) with input name `"input"` and output name `"output"`, expecting input shape `(1, 3, 224, 224)`.

### 8. C++ Inference (ONNX Runtime)

Run inference on x86 using the exported ONNX model with ONNX Runtime and OpenCV:

```bash
cd CPP_RELATED
```

Edit `run.sh` to set the `ORT` variable to your ONNX Runtime installation path:

```bash
ORT=/path/to/onnxruntime-linux-x64-x.xx.x
```

Build and run:

```bash
./run.sh
```

This compiles `video_infer_x86.cpp` with `-O3` optimization and runs inference on `../imagenet10_val_video.mp4`. The C++ pipeline replicates the exact same preprocessing as Python (resize, center crop, normalize) for consistent results.

Requirements:
- ONNX Runtime C++ SDK (tested with v1.24.2)
- OpenCV 4.x with `pkg-config` support
- g++ with C++17 support

## Model Architecture

The project uses a custom ResNet implementation (matching torchvision's structure) defined in `model.py`:

- **ResNet18** -- `ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)` (~11.2M parameters)
- **ResNet50** -- `ResNet(Bottleneck, [3, 4, 6, 3], num_classes=N)` (~23.5M parameters for 1000 classes)

The forward pass: `Conv7x7 -> BN -> ReLU -> MaxPool -> Layer1-4 -> AdaptiveAvgPool -> FC`

ResNet18 with `BasicBlock` uses two 3x3 convolutions per block. ResNet50 with `Bottleneck` uses 1x1 -> 3x3 -> 1x1 per block with 4x channel expansion.
