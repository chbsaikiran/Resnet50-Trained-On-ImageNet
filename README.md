# ResNet50 Trained on ImageNet

This repository contains code and resources for training, evaluating, and using ResNet models (including ResNet18 and ResNet50) on the ImageNet dataset and its subsets.

## File Descriptions

- **calculate_accuracy.py**: Script for evaluating model accuracy, using a trained model and a dataset.
- **imagenet_classes.txt**: Text file mapping class indices to human-readable class names for ImageNet.
- **imagenet_index_to_wnid.json**: JSON mapping from class indices to WordNet IDs (wnid), labels, and descriptions for ImageNet classes.
- **mapping_indexes_wnids.py**: Script to generate mappings between class indices, WordNet IDs, and labels using ImageNet resources.
- **model.py**: Implementation of the ResNet architecture, including `BasicBlock`, `Bottleneck`, and `ResNet` classes.
- **train_resnet18.py**: Script for training and evaluating ResNet18 on ImageNet, including model saving and inference example.
- **train.py**: Likely a general training script for ResNet models (not specific to ResNet18).