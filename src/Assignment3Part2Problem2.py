"""
Assignment 3 coding problem 2 

Problem 2
Implement an image segmentation model using 
    deep learning to segment objects from images at the pixel level. 
    Explore architectures U-Net, DeepLabV3, 
        and Mask R-CNN for segmentation purposes. 
    Use the Pascal VOC segmentation dataset for this problem.

In this problem, perform the following tasks:
    a. Load and inspect the dataset (images and masks). Resize, normalize, and apply any augmentations (e.g., flips, rotations). Split the dataset into training and test sets.
    b. Use U-Net, DeepLabV3, and Mask R-CNN for the segmentation.
    c. Train the models using appropriate loss functions, such as binary cross-entropy, dice loss, and IoU. Track the training progress (e.g., loss, dice score, etc.).
    d. Evaluate the performance using IoU, dice coefficient, pixel accuracy, and by visualizing outputs by overlaying masks on original images.
    e. Use a pre-trained model (e.g., ResNet or MobileNet). Add post-processing (e.g., CRFs or morphological ops). Compare the performances side by side.


"""
import os
# Must be set before torch is imported: torchvision::nms (used by YOLO validation/predict) has no MPS kernel,
# so unsupported ops fall back to the CPU while the rest of the model stays on the GPU.
#   this will be slower than running natively on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
# UserWarning: scatter_reduce_mps does not have a deterministic implementation
torch.use_deterministic_algorithms(True, warn_only=True)

import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from ultralytics import YOLO

import yaml
from ultralytics.utils import YAML  # only used to read the class names
from ultralytics.utils.checks import check_yaml

from model_eval import fit, evaluateAndPlot, evaluateLoader
DOC_FIGS_NAME ="assignment3part2problem2"


def setup():
    import torch
    import torchvision.transforms.v2 as transforms
    from torchvision.datasets import VOCSegmentation
    from torch.utils.data import DataLoader

    # 1. Define standard transforms for Segmentation (transforms.v2 handles images and masks together)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])

    # 2. Initialize the train dataset (Set download=True on first run)
    train_dataset = VOCSegmentation(
        root='./data', 
        year='2012', 
        image_set='train', 
        download=True,
        transforms=transform
    )

    # 3. Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # 4. Fetch a sample batch
    for images, targets in train_loader:
        print("Images batch shape:", images.shape)   # [batch_size, 3, 256, 256]
        print("Targets batch shape:", targets.shape) # [batch_size, 256, 256] or [batch_size, 1, 256, 256]
        break


def problem2():
    """
    Implement an image segmentation model using 
    deep learning to segment objects from images at the pixel level. 
    Explore architectures U-Net, DeepLabV3, 
        and Mask R-CNN for segmentation purposes. 
    
    Use the Pascal VOC segmentation dataset for this problem.

    Explore architectures U-Net, DeepLabV3, and Mask R-CNN for segmentation purposes
    """
    
    setup()
    
def main():
    problem2()

if __name__ == "__main__":
    main()
