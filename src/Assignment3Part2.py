"""
Assignment 3 coding problems in part 2 of assignment

Problem 1
Implement an object detection model using a pre-trained deep learning architecture (e.g., YOLO) to detect and classify objects in images. 
    For this problem, use the COCO128 (or a small subset of COCO) dataset.

In this problem, perform the following tasks:
    a. Load and explore the dataset (classes, annotation formats, etc.), visualize sample images with bounding boxes, and split it into training and validation/test sets.
    b. Apply YOLOv5  and SSD object detection models.
    c. Train the models on the training set. Apply relevant augmentations (flipping, scaling, colour jittering). Track training progress, and monitor loss and mean Average Precision (mAP).
    d. Evaluate the models using a test set. Report the evaluation results in terms of the metrics mAP@0.5, precision, recall, and inference speed. Visualize detection outputs on at least 10 test images.


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
import random
from collections import Counter
from pathlib import Path




def problem1():
    """
    a. Load and explore the dataset (classes, annotation formats, etc.), 
        visualize sample images with bounding boxes, 
        and split it into training and validation/test sets.

    b. Apply YOLOv5  and SSD object detection models.
    c. Train the models on the training set. Apply relevant augmentations (flipping, scaling, colour jittering). Track training progress, and monitor loss and mean Average Precision (mAP).
    d. Evaluate the models using a test set. Report the evaluation results in terms of the metrics mAP@0.5, precision, recall, and inference speed. Visualize detection outputs on at least 10 test images.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    from ultralytics.utils import YAML  # only used to read the class names
    from ultralytics.utils.checks import check_yaml

    from model_eval import fit, evaluateAndPlot, evaluateLoader
    ROOT = Path(__file__).resolve().parent / "data" / "coco128"
    IMG_DIR = ROOT / "images" / "train2017"
    LABEL_DIR = ROOT / "labels" / "train2017"
    
    # NAMES is from yaml of form: {0: 'person', 1: 'bicycle', ...}
    NAMES = YAML.load(check_yaml("coco128.yaml"))["names"]

    explore()
    apply_yolo()
    apply_ssd()




    
def main():
    pass

if __name__ == "__main__":
    main()
