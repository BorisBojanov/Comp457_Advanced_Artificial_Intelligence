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

Setup:
    conda install -c conda-forge ultralytics -y
    conda activate Comp457

    python src/Assignment3Part2.py

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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "coco128" / "coco128_split.yaml"
DATA_DIR = ROOT / "data" / "coco128"
IMG_DIR = DATA_DIR / "images" / "train2017"
LABEL_DIR = DATA_DIR / "labels" / "train2017"

def get_device(model: YOLO) -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

    # model.device returns a torch.device object (device(type='mps', index=0)) or type='cuda' or type='cpu'
    # Mac: "mps" | NVIDIA: 0 | fallback: "cpu"
    # return model.device.type

def read_labels(img_path: Path):
    """YOLO txt format: one row per object -> class cx cy w h (all normalised 0-1)."""
    lbl = LABEL_DIR / (img_path.stem + ".txt")
    if not lbl.exists():  # some COCO128 images have no objects
        return []
    return [(int(c), *map(float, xywh)) for c, *xywh in
            (line.split() for line in lbl.read_text().splitlines() if line.strip())]    

def loadExploreDataset(IMG_DIR: Path):
    """
    execute: 
        while 'loadExploreDataset' is called in the main loop
        within activated conda environment
        python3 src/Assignment3Part2.py
        
    expected output:
        Creating new Ultralytics Settings v0.0.8 file ✅ 
        View Ultralytics Settings with 'yolo settings' or at '/Users/boris/Library/Application Support/Ultralytics/settings.json'
        Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/usage/settings.
        
        128 images, 128 label files
        Sum: 929 boxes over 71 of 80 classes
        person         254
        car            46
        cup            36
        chair          35
        book           29
        bowl           28
        carrot         24
        spoon          22
        teddy bear     21
        handbag        19
        train: 102 images
        val: 12 images
        test: 14 images
        wrote /Users/boris/Documents/CodingStuff/Python_Stuff/Comp457_Advanced_Artificial_Intelligence/src/data/coco128/coco128_split.yaml
    """
    # # NAMES is from yaml of form: {0: 'person', 1: 'bicycle', ...}
    # NAMES = YAML.load(check_yaml("coco128.yaml"))["names"]

    # # Opt1: Using Ultralytics YOLO API
    # from ultralytics import YOLO
    # # Load a pretrained model (it automatically contains the COCO names mapping)
    # model = YOLO("yolov8n.pt")
    # # Access the names dictionary directly
    # NAMES = model.names

    # Opt2: Parsing the YAML Manually with PyYAML
    #  downloads coco128.yaml file downloaded locally
    # check_yaml downloads the file and returns its absolute path string
    yaml_path = check_yaml("coco128.yaml")
    # Open and safely load the YAML file
    #   use the resolved path
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Extract the names dictionary
    # NAMES is from yaml of form: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', ...}
    NAMES = data["names"]
    # print(NAMES)

    if not IMG_DIR.exists():   # first run: fetch the 7 MB zip (128 images of COCO train2017 + YOLO labels)
            import urllib.request, zipfile
            ROOT.parent.mkdir(parents=True, exist_ok=True)
            zp = ROOT.parent / "coco128.zip"
            urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", zp)
            zipfile.ZipFile(zp).extractall(ROOT.parent)

    imagePaths = sorted(IMG_DIR.glob("*.jpg"))
    print(f"{len(imagePaths)} images, {sum(1 for _ in LABEL_DIR.glob('*.txt'))} label files")
    
    # counts = Counter(c for p in images for c, *_ in read_labels(p))
    counts = Counter()
    for image in imagePaths:
         for c, *_ in read_labels(image):
            counts[c] += 1
            
    print(f"Sum: {sum(counts.values())} boxes over {len(counts)} of {len(NAMES)} classes")
    for c, n in counts.most_common(10):
        print(f"  {NAMES[c]:<15}{n}")

    # --- class histogram -------------------------------------------------------
    top = counts.most_common(20)
    
    plt.figure(figsize=(10, 4))
    plt.bar([NAMES[c] for c, _ in top], [n for _, n in top])
    plt.xticks(rotation=60, ha="right"); plt.ylabel("boxes"); plt.title("COCO128 – 20 most frequent classes")
    plt.tight_layout(); plt.savefig("p1_class_hist.png", dpi=150); plt.close()
    
    # --- sample images with boxes ---------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, p in zip(axes.flat, random.Random(0).sample(imagePaths, 6)):
        im = Image.open(p); W, H = im.size
        ax.imshow(im); ax.axis("off"); ax.set_title(p.name)
        for c, cx, cy, w, h in read_labels(p):
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H          # centre form -> corner form, pixels
            ax.add_patch(patches.Rectangle((x1, y1), w * W, h * H, fill=False, lw=2, ec="lime"))
            ax.text(x1, y1, NAMES[c], color="black", backgroundcolor="lime", fontsize=7)
    plt.tight_layout(); plt.savefig("p1_samples.png", dpi=150); plt.close()

    # --- 80 / 10 / 10 split ------------------------------------------------------
    random.Random('the meaning of life 42').shuffle(imagePaths)
    n_tr = int(0.80 * len(imagePaths))
    n_va = int(0.10 * len(imagePaths))
    n_trva = n_tr + n_va
    splits = {
        "train": imagePaths[:n_tr], 
        "val": imagePaths[n_tr:n_trva], 
        "test": imagePaths[n_trva:]
    }
    for name, files in splits.items():
        (ROOT / f"{name}.txt").write_text("\n".join(str(f) for f in files) + "\n")
        print(f"{name}: {len(files)} images")


    # Ultralytics accepts .txt files listing image paths; labels are found by swapping images/ -> labels/
    yaml_text = f"path: {ROOT}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n" + \
        "".join(f"  {k}: {v}\n" for k, v in NAMES.items())
    
    (ROOT / "coco128_split.yaml").write_text(yaml_text)
    print("wrote", ROOT / "coco128_split.yaml")

def yolo():
    """
        Step 2-4 (YOLO side): fine-tune YOLOv5 on the COCO128 split, evaluate, visualise.
    """
    # yolov5su.pt = YOLOv5-small, COCO-pretrained, re-released by Ultralytics with an anchor-free head.
    # (The original anchor-based repo is github.com/ultralytics/yolov5 — `python train.py --weights yolov5s.pt`.)
    yolo_model = YOLO("yolov5su.pt")
    print(get_device(yolo_model))
    DEVICE = "mps"        # Mac: "mps" | NVIDIA: 0 | fallback: "cpu"
    EPOCHS = 30

    # --- train: augmentations are hyper-parameters, applied on the fly to every batch ---
    yolo_model.train(
        data=str(DATA), epochs=EPOCHS, imgsz=640, batch=16, device=DEVICE,
        fliplr=0.5,                           # horizontal flip probability
        scale=0.5,                            # random rescale +/- 50 %
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,    # colour jitter (hue / saturation / value)
        mosaic=1.0,                           # 4-image mosaic (YOLO's signature augmentation)
        project=str(ROOT / "runs"), name="yolov5s_coco128", exist_ok=True, plots=True,
    )

    """
    Training curves (box/cls/dfl loss, precision, recall, mAP50, mAP50-95 per epoch) are saved to
      runs/yolov5s_coco128/results.csv and results.png 
      — results.png is in the report.
    """

    # --- evaluate on the held-out TEST split ---
    best = YOLO(ROOT / "runs" / "yolov5s_coco128" / "weights" / "best.pt")
    m = best.val(data=str(DATA), split="test", imgsz=640, device=DEVICE,
                project=str(ROOT / "runs"), name="yolov5s_test", exist_ok=True)
    
    # print(f"YOLOv5s test | mAP@0.5 {m.box.map50:.3f}  mAP@0.5:0.95 {m.box.map:.3f}  "
    #     f"P {m.box.mp:.3f}  R {m.box.mr:.3f}  "
    #     f"speed/img: pre {m.speed['preprocess']:.1f} ms  infer {m.speed['inference']:.1f} ms  "
    #     f"post {m.speed['postprocess']:.1f} ms")
    fancyString = f"""
            YOLOv5s test | mAP@0.5 {m.box.map50:.3f}  mAP@0.5:0.95 {m.box.map:.3f}  
            P {m.box.mp:.3f}  R {m.box.mr:.3f}  
            speed/img: pre {m.speed['preprocess']:.1f} ms  infer {m.speed['inference']:.1f} ms  
            post {m.speed['postprocess']:.1f} ms
    """
    print(fancyString)
    # --- visualise detections on 10+ test images ---
    test_imgs = (ROOT / "data" / "coco128" / "test.txt").read_text().split()[:12]
    best.predict(test_imgs, 
                conf=0.25, 
                device=DEVICE, 
                save=True,
                project=str(ROOT / "runs"), 
                name="yolov5s_pred", 
                exist_ok=True
    )
    """Output
    ...Epochs...

30 epochs completed in 0.046 hours.
Optimizer stripped from /Users/boris/Documents/CodingStuff/Python_Stuff/Comp457_Advanced_Artificial_Intelligence/src/runs/yolov5s_coco128/weights/last.pt, 18.6MB
Optimizer stripped from /Users/boris/Documents/CodingStuff/Python_Stuff/Comp457_Advanced_Artificial_Intelligence/src/runs/yolov5s_coco128/weights/best.pt, 18.6MB

Validating /Users/boris/Documents/CodingStuff/Python_Stuff/Comp457_Advanced_Artificial_Intelligence/src/runs/yolov5s_coco128/weights/best.pt...
Ultralytics 8.4.183 🚀 Python-3.12.15 torch-2.13.0 MPS (Apple M3 Max)
YOLOv5s summary (fused): 84 layers, 9,142,496 parameters, 0 gradients, 24.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 2.5it/s 0.4s
                   all         12         61      0.916      0.601      0.791      0.624
                person          7         26          1      0.526      0.754      0.562
               bicycle          1          1      0.868          1      0.995      0.895
                   car          1          1      0.849          1      0.995      0.796
            motorcycle          2          2          1      0.682      0.995      0.696
                   dog          2          2      0.921          1      0.995      0.846
               handbag          2          4          1          0       0.48      0.271
                   tie          1          1      0.914          1      0.995      0.697
               frisbee          1          1      0.888          1      0.995      0.796
          baseball bat          1          1          1          0          0          0
        baseball glove          2          4          1          0      0.245      0.221
            wine glass          1          1      0.897          1      0.995      0.895
                   cup          1          3          1      0.611      0.913      0.498
                  fork          1          1          1          0      0.332      0.232
                  bowl          1          1      0.478          1      0.995      0.895
                carrot          1          2      0.955          1      0.995      0.796
                 chair          1          2          1          0      0.995      0.945
                    tv          1          1      0.939          1      0.995      0.895
            cell phone          1          3          1          0      0.467      0.296
            teddy bear          1          4      0.698      0.594      0.895      0.629
Speed: 0.2ms preprocess, 23.9ms inference, 0.0ms loss, 1.9ms postprocess per image
Results saved to /Users/boris/Documents/CodingStuff/Python_Stuff/Comp457_Advanced_Artificial_Intelligence/src/runs/yolov5s_coco128
Ultralytics 8.4.183 🚀 Python-3.12.15 torch-2.13.0 MPS (Apple M3 Max)
YOLOv5s summary (fused): 84 layers, 9,142,496 parameters, 0 gradients, 24.0 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.1 ms, read: 158.2±20.8 MB/s, size: 34.6 KB)
val: Scanning /Users/boris/Documents/CodingStuff/Python_Stuff/Comp457_Advanced_Artificial_Intelligence/src/d…: 100% ━━━━━━━━━━━━ 14/14 5.3Kit/s 0.0s
val: New cache created: /Users/boris/Documents/CodingStuff/Python_Stuff/Comp457_Advanced_Artificial_Intelligence/src/data/coco128/labels/train2017.cache
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 4.6it/s 0.2s
                   all         14        102        0.5      0.739      0.712      0.517
                person          7         38      0.526      0.947      0.833      0.568
                   car          2          2      0.143        0.5      0.501       0.45
              airplane          1          1      0.276          1      0.995      0.497
                 train          1          1      0.623          1      0.995      0.597
                 truck          1          5      0.616      0.648      0.728       0.41
                  boat          1          5      0.295        0.4      0.382      0.244
         traffic light          1          9      0.737      0.444      0.401      0.118
                 bench          2          3      0.545          1      0.913      0.583
                   cat          2          2       0.73          1      0.995      0.945
                   dog          2          2          1      0.925      0.995       0.72
              backpack          3          5      0.656        0.6      0.675      0.495
               handbag          2          7      0.568      0.143      0.164     0.0926
                   tie          2          3      0.426      0.333      0.333      0.124
               frisbee          1          1      0.299          1      0.995      0.995
                  skis          1          1      0.566          1      0.995      0.647
                bottle          1          1      0.496          1      0.995      0.995
                  bowl          1          1          0          0      0.142      0.114
                banana          1          1      0.112          1      0.142     0.0853
              sandwich          1          1      0.907          1      0.995      0.647
                 couch          1          1      0.245          1      0.497      0.308
          potted plant          1          3      0.691      0.763      0.913      0.681
                   bed          1          1       0.29          1      0.995      0.332
          dining table          1          1          1          0      0.199     0.0597
                remote          1          1      0.478          1      0.995      0.995
            cell phone          1          1      0.697          1      0.995      0.995
                  sink          1          1          0          0      0.166      0.116
                  book          1          1      0.515          1      0.995      0.895
            teddy bear          1          3      0.578          1      0.995      0.763
Speed: 0.2ms preprocess, 6.7ms inference, 0.0ms loss, 2.7ms postprocess per image
Results saved to /Users/boris/Documents/CodingStuff/Python_Stuff/Comp457_Advanced_Artificial_Intelligence/src/runs/yolov5s_test

            YOLOv5s test | mAP@0.5 0.712  mAP@0.5:0.95 0.517  
            P 0.500  R 0.739  
            speed/img: pre 0.2 ms  infer 6.7 ms  
            post 2.7 ms
    

0: 640x640 2 persons, 1 cat, 3 couchs, 1 remote, 2.8ms
1: 640x640 1 train, 2 potted plants, 2.8ms
2: 640x640 1 car, 1 cat, 2.8ms
3: 640x640 12 persons, 3 airplanes, 1 handbag, 2.8ms
4: 640x640 1 person, 2 frisbees, 2.8ms
5: 640x640 3 persons, 2 backpacks, 1 skis, 2.8ms
6: 640x640 24 persons, 1 car, 3 trucks, 3 traffic lights, 1 backpack, 2.8ms
7: 640x640 3 beds, 2.8ms
8: 640x640 2 persons, 2 boats, 2.8ms
9: 640x640 1 bottle, 2 cups, 1 bowl, 7 bananas, 2.8ms
10: 640x640 1 bird, 1 bowl, 4 teddy bears, 2.8ms
11: 640x640 10 persons, 1 boat, 2 benchs, 1 bird, 2.8ms
Speed: 6.3ms preprocess, 2.8ms inference, 6.6ms postprocess per image at shape (1, 3, 640, 640)
Results saved to /Users/boris/Documents/CodingStuff/Python_Stuff/Comp457_Advanced_Artificial_Intelligence/src/runs/yolov5s_pred
    """

def ssd():
    """
    Step 2-4 (SSD side): fine-tune torchvision's SSD300-VGG16 on the COCO128 split.
    
    Also evaluates the trained YOLOv5 with the *same* evaluation code, so the two models are compared
    on one yardstick (mAP@0.5, precision, recall, ms/image).
    """
    import json
    import time
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchmetrics.detection import MeanAveragePrecision
    from torchvision import tv_tensors
    from torchvision.io import decode_image
    from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
    from torchvision.ops import box_iou
    from torchvision.transforms import v2
    from torchvision.utils import draw_bounding_boxes, save_image
    from ultralytics.data.converter import coco80_to_coco91_class

    DATA = DATA_DIR
    OUT = ROOT / "runs" / "ssd300"; OUT.mkdir(parents=True, exist_ok=True)
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS, BATCH, LR = 30, 8, 1e-3
    PRETRAINED = True
    CONF = 0.25          # score threshold used for precision / recall / pictures

    # YOLO labels use 0..79; torchvision's COCO head uses the original COCO ids 1..90 (0 = background).
    # Mapping keeps the pretrained classification head usable — no new head needed.
    YOLO2COCO = coco80_to_coco91_class()          # list: index 0..79 -> COCO id
    CATEGORIES = SSD300_VGG16_Weights.COCO_V1.meta["categories"]   # COCO id -> name


    # transforms.v2 moves the boxes together with the pixels (flip, crop, zoom) — that is the point of v2.
    train_tf = v2.Compose([
        v2.RandomPhotometricDistort(p=0.5),          # colour jitter: brightness/contrast/saturation/hue
        v2.RandomZoomOut(fill={tv_tensors.Image: (124, 116, 104), "others": 0}, side_range=(1.0, 2.0), p=0.5),
        v2.RandomIoUCrop(),                          # scale up / crop (the SSD paper's augmentation)
        v2.RandomHorizontalFlip(p=0.5),
        v2.SanitizeBoundingBoxes(),                  # drop boxes that became degenerate after cropping
        v2.ToDtype(torch.float32, scale=True),       # uint8 0-255 -> float 0-1
    ])
    eval_tf = v2.ToDtype(torch.float32, scale=True)  # SSD resizes to 300x300 + normalises internally

    class COCO128Det(Dataset):
        """Returns (image tensor uint8 CxHxW, target dict) in the format torchvision detectors expect."""

        def __init__(self, split, transforms=None, skip_empty=False):
            self.files = [Path(p) for p in (DATA / f"{split}.txt").read_text().split()]
            if skip_empty:   # RandomIoUCrop needs at least one box
                self.files = [p for p in self.files if self._read(p)[0].numel()]
            self.transforms = transforms

        @staticmethod
        def _label_path(img):
            return Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")

        def _read(self, img):
            lp = self._label_path(img)
            rows = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
            cls = torch.tensor([YOLO2COCO[int(r[0])] for r in rows], dtype=torch.int64)
            cxcywh = torch.tensor([[float(v) for v in r[1:]] for r in rows]).reshape(-1, 4)
            return cls, cxcywh

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            from enum import Enum
            class modeType(Enum):
                UNCHANGED = 0
                GRAY = 1
                GRAY_ALPHA = 2
                RGB = 3
                RGB_ALPHA = 4
                RGBA = RGB_ALPHA  # Alias for convenience
            
            img = decode_image(str(self.files[i]), mode="RGB")
            H, W = img.shape[-2:]
            cls, b = self._read(self.files[i])
            b = b * torch.tensor([W, H, W, H])
            xyxy = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)  # centre -> corners
            target = {"boxes": tv_tensors.BoundingBoxes(xyxy, format="XYXY", canvas_size=(H, W)),
                    "labels": cls}
            if self.transforms:
                img, target = self.transforms(img, target)
            return img, target

    def collate(batch):
        return tuple(zip(*batch))   # detection targets have different numbers of boxes -> keep as lists

    def build_model():
        if PRETRAINED:
            return ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
        return ssd300_vgg16(weights=None, weights_backbone=None, num_classes=91)

    @torch.no_grad()
    def evaluate(predict_fn, ds, conf=CONF):
        """predict_fn(list[img]) -> list[{'boxes','scores','labels'}] (boxes xyxy pixels, COCO ids)."""
        metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
        metric.warn_on_many_detections = False        # SSD returns up to 200 boxes; top-100 are scored
        tp = fp = n_gt = 0
        times = []
        for img, tgt in ds:
            t0 = time.perf_counter()
            pred = predict_fn([img])[0]
            if DEVICE.type == "mps": torch.mps.synchronize()
            elif DEVICE.type == "cuda": torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            pred = {k: v.cpu() for k, v in pred.items()}
            gt = {"boxes": tgt["boxes"].as_subclass(torch.Tensor), "labels": tgt["labels"]}
            metric.update([pred], [gt])
            # precision / recall at a fixed confidence: greedy match, same class, IoU >= 0.5
            keep = pred["scores"] >= conf
            pb, pl = pred["boxes"][keep], pred["labels"][keep]
            n_gt += len(gt["labels"])
            used = torch.zeros(len(gt["labels"]), dtype=torch.bool)
            if len(pb) and len(gt["boxes"]):
                iou = box_iou(pb, gt["boxes"])
                iou[pl[:, None] != gt["labels"][None, :]] = 0       # class must match
                for j in range(len(pb)):                             # preds are already score-sorted
                    iou[j, used] = 0
                    k = int(iou[j].argmax())
                    if iou[j, k] >= 0.5:
                        used[k] = True; tp += 1
                    else:
                        fp += 1
            else:
                fp += len(pb)
        r = metric.compute()
        times = times[3:] if len(times) > 5 else times               # drop warm-up iterations
        return {"mAP@0.5": r["map"].item(),
                "precision": tp / max(tp + fp, 1), "recall": tp / max(n_gt, 1),
                "ms_per_img": 1000 * sum(times) / len(times), "FPS": len(times) / sum(times)}

    def ssd_main():
        train_ds = COCO128Det("train", train_tf, skip_empty=True)
        val_ds, test_ds = COCO128Det("val", eval_tf), COCO128Det("test", eval_tf)
        loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)

        model = build_model().to(DEVICE)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        def ssd_predict(imgs):
            model.eval()
            return model([i.to(DEVICE) for i in imgs])

        history = []
        for epoch in range(EPOCHS):
            model.train()
            tot = {"bbox_regression": 0.0, "classification": 0.0}
            for imgs, tgts in loader:
                imgs = [i.to(DEVICE) for i in imgs]
                tgts = [{"boxes": t["boxes"].to(DEVICE), "labels": t["labels"].to(DEVICE)} for t in tgts]
                losses = model(imgs, tgts)                  # train mode -> dict of losses
                loss = sum(losses.values())
                opt.zero_grad(); loss.backward(); opt.step()
                for k in tot: tot[k] += losses[k].item() / len(loader)
            sched.step()
            val = evaluate(ssd_predict, val_ds)
            history.append({"epoch": epoch + 1, **tot, "val_mAP50": val["mAP@0.5"]})
            print(f"epoch {epoch + 1:>3}/{EPOCHS}  box {tot['bbox_regression']:.3f}  "
                f"cls {tot['classification']:.3f}  val mAP@0.5 {val['mAP@0.5']:.3f}")

        torch.save(model.state_dict(), OUT / "ssd300_coco128.pth")
        json.dump(history, open(OUT / "history.json", "w"), indent=1)

        # --- training curves ---
        ep = [h["epoch"] for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ep, [h["bbox_regression"] for h in history], label="box regression (Smooth L1)")
        ax[0].plot(ep, [h["classification"] for h in history], label="classification (CE, hard-neg mined)")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("SSD300 training loss")
        ax[1].plot(ep, [h["val_mAP50"] for h in history], marker="o")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mAP@0.5"); ax[1].set_title("SSD300 validation mAP@0.5")
        plt.tight_layout(); plt.savefig(OUT / "ssd_curves.png", dpi=150); plt.close()

        # --- test-set evaluation: SSD ---
        results = {"SSD300-VGG16": evaluate(ssd_predict, test_ds)}

        # --- test-set evaluation: YOLOv5 through the SAME evaluate() ---
        yolo_w = ROOT / "runs" / "yolov5s_coco128" / "weights" / "best.pt"
        if yolo_w.exists():
            from ultralytics import YOLO
            yolo = YOLO(yolo_w)
            to_coco = torch.tensor(YOLO2COCO)

            def yolo_predict(imgs):
                # Ultralytics wants HWC uint8 numpy (BGR); it letterboxes + runs NMS internally
                arrs = [(i.permute(1, 2, 0) * 255).byte().numpy()[..., ::-1].copy() for i in imgs]
                out = []
                for r in yolo.predict(arrs, conf=0.001, imgsz=640, device=DEVICE.type if DEVICE.type != "cuda" else 0, verbose=False):
                    b = r.boxes
                    out.append({"boxes": b.xyxy, "scores": b.conf, "labels": to_coco[b.cls.long().cpu()]})
                return out
            results["YOLOv5s"] = evaluate(yolo_predict, test_ds)

        print(f"\n{'model':<14}{'mAP@0.5':>9}{'P':>8}{'R':>8}{'ms/img':>9}{'FPS':>8}")
        for name, r in results.items():
            print(f"{name:<14}{r['mAP@0.5']:>9.3f}{r['precision']:>8.3f}{r['recall']:>8.3f}"
                f"{r['ms_per_img']:>9.1f}{r['FPS']:>8.1f}")
        json.dump(results, open(OUT / "test_results.json", "w"), indent=1)

        # --- visualise SSD detections on 12 test images ---
        model.eval()
        for i in range(min(12, len(test_ds))):
            img, tgt = test_ds[i]
            with torch.no_grad():
                p = model([img.to(DEVICE)])[0]
            k = p["scores"].cpu() >= CONF
            names = [f"{CATEGORIES[l]} {s:.2f}" for l, s in zip(p["labels"].cpu()[k], p["scores"].cpu()[k])]
            vis = draw_bounding_boxes((img * 255).byte(), p["boxes"].cpu()[k], names, colors="red", width=3)
            save_image(vis.float() / 255, OUT / f"ssd_pred_{i:02d}.jpg")

    # start the main loop
    ssd_main()



def problem1():
    """
    a. Load and explore the dataset (classes, annotation formats, etc.), 
        visualize sample images with bounding boxes, 
        and split it into training and validation/test sets.

    b. Apply YOLOv5  and SSD object detection models.
    c. Train the models on the training set. Apply relevant augmentations (flipping, scaling, colour jittering). Track training progress, and monitor loss and mean Average Precision (mAP).
    d. Evaluate the models using a test set. Report the evaluation results in terms of the metrics mAP@0.5, precision, recall, and inference speed. Visualize detection outputs on at least 10 test images.

    Notes:
        Ultralytics' stock coco128.yaml uses the same 128 images for train and val
        YOLOv5 and SSD weights were trained on COCO train2017, 
            which contain the 128 images that are being used for this assignment. 
            As a result test mAP (Acuracy) is expected to be inflated. 
    """

    # loadExploreDataset(IMG_DIR)


    # yolo()
    ssd()




    
def main():
    problem1()

if __name__ == "__main__":
    main()
