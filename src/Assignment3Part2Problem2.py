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
    e. Use a pre-trained model (e.g., ResNet or MobileNet). 
        Add post-processing (e.g., CRFs or morphological ops). 
        Compare the performances side by side.



    Mask R-CNN (instance segmentation) fine-tuned on VOC 2012, 
    then scored with the SAME semantic metrics as
        U-Net / DeepLabV3 by 
    painting its instance masks into a class map.

"""
import os
# Must be set before torch is imported: torchvision::nms (used by YOLO validation/predict) has no MPS kernel,
# so unsupported ops fall back to the CPU while the rest of the model stays on the GPU.
#   this will be slower than running natively on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import sys
import time
import numpy as np
import random
from collections import Counter
from pathlib import Path
from PIL import Image
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
# UserWarning: scatter_reduce_mps does not have a deterministic implementation
torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Subset
from torchvision import models, tv_tensors
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes


ROOT = Path(__file__).resolve().parent
VOC_ROOT = ROOT / "data"                # expects data/VOCdevkit/VOC2012/...
# Smoke-test mode:  DEBUG=1 python src/Assignment3Part2Problem2.py maskrcnn
#   1 epoch, tiny subsets -> finishes in a minute or two so end-of-script bugs show up fast.
DEBUG = os.environ.get("DEBUG") == "1"
OUT = ROOT / "runs" / ("segmentations_debug" if DEBUG else "segmentations"); OUT.mkdir(parents=True, exist_ok=True)

SIZE = 256
BATCH = 4 # 5-10
EPOCHS = 1 if DEBUG else 30
NUM_CLASSES = 21
IGNORE = 255

PRETRAINED = True
LR = 5e-3

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet stats (pretrained encoders)


from model_eval import fit, evaluateAndPlot, evaluateLoader
DOC_FIGS_NAME ="assignment3part2problem2"

# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
#                       "cuda" if torch.cuda.is_available() else "cpu")
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"
DEVICE = get_device()
# Mask R-CNN leans on roi_align + NMS, which have no real MPS kernels: they fall back to the CPU with a sync/copy
# on every call. Measured here: ~40 s per 4-image train step on "mps" vs ~3.4 s on "cpu".
RCNN_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"

# ---semantic segmentation on Pascal VOC---
# 1. Data
# v2 transforms sees the mask as a tv_tensors.Mask
#   , so geometric ops are applied to image AND mask
# 
# (nearest-neighbour for the mask, so class ids are never blended)
#   and colour ops only to the image.
train_tf = v2.Compose([
    v2.RandomResizedCrop(SIZE, scale=(0.5, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(str(15), fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE}),
    v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
    v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
    v2.Normalize(MEAN, STD),
])
eval_tf = v2.Compose([
    v2.Resize((SIZE, SIZE), antialias=True),
    v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
    v2.Normalize(MEAN, STD),
])

class VOCSeg(torch.utils.data.Dataset):
    """Wraps torchvision's VOCSegmentation so the mask is a tv_tensors.Mask of class ids (0-20, 255)."""

    def __init__(self, image_set, transforms):
        self.voc = VOCSegmentation(VOC_ROOT, year="2012", image_set=image_set, download=False)
        self.transforms = transforms

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, i):
        img, mask = self.voc[i]                       # PIL RGB, PIL 'P' (palette index = class id)
        img = tv_tensors.Image(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(np.array(mask)))   # H x W, uint8
        img, mask = self.transforms(img, mask)
        return img, mask.long()

def download_voc():
    """~2 GB. torchvision's built-in download points at host.robots.ox.ac.uk, which is often down;
    the D2L book (Unit 12 §4.1) hosts the identical tar on S3, so fall back to that."""
    if (VOC_ROOT / "VOCdevkit" / "VOC2012" / "SegmentationClass").exists():
        return
    try:
        VOCSegmentation(VOC_ROOT, year="2012", image_set="train", download=True)
    except Exception as e:
        import tarfile, urllib.request
        print("official mirror failed:", e, "-> using the D2L mirror")
        tar = VOC_ROOT / "VOCtrainval_11-May-2012.tar"
        urllib.request.urlretrieve("http://d2l-data.s3-accelerate.amazonaws.com/VOCtrainval_11-May-2012.tar", tar)
        tarfile.open(tar).extractall(VOC_ROOT)

def get_splits():
    """VOC 'train' (1,464) -> train; VOC 'val' (1,449) -> half validation, half test."""
    train = VOCSeg("train", train_tf)
    val_full = VOCSeg("val", eval_tf)
    idx = torch.randperm(len(val_full), generator=torch.Generator().manual_seed(0)).tolist()
    half = len(idx) // 2
    return train, Subset(val_full, idx[:half]), Subset(val_full, idx[half:])


# Models
def _double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class UNet(nn.Module):
    """Classic U-Net (Ronneberger et al. 2015): 4 down levels, bottleneck, 4 up levels with
    concatenating skip connections. Same filter plan as the Brain-MRI U-Net in Unit 12 §5.3."""

    def __init__(self, n_classes=NUM_CLASSES, f=(32, 64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList([_double_conv(3, f[0])] +
                                   [_double_conv(f[i], f[i + 1]) for i in range(3)])
        self.bottleneck = _double_conv(f[3], f[4])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([_double_conv(f[i] * 2, f[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(f[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)                                   # learnable 2x upsampling (transposed conv)
            x = dec(torch.cat([skip, x], dim=1))        # the U-Net skip connection
        return self.head(x)

class ResNetUNet(nn.Module):
    """U-Net decoder on an ImageNet-pretrained ResNet-34 encoder (transfer learning)."""

    def __init__(self, n_classes=NUM_CLASSES, pretrained=PRETRAINED):
        super().__init__()
        r = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)           # 1/2,  64 ch
        self.pool = r.maxpool
        self.enc = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])  # 1/4..1/32: 64,128,256,512
        ch = [64, 64, 128, 256, 512]
        self.dec = nn.ModuleList([_double_conv(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4))])
        self.head = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  _double_conv(64, 32), nn.Conv2d(32, n_classes, 1))

    def forward(self, x):
        s0 = self.stem(x)
        feats, h = [s0], self.pool(s0)
        for layer in self.enc:
            h = layer(h); feats.append(h)
        x = feats.pop()                                  # deepest feature map (bottleneck)
        for dec in self.dec:
            skip = feats.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)

class DeepLab(nn.Module):
    """torchvision DeepLabV3 (ASPP head, dilated ResNet-50). Only the BACKBONE is pretrained
    (ImageNet). The 'DEFAULT' full-model weights were trained on COCO with the 20 VOC classes, so
    they would hand DeepLab an unfair head start over the other models — keep that out, or report it."""

    def __init__(self, pretrained=PRETRAINED, mobilenet=False):
        super().__init__()
        seg = models.segmentation
        if mobilenet:   # lighter option for the 'ResNet vs MobileNet' comparison
            self.net = seg.deeplabv3_mobilenet_v3_large(weights=None, num_classes=NUM_CLASSES, aux_loss=True,
                weights_backbone=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            self.net = seg.deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES, aux_loss=True,
                weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

    def forward(self, x):
        out = self.net(x)                 # dict: 'out' and (training) 'aux'
        return out if self.training else out["out"]


# Losses methdos
def dice_loss(logits, target, eps=1.0):
    """Multi-class soft Dice:
        1 - 2|P∩G| / (|P|+|G|)
        , averaged over all classes; ignores 255 pixels.

    (For a binary-mask this reduces to the familiar sigmoid + Dice used with BCE.)
    """
    valid = (target != IGNORE)
    prob = logits.softmax(1) * valid.unsqueeze(1)
    onehot = F.one_hot(target.clamp(max=NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2) * valid.unsqueeze(1)
    inter = (prob * onehot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def soft_iou_loss(logits, target, eps=1.0):
    """Soft Jaccard: 1 - |P∩G| / |P∪G| — optimises the evaluation metric (IoU) directly."""
    valid = (target != IGNORE)
    prob = logits.softmax(1) * valid.unsqueeze(1)
    onehot = F.one_hot(target.clamp(max=NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2) * valid.unsqueeze(1)
    inter = (prob * onehot).sum((0, 2, 3))
    union = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) - inter
    return 1 - ((inter + eps) / (union + eps)).mean()


def seg_loss(out, target, mode="ce+dice"):
    """Cross-entropy is the multi-class form of BCE; combine it with an overlap loss."""
    if isinstance(out, dict):        # DeepLab training: main + 0.4 * auxiliary head
        return seg_loss(out["out"], target, mode) + 0.4 * seg_loss(out["aux"], target, mode)
    ce = F.cross_entropy(out, target, ignore_index=IGNORE)
    if mode == "ce":
        return ce
    return ce + (dice_loss(out, target) if mode == "ce+dice" else soft_iou_loss(out, target))


# 4. Metrics (from one confusion matrix)
def update_confusion(cm, pred, target):
    k = target != IGNORE
    cm += torch.bincount(target[k] * NUM_CLASSES + pred[k], minlength=NUM_CLASSES ** 2
                         ).reshape(NUM_CLASSES, NUM_CLASSES).cpu()
    return cm


def scores(cm):
    cm = cm.double()
    tp, fp, fn = cm.diag(), cm.sum(0) - cm.diag(), cm.sum(1) - cm.diag()
    present = cm.sum(1) > 0                               # classes that appear in the ground truth
    iou, dice = tp / (tp + fp + fn).clamp(min=1), 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    return {"pixel_acc": (tp.sum() / cm.sum()).item(), "mIoU": iou[present].mean().item(),
            "mDice": dice[present].mean().item(), "per_class_IoU": iou.tolist()}

# 5. Post-processing
def morph_postprocess(pred, min_size=64):
    """Per class: opening (removes specks), closing (fills pin-holes), then drop tiny connected
    components; reassigned pixels fall back to the most common neighbouring label (background here)."""
    out = np.zeros_like(pred)
    st = ndimage.generate_binary_structure(2, 1)
    for c in np.unique(pred):
        if c == 0:
            continue
        m = ndimage.binary_closing(ndimage.binary_opening(pred == c, st, 1), st, 2)
        lab, n = ndimage.label(m)
        if n:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            m = np.isin(lab, 1 + np.flatnonzero(sizes >= min_size))
        out[m] = c
    return out

def crf_postprocess(img_uint8, probs, iters=5):
    """Dense CRF (Krähenbühl & Koltun 2011), as used by DeepLab v1/v2. Optional — needs
    `pip install git+https://github.com/lucasb-eyer/pydensecrf.git`."""
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    C, H, W = probs.shape
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(unary_from_softmax(probs))
    d.addPairwiseGaussian(sxy=3, compat=3)                                        # smoothness
    d.addPairwiseBilateral(sxy=60, srgb=10, rgbim=np.ascontiguousarray(img_uint8), compat=5)  # edges
    return np.argmax(np.array(d.inference(iters)).reshape(C, H, W), 0)


def denorm(img):
    return (img.cpu() * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1)

# Train
@torch.no_grad()
def evaluate(model, ds, post=None):
    """post: None | 'morph' | 'crf'. Returns metric dict."""
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for img, mask in DataLoader(ds, batch_size=BATCH):
        logits = model(img.to(DEVICE))
        pred = logits.argmax(1).cpu()
        if post == "morph":
            pred = torch.from_numpy(np.stack([morph_postprocess(p.numpy()) for p in pred]))
        elif post == "crf":
            probs = logits.softmax(1).cpu().numpy().astype(np.float32)
            pred = torch.from_numpy(np.stack([crf_postprocess((denorm(i).permute(1, 2, 0).numpy() * 255).astype(np.uint8), p)
                                              for i, p in zip(img, probs)]))
        cm = update_confusion(cm, pred, mask)
    return scores(cm)

def train(name, model, train_ds, val_ds, lr, loss_mode="ce+dice"):
    model.to(DEVICE)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=EPOCHS * len(loader))
    hist = []
    for ep in range(EPOCHS):
        model.train(); run = 0.0
        for img, mask in loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            loss = seg_loss(model(img), mask, loss_mode)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item() / len(loader)
        v = evaluate(model, val_ds)
        hist.append({"epoch": ep + 1, "train_loss": run, **{k: v[k] for k in ("pixel_acc", "mIoU", "mDice")}})
        print(f"[{name}] epoch {ep + 1:>3}/{EPOCHS} loss {run:.3f} | val pixAcc {v['pixel_acc']:.3f} "
              f"mIoU {v['mIoU']:.3f} dice {v['mDice']:.3f}")
    torch.save(model.state_dict(), OUT / f"{name}.pth")
    json.dump(hist, open(OUT / f"{name}_history.json", "w"), indent=1)
    return hist


def plot_histories(histories):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        ax[0].plot(ep, [r["train_loss"] for r in h], label=name)
        ax[1].plot(ep, [r["mIoU"] for r in h], label=name)
        ax[2].plot(ep, [r["mDice"] for r in h], label=name)
    for a, t in zip(ax, ["training loss", "validation mIoU", "validation mean Dice"]):
        a.set_title(t); a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(OUT / "seg_curves.png", dpi=150); plt.close()


def voc_palette(n=256):
    """The standard VOC colour map (class 1 = (128,0,0) aeroplane, 15 = (192,128,128) person ...)."""
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j); g |= ((c >> 1) & 1) << (7 - j); b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (r, g, b)
    return pal


def overlay(img, mask, alpha=0.5):
    """Colour each class with the VOC palette and alpha-blend over the image."""
    pal = voc_palette()
    col = pal[mask.clip(0, 255)] / 255.0
    show = mask[..., None] > 0
    return np.where(show & (mask[..., None] != IGNORE), (1 - alpha) * img + alpha * col, img)


@torch.no_grad()
def visualise(models_preds, ds, n=6, fname="seg_compare.png"):
    """models_preds: {name: fn(img_batch)->label map (B,H,W)} — columns = GT + each model."""
    cols = ["ground truth"] + list(models_preds)
    fig, ax = plt.subplots(n, len(cols), figsize=(3 * len(cols), 3 * n))
    for r in range(n):
        img, mask = ds[r]
        rgb = denorm(img).permute(1, 2, 0).numpy()
        panels = [mask.numpy()] + [fn(img.unsqueeze(0).to(DEVICE))[0].cpu().numpy() for fn in models_preds.values()]
        for c, (title, m) in enumerate(zip(cols, panels)):
            ax[r, c].imshow(overlay(rgb, m)); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(title)
    plt.tight_layout(); plt.savefig(OUT / fname, dpi=120); plt.close()


def inspect(ds, n=4):
    """Step 1 'load and inspect': raw image, raw mask values, class pixel frequencies."""
    voc = ds.voc if hasattr(ds, "voc") else ds.dataset.voc
    img, mask = voc[0]
    m = np.array(mask)
    print("image", img.size, img.mode, "| mask", m.shape, mask.mode, "| values", np.unique(m))
    counts = np.zeros(NUM_CLASSES)
    for i in range(0, len(voc), 10):                     # every 10th mask is enough for a histogram
        counts += np.bincount(np.array(voc[i][1]).ravel(), minlength=256)[:NUM_CLASSES]
    plt.figure(figsize=(10, 4)); plt.bar(VOC_CLASSES, counts / counts.sum()); plt.yscale("log")
    plt.xticks(rotation=60, ha="right"); plt.ylabel("fraction of pixels (log)"); plt.title("VOC class pixel share")
    plt.tight_layout(); plt.savefig(OUT / "voc_class_share.png", dpi=150); plt.close()
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    for k in range(n):
        im, mk = voc[(k * 97) % len(voc)]
        ax[0, k].imshow(im); ax[1, k].imshow(overlay(np.array(im) / 255.0, np.array(mk)))
        ax[0, k].axis("off"); ax[1, k].axis("off")
    plt.tight_layout(); plt.savefig(OUT / "voc_samples.png", dpi=120); plt.close()


# IDK if this is right
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


def seg():
    download_voc()
    train_ds, val_ds, test_ds = get_splits()
    inspect(train_ds)
    print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")
    specs = {
        "UNet_scratch": (UNet(), 1e-3), 
        "UNet_ResNet34": (ResNetUNet(), 3e-4), 
        "DeepLabV3_R50": (DeepLab(), 3e-4)
    }
    histories, results, fns = {}, {}, {}

    for name, (model, lr) in specs.items():
        histories[name] = train(name, model, train_ds, val_ds, lr)
        for post in (None, "morph"):            # add "crf" if pydensecrf is installed
            results[f"{name}{'+' + post if post else ''}"] = evaluate(model, test_ds, post)
        fns[name] = lambda x, m=model: m.eval()(x).argmax(1)
    plot_histories(histories)
    visualise(fns, test_ds)
    print(f"\n{'model':<26}{'pixAcc':>8}{'mIoU':>8}{'mDice':>8}")
    for k, r in results.items():
        print(f"{k:<26}{r['pixel_acc']:>8.3f}{r['mIoU']:>8.3f}{r['mDice']:>8.3f}")
    json.dump(results, open(OUT / "test_results.json", "w"), indent=1)


# ---Mask RCNN---
class pascalVOC(torch.utils.data.Dataset):
    """
    Object of the Segmentation dataset
    SegmentationObject: PNG holds an instance id per pixel (1-N, 255 = boarder)
    The class of each instance is a read from SegmentationClass at the same pixels
    """
    def __init__(self, image_set, train) -> None:
        self.voc = VOCSegmentation(VOC_ROOT, year="2012", image_set=image_set, download=False)

        # self.obj = [m.replace("SegmentationClass", "SegmentationObject") for m in self.voc.masks]
        self.obj = []
        for m in self.voc.masks:
            mask = m.replace("SegmentationClass", "SegmentationObject")
            self.obj.append(mask)

        # geo = [v2.RandomHorizontalFlip(), v2.RandomPhotometricDistort(p=0.5)] if train else []
        geo = []
        if train:
            thing = v2.RandomHorizontalFlip(), v2.RandomPhotometricDistort(p=0.5)
            geo.append(thing)

        # (*) the unpacking operator
        self.tf = v2.Compose([
            v2.Resize(SIZE, antialias=True),            # shorter side -> SIZE
            *geo, 
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        # super().__init__()

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, i):
        img, cls = self.voc[i]
        cls, inst = np.array(cls), np.array(Image.open(self.obj[i]))
        
        # ids = [k for k in np.unique(inst) if k not in (0, IGNORE)]
        ids = []
        for i in np.unique(inst):
            if i not in (0, IGNORE):
                ids.append(i) 

        # masks = torch.from_numpy(np.stack([inst == k for k in ids])) if ids else torch.zeros(0, *inst.shape, dtype=torch.bool)
        if ids:
            boolmasks = []
            # Create a list of boolean masks for each ID
            for k in ids:
                boolmasks.append(inst == k  )
            
            # Stack the boolean masks along a new first dimension
            ndStackedMasks = np.stack(boolmasks)
            
            # Convert the NumPy array to a PyTorch tensor
            masks = torch.from_numpy(ndStackedMasks)
        else:
            masks = torch.zeros(0, *inst.shape, dtype=torch.bool)

        labels = torch.tensor([np.bincount(cls[inst == k]).argmax() for k in ids], dtype=torch.int64)
        label_list = []
        for i in ids:
            # Get class value for the pixels belonging to i
            pixel_clses = cls[inst == i]
            # Count the frequency of each class in those pixels
            cls_counts = np.bincount(pixel_clses)
            # Find the most frequent class (the majority vote)
            majority_class = cls_counts.argmax()
            # Save it to our list
            label_list.append(majority_class)

        # Finally! Convert the final list of majority classes into a PyTorch tensor
        labels = torch.tensor(label_list, dtype=torch.int64)
        H, W = inst.shape
        canvas_size: tuple[int, int] = (int(H), int(W))
        
        target = {
            "boxes":tv_tensors.BoundingBoxes(
                    masks_to_boxes(masks) if ids else torch.zeros(0, 4),
                    format=BoundingBoxFormat.XYXY, 
                    canvas_size=canvas_size
                ),
            "masks":tv_tensors.Mask(masks.to(torch.uint8)), 
            "labels": labels,
            "semantic": tv_tensors.Mask(torch.from_numpy(cls).long())   # keep for evaluation
        }

        img = tv_tensors.Image(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        return self.tf(img, target)

def build_model():
    """Unit 11 §3.2 'Approach A': start from COCO-pretrained Mask R-CNN, replace both heads."""
    m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if PRETRAINED else None,
                              weights_backbone=None,
                              min_size=SIZE, max_size=2 * SIZE)   # default rescales to 800 px -> slow + memory hungry
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, NUM_CLASSES)
    return m

def to_semantic(pred, shape, score_thr=0.5, mask_thr=0.5):
    """Paint instances lowest-score first, so the most confident instance wins overlaps."""
    sem = torch.zeros(shape, dtype=torch.long)
    keep = pred["scores"] >= score_thr
    order = pred["scores"][keep].argsort()
    for m, l in zip(pred["masks"][keep][order], pred["labels"][keep][order]):
        sem[m[0].cpu() >= mask_thr] = l.item()
    return sem

@torch.no_grad()
def evaluate2(model, ds, post=None):
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for img, tgt in ds:
        p = model([img.to(RCNN_DEVICE)])[0]
        sem = to_semantic(p, img.shape[-2:])
        if post == "morph":
            sem = torch.from_numpy(morph_postprocess(sem.numpy()))
        cm = update_confusion(cm, sem, tgt["semantic"])
    return scores(cm)

def maskRcnn():
    EPOCHS = 10
    train_ds = pascalVOC("train", train=True)
    val_full = pascalVOC("val", train=False)
    idx = torch.randperm(len(val_full), generator=torch.Generator().manual_seed(0)).tolist()   # same split as p2_seg
    val_ds, test_ds = Subset(val_full, idx[:len(idx) // 2]), Subset(val_full, idx[len(idx) // 2:])
    if DEBUG:
        train_ds, val_ds, test_ds = (Subset(d, range(16)) for d in (train_ds, val_ds, test_ds))
    print(f"[MaskRCNN] train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}  device {RCNN_DEVICE}", flush=True)

    model = build_model().to(RCNN_DEVICE)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    hist = []
    for ep in range(EPOCHS):
        model.train(); tot = {}; t0 = time.time()
        for step, (imgs, tgts) in enumerate(loader):
            if step % 50 == 0:
                print(f"[MaskRCNN] epoch {ep + 1} batch {step}/{len(loader)}  {time.time() - t0:.0f}s", flush=True)
            imgs = [i.to(RCNN_DEVICE) for i in imgs]
            tgts = [{k: t[k].to(RCNN_DEVICE) for k in ("boxes", "labels", "masks")} for t in tgts]
            losses = model(imgs, tgts)        # loss_classifier, loss_box_reg, loss_mask (per-pixel BCE), loss_objectness, loss_rpn_box_reg
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in losses.items(): tot[k] = tot.get(k, 0) + v.item() / len(loader)
        sched.step()
        t_train = time.time() - t0
        v = evaluate2(model, val_ds)
        hist.append({"epoch": ep + 1, "train_loss": sum(tot.values()), **tot,
                     **{k: v[k] for k in ("pixel_acc", "mIoU", "mDice")}})
        print(f"[MaskRCNN] epoch {ep + 1}/{EPOCHS} loss {sum(tot.values()):.3f} (mask {tot['loss_mask']:.3f}) "
              f"| val mIoU {v['mIoU']:.3f} dice {v['mDice']:.3f} "
              f"| train {t_train:.0f}s eval {time.time() - t0 - t_train:.0f}s", flush=True)
    torch.save(model.state_dict(), OUT / "MaskRCNN.pth")
    json.dump(hist, open(OUT / "MaskRCNN_history.json", "w"), indent=1)
    res = {"MaskRCNN": evaluate2(model, test_ds), "MaskRCNN+morph": evaluate2(model, test_ds, "morph")}
    for k, r in res.items():
        print(f"{k:<26}{r['pixel_acc']:>8.3f}{r['mIoU']:>8.3f}{r['mDice']:>8.3f}")
    json.dump(res, open(OUT / "MaskRCNN_test_results.json", "w"), indent=1)


# ---Report figures + tables: ONE evaluation pass over the saved weights feeds every number and picture---
MODEL_ORDER = ["UNet_scratch", "UNet_ResNet34", "DeepLabV3_R50", "MaskRCNN"]
MODEL_LABEL = {"UNet_scratch": "U-Net (scratch)", "UNet_ResNet34": "U-Net + ResNet-34",
               "DeepLabV3_R50": "DeepLabV3-ResNet50", "MaskRCNN": "Mask R-CNN"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # dataviz categorical slots 1-4 (validated, light mode)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def per_class_scores(cm):
    """Per-class IoU and Dice straight from a confusion matrix (rows = ground truth, cols = prediction)."""
    cm = cm.double(); tp = cm.diag(); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (tp / (tp + fp + fn).clamp(min=1)).tolist(), (2 * tp / (2 * tp + fp + fn).clamp(min=1)).tolist()


def _load_report_models():
    models_ = {"UNet_scratch": UNet(), "UNet_ResNet34": ResNetUNet(pretrained=False),
               "DeepLabV3_R50": DeepLab(pretrained=False), "MaskRCNN": build_model()}
    for name, m in models_.items():
        m.load_state_dict(torch.load(OUT / f"{name}.pth", map_location="cpu"))
        m.eval()
    for name in MODEL_ORDER[:3]:
        models_[name].to(DEVICE)
    models_["MaskRCNN"].to(RCNN_DEVICE)
    return models_


@torch.no_grad()
def _confusions(name, model, sem_ds, rcnn_ds):
    """Raw and morph-post-processed confusion matrices from a single forward pass per image."""
    cms = {"raw": torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long),
           "morph": torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)}
    if name == "MaskRCNN":
        for img, tgt in rcnn_ds:
            sem = to_semantic(model([img.to(RCNN_DEVICE)])[0], img.shape[-2:])
            update_confusion(cms["raw"], sem, tgt["semantic"])
            update_confusion(cms["morph"], torch.from_numpy(morph_postprocess(sem.numpy())), tgt["semantic"])
    else:
        for img, mask in DataLoader(sem_ds, batch_size=BATCH):
            pred = model(img.to(DEVICE)).argmax(1).cpu()
            update_confusion(cms["raw"], pred, mask)
            pm = torch.from_numpy(np.stack([morph_postprocess(q.numpy()) for q in pred]))
            update_confusion(cms["morph"], pm, mask)
    return cms


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def plot_per_class(results, fname="seg_per_class.png"):
    """Grouped bars: per-class IoU (top) and Dice (bottom) for the four models, raw predictions."""
    x = np.arange(NUM_CLASSES); w = 0.2
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True, facecolor=SURFACE)
    handles = []
    for ax, key, title in zip(axes, ("per_class_IoU", "per_class_Dice"), ("IoU", "Dice")):
        _style_axes(ax)
        for k, name in enumerate(MODEL_ORDER):
            r = results[name]["raw"]
            label = f"{MODEL_LABEL[name]}  (mIoU {r['mIoU']:.3f} · mDice {r['mDice']:.3f})"
            bars = ax.bar(x + (k - 1.5) * w, r[key], width=w - 0.02, color=SERIES[k], label=label,
                          zorder=3, linewidth=0)
            if title == "IoU":
                handles.append(bars)
        ax.set_ylim(0, 1); ax.set_ylabel(title, color=INK2)
        ax.set_title(f"Per-class {title} on the test split (725 images, raw predictions)", color=INK,
                     loc="left", fontsize=12)
    axes[1].set_xticks(x); axes[1].set_xticklabels(VOC_CLASSES, rotation=45, ha="right", color=INK2)
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=2, frameon=False,
               labelcolor=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / fname, dpi=150, facecolor=SURFACE); plt.close(fig)


@torch.no_grad()
def plot_overlays(models_, test_ds, rcnn_full, n=8, fname="seg_overlays.png"):
    """Rows = test images; columns = image, ground truth, four models, and morph-post-processed DeepLab / Mask R-CNN."""
    cols = ["image", "ground truth"] + [MODEL_LABEL[m] for m in MODEL_ORDER] + \
           [MODEL_LABEL["DeepLabV3_R50"] + "\n+ morph", MODEL_LABEL["MaskRCNN"] + "\n+ morph"]
    fig, ax = plt.subplots(n, len(cols), figsize=(2.3 * len(cols), 2.3 * n + 0.8), facecolor=SURFACE)
    to_256 = lambda m: F.interpolate(m[None, None].float(), size=(SIZE, SIZE), mode="nearest")[0, 0].long()
    for r in range(n):
        img, mask = test_ds[r]
        rgb = denorm(img).permute(1, 2, 0).numpy()
        preds = {m: models_[m](img[None].to(DEVICE)).argmax(1)[0].cpu() for m in MODEL_ORDER[:3]}
        rimg, _ = rcnn_full[test_ds.indices[r]]                 # same VOC image, Mask R-CNN's own preprocessing
        sem = to_semantic(models_["MaskRCNN"]([rimg.to(RCNN_DEVICE)])[0], rimg.shape[-2:])
        preds["MaskRCNN"] = to_256(sem)
        morph = lambda t: torch.from_numpy(morph_postprocess(t.numpy()))
        panels = [None, mask.numpy()] + [preds[m].numpy() for m in MODEL_ORDER] + \
                 [morph(preds["DeepLabV3_R50"]).numpy(), to_256(morph(sem)).numpy()]
        for c, m in enumerate(panels):
            ax[r, c].imshow(rgb if m is None else overlay(rgb, m)); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=9, color=INK)
    pal = voc_palette() / 255.0
    handles = [patches.Patch(facecolor="white" if c == 0 else pal[c], edgecolor=INK2, linewidth=0.6,
                             label="background (not shaded)" if c == 0 else VOC_CLASSES[c]) for c in range(NUM_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=11, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 1)); fig.savefig(OUT / fname, dpi=110, facecolor=SURFACE); plt.close(fig)


def markdown_tables(results):
    rows = ["| Model | Pixel acc | mIoU | mDice |", "|---|---|---|---|"]
    for name in MODEL_ORDER:
        for post, tag in (("raw", ""), ("morph", " + morph")):
            r = results[name][post]
            rows.append(f"| {MODEL_LABEL[name]}{tag} | {r['pixel_acc']:.3f} | {r['mIoU']:.3f} | {r['mDice']:.3f} |")
    head = "| Class | " + " | ".join(f"{MODEL_LABEL[m]} IoU" for m in MODEL_ORDER) + " | " + \
           " | ".join(f"{MODEL_LABEL[m]} Dice" for m in MODEL_ORDER) + " |"
    pc = ["", head, "|" + "---|" * (1 + 2 * len(MODEL_ORDER))]
    for c in range(NUM_CLASSES):
        iou = [results[m]["raw"]["per_class_IoU"][c] for m in MODEL_ORDER]
        dice = [results[m]["raw"]["per_class_Dice"][c] for m in MODEL_ORDER]
        pc.append(f"| {VOC_CLASSES[c]} | " + " | ".join(f"{v:.3f}" for v in iou + dice) + " |")
    return "\n".join(rows + pc)


def report_figures():
    """Re-evaluate the saved weights once; write results JSON, markdown tables, the overlay figure and the
    per-class chart. Every number in the report should be copied from report_results.json / report_tables.md."""
    _, val_ds, test_ds = get_splits()
    rcnn_full = pascalVOC("val", train=False)
    rcnn_test = Subset(rcnn_full, test_ds.indices)             # identical images to the semantic test split
    models_ = _load_report_models()
    results = {}
    for name in MODEL_ORDER:
        t0 = time.time()
        cms = _confusions(name, models_[name], test_ds, rcnn_test)
        results[name] = {}
        for post, cm in cms.items():
            iou, dice = per_class_scores(cm)
            results[name][post] = {**{k: v for k, v in scores(cm).items() if k != "per_class_IoU"},
                                   "per_class_IoU": iou, "per_class_Dice": dice}
        print(f"[report] {name:<14} test mIoU {results[name]['raw']['mIoU']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(results, open(OUT / "report_results.json", "w"), indent=1)
    (OUT / "report_tables.md").write_text(markdown_tables(results))
    plot_per_class(results)
    plot_overlays(models_, test_ds, rcnn_full)
    print(markdown_tables(results))
    # sanity check against the numbers written when the models were trained
    old = {**json.load(open(OUT / "test_results.json")), **json.load(open(OUT / "MaskRCNN_test_results.json"))}
    for name in MODEL_ORDER:
        for post, suffix in (("raw", ""), ("morph", "+morph")):
            d = abs(old[name + suffix]["mIoU"] - results[name][post]["mIoU"])
            print(f"[check] {name + suffix:<22} |mIoU old - new| = {d:.4f}")


def problem2():
    """
    Implement an image segmentation model 
        using deep learning to segment objects 
        from images at the pixel level. 
    
    Explore architectures U-Net, DeepLabV3, and Mask R-CNN 
        for segmentation purposes. 
    
    Use the Pascal VOC segmentation dataset for this problem.

    Explore architectures U-Net, DeepLabV3, and Mask R-CNN for segmentation purposes
    """
    
    # setup()
    # usage: python src/Assignment3Part2Problem2.py [seg] [maskrcnn] [figures]   (default: seg + maskrcnn)
    args = sys.argv[1:] or ["seg", "maskrcnn"]
    if "seg" in args:
        seg()
    if "maskrcnn" in args:
        maskRcnn()
    if "figures" in args:
        report_figures()
    
def main():
    problem2()

if __name__ == "__main__":
    main()
