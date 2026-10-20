"""Problem 2: semantic segmentation on Pascal VOC 2012 with U-Net (scratch), U-Net (pretrained
ResNet-34 encoder) and DeepLabV3 (pretrained ResNet-50 backbone). Mask R-CNN is in p2_maskrcnn.py
and reuses the dataset, metrics, post-processing and plotting defined here.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Subset
from torchvision import models, tv_tensors
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent
VOC_ROOT = ROOT / "data"                # expects data/VOCdevkit/VOC2012/...
OUT = ROOT / "runs" / "seg"; OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
SIZE, BATCH, EPOCHS, NUM_CLASSES, IGNORE = 256, 8, 30, 21, 255
PRETRAINED = True
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet stats (pretrained encoders)

# ----------------------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------------------
# v2 transforms see the mask as a tv_tensors.Mask, so geometric ops are applied to image AND mask
# (nearest-neighbour for the mask, so class ids are never blended) and colour ops only to the image.
train_tf = v2.Compose([
    v2.RandomResizedCrop(SIZE, scale=(0.5, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(15, fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE}),
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


# ----------------------------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------------------------
def double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Classic U-Net (Ronneberger et al. 2015): 4 down levels, bottleneck, 4 up levels with
    concatenating skip connections. Same filter plan as the Brain-MRI U-Net in Unit 12 §5.3."""

    def __init__(self, n_classes=NUM_CLASSES, f=(32, 64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList([double_conv(3, f[0])] +
                                   [double_conv(f[i], f[i + 1]) for i in range(3)])
        self.bottleneck = double_conv(f[3], f[4])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([double_conv(f[i] * 2, f[i]) for i in reversed(range(4))])
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
        self.dec = nn.ModuleList([double_conv(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4))])
        self.head = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  double_conv(64, 32), nn.Conv2d(32, n_classes, 1))

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


# ----------------------------------------------------------------------------------------------
# 3. Losses
# ----------------------------------------------------------------------------------------------
def dice_loss(logits, target, eps=1.0):
    """Multi-class soft Dice: 1 - 2|P∩G| / (|P|+|G|), averaged over classes; ignores 255 pixels.
    (For a binary mask this reduces to the familiar sigmoid + Dice used with BCE.)"""
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


# ----------------------------------------------------------------------------------------------
# 4. Metrics (from one confusion matrix)
# ----------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------
# 5. Post-processing
# ----------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------
# 6. Train / evaluate
# ----------------------------------------------------------------------------------------------
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


def main():
    download_voc()
    train_ds, val_ds, test_ds = get_splits()
    inspect(train_ds)
    print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")
    specs = {"UNet_scratch": (UNet(), 1e-3), "UNet_ResNet34": (ResNetUNet(), 3e-4), "DeepLabV3_R50": (DeepLab(), 3e-4)}
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


if __name__ == "__main__":
    main()
