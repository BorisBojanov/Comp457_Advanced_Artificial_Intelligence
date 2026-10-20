"""Problem 2: Mask R-CNN (instance segmentation) fine-tuned on VOC 2012, then scored with the SAME
semantic metrics as U-Net / DeepLabV3 by painting its instance masks into a class map."""
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import tv_tensors
from torchvision.datasets import VOCSegmentation
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2

from p2_seg import (DEVICE, EPOCHS, IGNORE, NUM_CLASSES, OUT, PRETRAINED, SIZE, VOC_ROOT,
                    morph_postprocess, scores, update_confusion)

BATCH, LR = 4, 5e-3


class VOCInstances(torch.utils.data.Dataset):
    """SegmentationObject PNG holds an instance id per pixel (1..N, 255 = border); the class of
    each instance is read from SegmentationClass at the same pixels."""

    def __init__(self, image_set, train):
        self.voc = VOCSegmentation(VOC_ROOT, year="2012", image_set=image_set, download=False)
        self.obj = [m.replace("SegmentationClass", "SegmentationObject") for m in self.voc.masks]
        geo = [v2.RandomHorizontalFlip(), v2.RandomPhotometricDistort(p=0.5)] if train else []
        self.tf = v2.Compose([v2.Resize(SIZE, antialias=True), *geo,          # shorter side -> SIZE
                              v2.SanitizeBoundingBoxes(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, i):
        img, cls = self.voc[i]
        cls, inst = np.array(cls), np.array(Image.open(self.obj[i]))
        ids = [k for k in np.unique(inst) if k not in (0, IGNORE)]
        masks = torch.from_numpy(np.stack([inst == k for k in ids])) if ids else torch.zeros(0, *inst.shape, dtype=torch.bool)
        labels = torch.tensor([np.bincount(cls[inst == k]).argmax() for k in ids], dtype=torch.int64)
        H, W = inst.shape
        target = {"boxes": tv_tensors.BoundingBoxes(masks_to_boxes(masks) if ids else torch.zeros(0, 4),
                                                    format="XYXY", canvas_size=(H, W)),
                  "masks": tv_tensors.Mask(masks.to(torch.uint8)), "labels": labels,
                  "semantic": tv_tensors.Mask(torch.from_numpy(cls).long())}   # kept for evaluation
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
def evaluate(model, ds, post=None):
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for img, tgt in ds:
        p = model([img.to(DEVICE)])[0]
        sem = to_semantic(p, img.shape[-2:])
        if post == "morph":
            sem = torch.from_numpy(morph_postprocess(sem.numpy()))
        cm = update_confusion(cm, sem, tgt["semantic"])
    return scores(cm)


def main():
    train_ds = VOCInstances("train", train=True)
    val_full = VOCInstances("val", train=False)
    idx = torch.randperm(len(val_full), generator=torch.Generator().manual_seed(0)).tolist()   # same split as p2_seg
    val_ds, test_ds = Subset(val_full, idx[:len(idx) // 2]), Subset(val_full, idx[len(idx) // 2:])

    model = build_model().to(DEVICE)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    hist = []
    for ep in range(EPOCHS):
        model.train(); tot = {}
        for imgs, tgts in loader:
            imgs = [i.to(DEVICE) for i in imgs]
            tgts = [{k: t[k].to(DEVICE) for k in ("boxes", "labels", "masks")} for t in tgts]
            losses = model(imgs, tgts)        # loss_classifier, loss_box_reg, loss_mask (per-pixel BCE), loss_objectness, loss_rpn_box_reg
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in losses.items(): tot[k] = tot.get(k, 0) + v.item() / len(loader)
        sched.step()
        v = evaluate(model, val_ds)
        hist.append({"epoch": ep + 1, "train_loss": sum(tot.values()), **tot,
                     **{k: v[k] for k in ("pixel_acc", "mIoU", "mDice")}})
        print(f"[MaskRCNN] epoch {ep + 1}/{EPOCHS} loss {sum(tot.values()):.3f} (mask {tot['loss_mask']:.3f}) "
              f"| val mIoU {v['mIoU']:.3f} dice {v['mDice']:.3f}")
    torch.save(model.state_dict(), OUT / "MaskRCNN.pth")
    json.dump(hist, open(OUT / "MaskRCNN_history.json", "w"), indent=1)
    res = {"MaskRCNN": evaluate(model, test_ds), "MaskRCNN+morph": evaluate(model, test_ds, "morph")}
    for k, r in res.items():
        print(f"{k:<26}{r['pixel_acc']:>8.3f}{r['mIoU']:>8.3f}{r['mDice']:>8.3f}")
    json.dump(res, open(OUT / "MaskRCNN_test_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
