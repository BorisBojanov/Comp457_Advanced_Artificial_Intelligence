"""Problem 1, Step 2-4 (SSD side): fine-tune torchvision's SSD300-VGG16 on the COCO128 split.

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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "coco128"
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


def main():
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


if __name__ == "__main__":
    main()
