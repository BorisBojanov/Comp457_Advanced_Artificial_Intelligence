"""Problem 1, Step 2-4 (YOLO side): fine-tune YOLOv5 on the COCO128 split, evaluate, visualise."""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "coco128" / "coco128_split.yaml"
DEVICE = "mps"        # Mac: "mps" | NVIDIA: 0 | fallback: "cpu"
EPOCHS = 30

# yolov5su.pt = YOLOv5-small, COCO-pretrained, re-released by Ultralytics with an anchor-free head.
# (The original anchor-based repo is github.com/ultralytics/yolov5 — `python train.py --weights yolov5s.pt`.)
model = YOLO("yolov5su.pt")

# --- train: augmentations are hyper-parameters, applied on the fly to every batch ---
model.train(
    data=str(DATA), epochs=EPOCHS, imgsz=640, batch=16, device=DEVICE,
    fliplr=0.5,                           # horizontal flip probability
    scale=0.5,                            # random rescale +/- 50 %
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,    # colour jitter (hue / saturation / value)
    mosaic=1.0,                           # 4-image mosaic (YOLO's signature augmentation)
    project=str(ROOT / "runs"), name="yolov5s_coco128", exist_ok=True, plots=True,
)
# Training curves (box/cls/dfl loss, precision, recall, mAP50, mAP50-95 per epoch) are saved to
# runs/yolov5s_coco128/results.csv and results.png — put results.png in the report.

# --- evaluate on the held-out TEST split ---
best = YOLO(ROOT / "runs" / "yolov5s_coco128" / "weights" / "best.pt")
m = best.val(data=str(DATA), split="test", imgsz=640, device=DEVICE,
             project=str(ROOT / "runs"), name="yolov5s_test", exist_ok=True)
print(f"YOLOv5s test | mAP@0.5 {m.box.map50:.3f}  mAP@0.5:0.95 {m.box.map:.3f}  "
      f"P {m.box.mp:.3f}  R {m.box.mr:.3f}  "
      f"speed/img: pre {m.speed['preprocess']:.1f} ms  infer {m.speed['inference']:.1f} ms  "
      f"post {m.speed['postprocess']:.1f} ms")

# --- visualise detections on 10+ test images ---
test_imgs = (ROOT / "data" / "coco128" / "test.txt").read_text().split()[:12]
best.predict(test_imgs, conf=0.25, device=DEVICE, save=True,
             project=str(ROOT / "runs"), name="yolov5s_pred", exist_ok=True)
