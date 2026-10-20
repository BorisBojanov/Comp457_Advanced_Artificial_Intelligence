# Assignment 3 — tutorial reference code

These are worked examples to go with the Notion page *Assignment 3 Tutorial*.
They work end to end on small test runs. Read them, then write your own `src/Assignment3Part2*.py`.

| File | Problem | What it does |
|---|---|---|
| `p1_explore.py` | P1 step 1 | downloads COCO128, class histogram, boxes on samples, 70/15/15 split + YAML |
| `p1_yolo.py` | P1 steps 2–4 | fine-tunes YOLOv5s (Ultralytics), test-set mAP@0.5/P/R/speed, saves predictions |
| `p1_ssd.py` | P1 steps 2–4 | fine-tunes torchvision SSD300-VGG16, curves, and scores **both** models with one evaluator |
| `p2_seg.py` | P2 steps 1–5 | VOC download/inspect, U-Net (scratch), U-Net (ResNet-34), DeepLabV3, CE+Dice, metrics, morphology/CRF, overlays |
| `p2_maskrcnn.py` | P2 step 2–4 | fine-tunes Mask R-CNN on VOC instances and scores it with the same semantic metrics |

## Extra packages (add to the `pip:` block of `environment.yml`)

```yaml
    - ultralytics      # YOLOv5/YOLOv8 trainer
    - torchmetrics     # MeanAveragePrecision
```
`scipy` is already pulled in by scikit-learn. Optional: dense CRF
`pip install git+https://github.com/lucasb-eyer/pydensecrf.git`. It often fails to compile on new Pythons.
If it does, use the morphological post-processing and say so in the report.

## Run order

```bash
cd tutorials/assignment3
python p1_explore.py
python p1_yolo.py        # writes runs/yolov5s_coco128/...
python p1_ssd.py         # needs the YOLO run above for the side-by-side table
python p2_seg.py         # ~2 GB VOC download on first run
python p2_maskrcnn.py
```
Quick smoke test first: set `EPOCHS` to 1 at the top of each file.
On Apple Silicon, if an op is missing on MPS, run `export PYTORCH_ENABLE_MPS_FALLBACK=1`.
