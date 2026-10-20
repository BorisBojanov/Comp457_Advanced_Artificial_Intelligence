"""Problem 1, Step 1: load, explore, visualise and split COCO128."""
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics.utils import YAML  # only used to read the class names
from ultralytics.utils.checks import check_yaml

ROOT = Path(__file__).resolve().parent / "data" / "coco128"
IMG_DIR, LBL_DIR = ROOT / "images" / "train2017", ROOT / "labels" / "train2017"
NAMES = YAML.load(check_yaml("coco128.yaml"))["names"]  # {0: 'person', 1: 'bicycle', ...}

if not IMG_DIR.exists():   # first run: fetch the 7 MB zip (128 images of COCO train2017 + YOLO labels)
    import urllib.request, zipfile
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    zp = ROOT.parent / "coco128.zip"
    urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", zp)
    zipfile.ZipFile(zp).extractall(ROOT.parent)


def read_labels(img_path):
    """YOLO txt format: one row per object -> class cx cy w h (all normalised 0-1)."""
    lbl = LBL_DIR / (img_path.stem + ".txt")
    if not lbl.exists():  # some COCO128 images have no objects
        return []
    return [(int(c), *map(float, xywh)) for c, *xywh in
            (line.split() for line in lbl.read_text().splitlines() if line.strip())]


images = sorted(IMG_DIR.glob("*.jpg"))
print(f"{len(images)} images, {sum(1 for _ in LBL_DIR.glob('*.txt'))} label files")

counts = Counter(c for p in images for c, *_ in read_labels(p))
print(f"{sum(counts.values())} boxes over {len(counts)} of {len(NAMES)} classes")
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
for ax, p in zip(axes.flat, random.Random(0).sample(images, 6)):
    im = Image.open(p); W, H = im.size
    ax.imshow(im); ax.axis("off"); ax.set_title(p.name)
    for c, cx, cy, w, h in read_labels(p):
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H          # centre form -> corner form, pixels
        ax.add_patch(patches.Rectangle((x1, y1), w * W, h * H, fill=False, lw=2, ec="lime"))
        ax.text(x1, y1, NAMES[c], color="black", backgroundcolor="lime", fontsize=7)
plt.tight_layout(); plt.savefig("p1_samples.png", dpi=150); plt.close()

# --- 70 / 15 / 15 split ------------------------------------------------------
random.Random(42).shuffle(images)
n_tr, n_va = int(0.70 * len(images)), int(0.15 * len(images))
splits = {"train": images[:n_tr], "val": images[n_tr:n_tr + n_va], "test": images[n_tr + n_va:]}
for name, files in splits.items():
    (ROOT / f"{name}.txt").write_text("\n".join(str(f) for f in files) + "\n")
    print(f"{name}: {len(files)} images")

# Ultralytics accepts .txt files listing image paths; labels are found by swapping images/ -> labels/
yaml_text = f"path: {ROOT}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n" + \
    "".join(f"  {k}: {v}\n" for k, v in NAMES.items())
(ROOT / "coco128_split.yaml").write_text(yaml_text)
print("wrote", ROOT / "coco128_split.yaml")
