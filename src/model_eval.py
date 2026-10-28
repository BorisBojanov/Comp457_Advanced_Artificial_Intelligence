"""
model_eval.py - training, tracking and evaluation charts for the COMP 457 assignments.

Nothing here is tied to one assignment. Training is described by a "step" (what one batch
looks like and how it is scored), tracking is a dict of per-epoch lists, and every chart is
drawn from plain dicts / arrays, so the same file serves classifiers (A1, A2), detectors and
segmenters (A3), language models and captioners (A4).

Pieces, from small to large:

    Steps           ClassificationStep   (X, y) -> class logits / sigmoid output     [default]
                    SequenceStep         token-level loss, perplexity, top-k (LM, captioner)
                    DictLossStep         torchvision detectors that return a dict of losses
    One pass        runEpoch(...)        train or evaluate one loader with any step
    Tracking        TrainingMonitor      per-epoch numbers -> live chart + PNG + JSON
                    fit(...)             loop + monitor + scheduler + best checkpoint + early stop
    Plots           plotCurves           one panel per metric (train vs val), any metric names
                    plotComparison       several runs overlaid, one panel per metric
                    plotMetricBars       final numbers of several models, one panel per metric
                    plotPerClass         per-class IoU / Dice / F1 bars for several models
                    plotEvaluation       confusion matrix + accuracy / precision / recall / F1
                    plotGroupBreakdown   accuracy per group (e.g. per target word)
    Numbers         computeMetrics, ConfusionMeter, scoresFromConfusion, measureSpeed
    Records         saveResults / loadResults, loadHistory / loadHistories, historyFromCSV,
                    resultsTable (plain or markdown, ready to paste into a report)

Three levels of use, pick whichever fits the file you are in:

1. Keep your own loop and just report numbers once per epoch. Any metric names work; the
   prefix train_ / val_ decides which line of a panel a number belongs to:

        monitor = TrainingMonitor("a3_ssd", total_epochs=EPOCHS)
        for epoch in range(EPOCHS):
            ...
            monitor.update(train_box=box, train_cls=cls, val_map50=map50)
        monitor.finish()
        plt.show()

2. Replace the loop. fit() trains, scores the validation loader every epoch, charts it live,
   and can also step a scheduler, clip gradients, keep the best checkpoint and stop early:

        history = fit(model, train_loader, val_loader, loss_fn, optimizer, device,
                      epochs=EPOCHS, run_name="a1_cnn")

        # next-word prediction: loss, perplexity, top-1 / top-5 per epoch
        history = fit(lm, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, device,
                      epochs=10, run_name="a4_lstm", step=SequenceStep(topk=(5,)),
                      scheduler=scheduler, clip=1.0, checkpoint="checkpoints/a4_lstm.pt",
                      patience=3)

        # captioning: model(features, tokens[:, :-1]) is scored against tokens[:, 1:], and a
        # METEOR number is added to the same chart each epoch through extra_eval
        step = SequenceStep(ignore_index=PAD, inputs=lambda b: (b[0], b[1][:, :-1]),
                            targets=lambda b: b[1][:, 1:])
        fit(decoder, tl, vl, criterion, optimizer, device, epochs=15, run_name="a4_cap",
            step=step, extra_eval=lambda model, epoch: {"val_meteor": meteorOnSubset(model)},
            checkpoint=..., monitor_key="val_meteor")

3. Evaluate / compare. Run a loader once, or put the finished runs side by side:

        results = evaluateAndPlot(model, test_loader, loss_fn, device, run_name="a1_cnn",
                                  class_names=["0", "1", ...])
        plotComparison(loadHistories(["a4_lstm", "a4_transformer"]), title="LSTM vs Transformer")
        plotMetricBars({"SSD": ssd_results, "YOLO": yolo_results},
                       metrics=["mAP@0.5", "precision", "recall", "ms_per_img"])
        print(resultsTable({"SSD": ssd_results, "YOLO": yolo_results}, markdown=True))

Model output conventions for the default ClassificationStep (detected from the output shape):
    [B, C] with C > 1 -> multi-class logits, loss_fn(output, y), prediction = argmax
    [B, 1] or [B]     -> binary, loss_fn(output.reshape(-1), y), prediction = output > 0.5
                         (the models in Assignment 2 end in a sigmoid; if yours returns
                         raw logits for BCEWithLogitsLoss pass from_logits=True)

Plot functions never call plt.show() - call it once at the end of main(), like the
assignments already do. Everything is saved under data/figures/<run_name>_*.png, history
under <run_name>_history.json. A finished run can be re-plotted later, without retraining,
with plotCurves(loadHistory("a1_cnn")).
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

FIGURES_DIR = Path(__file__).resolve().parent.parent / "data" / "figures"

# Chart look. Colours are the validated light-mode palette: categorical slot 1 (blue) is
# the training series, slot 2 (orange) is the test/validation series. When several runs or
# models are compared each one gets its own slot instead.
SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, GREEN, GOLD]
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Applied with plt.rc_context so importing this file never changes matplotlib for the rest
# of the assignment.
_STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": AXIS,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "legend.frameon": False,
}

_NO_WINDOW_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def _canShowWindows() -> bool:
    backend = matplotlib.get_backend().lower()
    return backend not in _NO_WINDOW_BACKENDS and "inline" not in backend


# ---------------------------------------------------------------------------------------
# Metric names: how a key like "val_map50" is titled, formatted and compared
# ---------------------------------------------------------------------------------------

# Keys that are recorded but not charted unless asked for with panels=[...]
_HIDDEN = {"lr", "time", "epoch"}
# Values in 0..1 that read better as percentages
_PERCENT = {"acc", "accuracy", "top1", "top5", "precision", "recall", "f1", "map", "map50",
            "map50_95", "map_0_5", "map_0_5_0_95", "miou", "mdice", "dice", "iou", "pixel_acc", "bleu1", "bleu2", "bleu3",
            "bleu4", "meteor", "rouge_l", "rougel"}
_LOG_AXIS = {"ppl", "perplexity", "lr"}
_TITLES = {
    "loss": "Loss", "acc": "Accuracy", "accuracy": "Accuracy", "ppl": "Perplexity",
    "top5": "Top-5 accuracy", "top1": "Top-1 accuracy", "f1": "F1", "miou": "mIoU",
    "mdice": "Mean Dice", "dice": "Dice", "iou": "IoU", "pixel_acc": "Pixel accuracy",
    "map": "mAP", "map50": "mAP@0.5", "map50_95": "mAP@0.5:0.95",
    "map_0_5": "mAP@0.5", "map_0_5_0_95": "mAP@0.5:0.95", "lr": "Learning rate",
    "rouge_l": "ROUGE-L", "meteor": "METEOR", "bleu4": "BLEU-4",
    "ms_per_img": "Inference (ms / image)", "fps": "Speed (FPS)",
}
_LOWER_BETTER_TOKENS = {"loss", "ppl", "perplexity", "error", "err", "ms", "time", "mse", "mae",
                        "wer", "cer"}
# result-dict keys -> history keys, so evaluate and training agree on names
_HISTORY_NAME = {"accuracy": "acc"}


def _tokens(name: str) -> list:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _metricName(key: str) -> str:
    """'val_map50' -> 'map50', 'train_loss' -> 'loss', 'mAP@0.5' -> 'map_0_5'."""
    for prefix in ("train_", "val_", "test_"):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return "_".join(_tokens(key))


def _isPercent(name: str) -> bool:
    return _metricName(name) in _PERCENT


def _lowerIsBetter(name: str) -> bool:
    return any(t in _LOWER_BETTER_TOKENS for t in _tokens(name))


def _title(name: str) -> str:
    base = _metricName(name)
    if base in _TITLES:
        return _TITLES[base]
    return name if any(c.isupper() for c in name) else name.replace("_", " ").capitalize()


def _fmt(name: str, value: float) -> str:
    """A number the way its chart shows it: 91.2%, 118.4, 3.2e-04, 0.4821."""
    base = _metricName(name)
    if base in _PERCENT:
        return f"{value:.1%}"
    if base == "lr":
        return f"{value:.1e}"
    if base in ("ppl", "perplexity", "fps") or "ms" in _tokens(name):
        return f"{value:.1f}"
    return f"{value:.4f}"


def _plain(name: str, value) -> str:
    """Table cell: fractions with 3 decimals, big numbers / timings with 1."""
    if not isinstance(value, (int, float, np.number)) or isinstance(value, bool):
        return str(value)
    if "ms" in _tokens(name) or "fps" in _tokens(name) or abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _bestIsMax(name: str, mode: str | None = None) -> bool:
    return mode == "max" if mode else not _lowerIsBetter(name)


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def _toDevice(obj, device):
    """Move a tensor, or any list / tuple / dict of tensors, to `device`."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_toDevice(o, device) for o in obj)
    if isinstance(obj, list):
        return [_toDevice(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _toDevice(v, device) for k, v in obj.items()}
    return obj


def _sync(device):
    """Wait for queued GPU work so a timer measures the model, not the launch."""
    kind = torch.device(device).type if not isinstance(device, int) else "cuda"
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------------------
# Steps: what one batch looks like and how it is scored
#
# A step is called as step(model, batch, loss_fn, device) and returns
#     {"loss": tensor, "n": units in the batch, "sums": {name: total over those units}}
# plus, optionally, y_true / y_pred tensors. runEpoch() averages loss and sums over n, so
# n is samples for a classifier and non-padding tokens for a language model.
# ---------------------------------------------------------------------------------------

def _predict(output, from_logits: bool):
    """Raw model output -> (integer class predictions, is_binary)."""
    if output.dim() == 1 or output.shape[-1] == 1:
        threshold = 0.0 if from_logits else 0.5
        return (output.reshape(-1) > threshold).long(), True
    return output.argmax(dim=1), False


class ClassificationStep:
    """(X, y) batches, one class label per sample. The default step; see the module docstring
    for how binary and multi-class outputs are told apart."""

    def __init__(self, from_logits: bool = False, collect: bool = False):
        self.from_logits = from_logits
        self.collect = collect
        self.num_classes = 2

    def __call__(self, model, batch, loss_fn, device):
        X, y = batch
        X, y = _toDevice(X, device), y.to(device)
        output = model(X)
        preds, binary = _predict(output, self.from_logits)
        if binary:
            loss = loss_fn(output.reshape(-1), y.float())
        else:
            loss = loss_fn(output, y)
            self.num_classes = output.shape[-1]
        out = {"loss": loss, "n": y.size(0), "sums": {"accuracy": (preds == y.long()).sum().item()}}
        if self.collect:
            out["y_true"], out["y_pred"] = y.long().cpu(), preds.cpu()
        return out

    def finalize(self, result: dict):
        result["num_classes"] = self.num_classes


class SequenceStep:
    """Token-level scoring for models whose output is [B, T, V] logits (next-word prediction,
    caption decoders). Reports loss, accuracy (top-1), top-k accuracies and perplexity.

    The default reads batches as (x, y) with y [B, T] already shifted, and calls model(x).
    For anything else say how to get the model inputs and targets from a batch:

        SequenceStep(ignore_index=PAD, inputs=lambda b: (b[0], b[1][:, :-1]),
                     targets=lambda b: b[1][:, 1:])

    Positions equal to ignore_index are left out of accuracy and of the averaging weight. The
    perplexity is exp(loss) and so follows whatever loss_fn computes (label smoothing included).
    """

    def __init__(self, ignore_index: int = -100, topk=(5,), inputs=None, targets=None,
                 perplexity: bool = True):
        self.ignore_index = ignore_index
        self.topk = tuple(topk)
        self.inputs = inputs
        self.targets = targets
        self.perplexity = perplexity

    def __call__(self, model, batch, loss_fn, device):
        inputs = self.inputs(batch) if self.inputs else (batch[0],)
        target = (self.targets(batch) if self.targets else batch[1]).to(device)
        logits = model(*_toDevice(tuple(inputs), device))
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        mask = target != self.ignore_index
        k = min(max((1, *self.topk)), logits.size(-1))
        top = logits.topk(k, dim=-1).indices
        sums = {"accuracy": ((top[..., 0] == target) & mask).sum().item()}
        for j in self.topk:
            hit = (top[..., :min(j, k)] == target.unsqueeze(-1)).any(-1)
            sums[f"top{j}"] = (hit & mask).sum().item()
        return {"loss": loss, "n": int(mask.sum().item()), "sums": sums}

    def finalize(self, result: dict):
        if self.perplexity:
            result["ppl"] = math.exp(min(result["loss"], 20))


class DictLossStep:
    """torchvision detection / instance-segmentation models: model(images, targets) returns a
    dict of losses (SSD, Faster / Mask R-CNN). Batches are (list of images, list of target
    dicts), which is what a collate_fn of `lambda b: tuple(zip(*b))` gives. Every component is
    tracked separately (train_bbox_regression, train_classification, ...) next to the total.

    The model has to stay in train mode to return losses, so validation loss is computed in
    train mode without gradients. loss_fn is not used and may be None. `target_keys` drops
    extra entries from the target dicts, e.g. ("boxes", "labels", "masks").
    """

    always_train = True

    def __init__(self, target_keys=None):
        self.target_keys = target_keys

    def __call__(self, model, batch, loss_fn, device):
        images, targets = batch
        images = [i.to(device) for i in images]
        if self.target_keys:
            targets = [{k: t[k] for k in self.target_keys} for t in targets]
        losses = model(images, _toDevice(list(targets), device))
        n = len(images)
        return {"loss": sum(losses.values()), "n": n,
                "sums": {k: v.item() * n for k, v in losses.items()}}


# ---------------------------------------------------------------------------------------
# Running a loader: one function for training an epoch and for evaluating
# ---------------------------------------------------------------------------------------

def runEpoch(model, loader, loss_fn, device, optimizer=None, from_logits: bool = False,
             collect: bool = False, step=None, clip: float | None = None, on_batch=None) -> dict:
    """One pass over `loader`. Trains when an optimizer is given, otherwise evaluates.

    `step` decides how a batch is run and scored (default ClassificationStep). `clip` clips
    the gradient norm; `on_batch` is called after every optimizer step (a per-batch LR
    scheduler goes here). Loss and the step's other numbers are averaged over samples (or
    tokens). During training they are running values (measured while the weights change),
    the same convention Keras uses for its curves. With collect=True the dict also holds
    y_true / y_pred as numpy arrays for the confusion matrix.
    """
    step = step or ClassificationStep(from_logits, collect)
    training = optimizer is not None
    model.train(True if getattr(step, "always_train", False) else training)
    total_loss, seen, sums = 0.0, 0, {}
    y_true, y_pred = [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            out = step(model, batch, loss_fn, device)
            loss = out["loss"]
            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                if on_batch is not None:
                    on_batch()

            n = out["n"]
            total_loss += loss.item() * n
            seen += n
            for key, value in out.get("sums", {}).items():
                sums[key] = sums.get(key, 0.0) + value
            if "y_true" in out:
                y_true.append(out["y_true"])
                y_pred.append(out["y_pred"])

    if seen == 0:
        raise ValueError("the loader produced no samples to score")
    result = {"loss": total_loss / seen, **{k: v / seen for k, v in sums.items()}}
    if hasattr(step, "finalize"):
        step.finalize(result)
    if y_true:
        result["y_true"] = torch.cat(y_true).numpy()
        result["y_pred"] = torch.cat(y_pred).numpy()
    return result


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------

def _safeDivide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def computeMetrics(y_true, y_pred, num_classes: int) -> dict:
    """Confusion matrix + precision / recall / F1.

    Binary (2 classes): headline numbers are for the positive class (label 1) and the
    matrix is [[tn, fp], [fn, tp]], the same layout evaluateModel() prints in Assignment 2.
    More classes: headline numbers are macro averages over classes that occur in y_true
    or y_pred (so senses that never appear do not drag the average to zero).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    tp = np.diag(cm).astype(float)
    predicted, actual = cm.sum(axis=0), cm.sum(axis=1)
    precision = _safeDivide(tp, predicted)
    recall = _safeDivide(tp, actual)
    f1 = _safeDivide(2 * precision * recall, precision + recall)

    if num_classes == 2:
        average, pick = "binary", np.array([False, True])
    else:
        average, pick = "macro", (predicted + actual) > 0

    return {
        "accuracy": tp.sum() / cm.sum(),
        "precision": precision[pick].mean(),
        "recall": recall[pick].mean(),
        "f1": f1[pick].mean(),
        "average": average,
        "confusion_matrix": cm.tolist(),
        "per_class": {"precision": precision.tolist(), "recall": recall.tolist(),
                      "f1": f1.tolist(), "support": actual.tolist()},
    }


def scoresFromConfusion(cm) -> dict:
    """Segmentation scores from a confusion matrix (rows = ground truth, columns = prediction).

    pixel_acc, mIoU and mDice average over classes that occur in the ground truth; the
    per-class lists cover every class (0 where a class never appears).
    """
    cm = torch.as_tensor(cm).double()
    tp = cm.diag()
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = tp / (tp + fp + fn).clamp(min=1)
    dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    present = cm.sum(1) > 0
    return {"pixel_acc": (tp.sum() / cm.sum().clamp(min=1)).item(),
            "mIoU": iou[present].mean().item(), "mDice": dice[present].mean().item(),
            "per_class_IoU": iou.tolist(), "per_class_Dice": dice.tolist()}


class ConfusionMeter:
    """Accumulates a confusion matrix batch by batch (pixels, tokens, anything with integer
    labels) and turns it into scores at the end. Positions equal to ignore_index are skipped.

        meter = ConfusionMeter(21, ignore_index=255)
        for img, mask in loader:
            meter.update(model(img).argmax(1), mask)
        print(meter.scores()["mIoU"])
    """

    def __init__(self, num_classes: int, ignore_index: int | None = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, pred, target):
        pred, target = torch.as_tensor(pred).cpu().long(), torch.as_tensor(target).cpu().long()
        keep = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            keep = target != self.ignore_index
        flat = target[keep] * self.num_classes + pred[keep]
        self.matrix += torch.bincount(flat, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)
        return self

    def scores(self) -> dict:
        return scoresFromConfusion(self.matrix)


def measureSpeed(fn, inputs, device="cpu", warmup: int = 3) -> dict:
    """Time fn(x) for each x in `inputs` (one image / batch at a time).

    The first `warmup` calls are dropped (kernel compilation, caches) and the device is
    synchronised around every call so asynchronous GPU work is counted. Returns
    ms_per_img (mean over the timed calls) and fps.
    """
    times = []
    for x in inputs:
        _sync(device)
        t0 = time.perf_counter()
        fn(x)
        _sync(device)
        times.append(time.perf_counter() - t0)
    timed = times[warmup:] if len(times) > warmup + 1 else times
    return {"ms_per_img": 1000 * sum(timed) / len(timed), "fps": len(timed) / sum(timed)}


def evaluateLoader(model, loader, loss_fn, device, from_logits: bool = False, step=None) -> dict:
    """Run `loader` once in eval mode and return loss + all metrics in one dict.

    With the default step that includes the confusion matrix and precision / recall / F1.
    With another step (SequenceStep, ...) it is that step's own numbers: loss, accuracy,
    perplexity, top-k.
    """
    if step is None:
        out = runEpoch(model, loader, loss_fn, device, from_logits=from_logits, collect=True)
        metrics = computeMetrics(out["y_true"], out["y_pred"], out["num_classes"])
        return {"model": type(model).__name__, "loss": out["loss"], **metrics}
    out = runEpoch(model, loader, loss_fn, device, step=step)
    return {"model": type(model).__name__, **{k: v for k, v in out.items() if k != "num_classes"}}


# ---------------------------------------------------------------------------------------
# Saving and loading what a run produced
# ---------------------------------------------------------------------------------------

def historyPath(run_name: str, out_dir: Path = FIGURES_DIR) -> Path:
    return Path(out_dir) / f"{run_name}_history.json"


def loadHistory(run_name: str, out_dir: Path = FIGURES_DIR) -> dict:
    """Reload the numbers a TrainingMonitor saved, e.g. to plot after loading model.pth."""
    return json.loads(historyPath(run_name, out_dir).read_text())


def loadHistories(run_names, out_dir: Path = FIGURES_DIR) -> dict:
    """{run_name: history} for several finished runs, ready for plotComparison()."""
    return {name: loadHistory(name, out_dir) for name in run_names}


def saveResults(results: dict, run_name: str, out_dir: Path = FIGURES_DIR) -> Path:
    """Write any results dict (numpy values and Paths are converted) to <run_name>_results.json."""
    path = Path(out_dir) / f"{run_name}_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=_jsonable))
    return path


def loadResults(run_name: str, out_dir: Path = FIGURES_DIR) -> dict:
    return json.loads((Path(out_dir) / f"{run_name}_results.json").read_text())


def historyFromCSV(path, rename: dict | None = None) -> dict:
    """Read a per-epoch CSV (Ultralytics' results.csv, or your own) into a history dict.

    Column names are cleaned so they chart like everything else: spaces and '(B)' dropped,
    '/' -> '_', lower case. Ultralytics' metrics/* columns become val_* (metrics/mAP50(B) ->
    val_map50) and its val/* losses stay val_*; its lr/* columns are dropped. Pass `rename`
    ({"old": "new"}, applied to the cleaned names) to override anything.

        plotCurves(historyFromCSV("runs/yolov5s_coco128/results.csv"), title="YOLOv5s")
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    history = {}
    for raw in (rows[0] if rows else {}):
        name = raw.strip().replace("(B)", "").replace("(M)", "")
        name = name.lower().replace("/", "_").replace("-", "_").replace("metrics_", "val_")
        if name.startswith("lr_"):
            continue
        name = (rename or {}).get(name, name)
        values = [r[raw].strip() for r in rows]
        try:
            history[name] = [float(v) for v in values]
        except ValueError:
            continue
    history.pop("epoch", None)
    return history


def bestEpoch(history: dict, key: str = "val_loss", mode: str | None = None):
    """(epoch, value) of the best entry of history[key]; epochs count from 1.

    mode is 'min' or 'max'; by default losses, perplexity and timings are minimised and
    everything else maximised."""
    values = history[key]
    index = int(np.argmax(values) if _bestIsMax(key, mode) else np.argmin(values))
    return index + 1, values[index]


# ---------------------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------------------

def _panelNames(history: dict, panels=None) -> list:
    """Metric names to draw: the explicit `panels`, else every charted key with train_ / val_
    stripped (so train_loss and val_loss share the 'loss' panel). Loss first, accuracy next."""
    if panels:
        return list(panels)
    names = []
    for key, values in history.items():
        if key in _HIDDEN or not isinstance(values, list) or not values:
            continue
        name = key
        for prefix in ("train_", "val_"):
            if key.startswith(prefix):
                name = key[len(prefix):]
        if name not in names and name not in _HIDDEN:
            names.append(name)
    return sorted(names, key=lambda n: (n != "loss", n not in ("acc", "accuracy")))


def _seriesOf(history: dict, name: str) -> dict:
    """The lines a panel holds: {'train': [...], 'val': [...], 'other': [...]}, only those present."""
    found = {}
    for split, key in (("train", f"train_{name}"), ("val", f"val_{name}"), ("other", name)):
        values = history.get(key)
        if isinstance(values, list) and values:
            found[split] = values
    return found


def _gridShape(count: int, max_cols: int = 3):
    cols = min(count, max_cols)
    return math.ceil(count / cols), cols


def _formatPanel(ax, name: str, last_epoch: int, values_for_limits=None):
    ax.set_title(_title(name), loc="left", color=INK_2, fontsize=11)
    ax.set_xlabel("epoch")
    if last_epoch <= 12:
        ax.set_xticks(range(1, last_epoch + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0.5, last_epoch + 0.5)
    base = _metricName(name)
    if base in _LOG_AXIS:
        ax.set_yscale("log")
    elif base in _PERCENT:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(top=min(ax.get_ylim()[1], 1.01))
    else:
        ax.set_ylim(bottom=max(ax.get_ylim()[0], 0))
    # the grid is only for reading values off the y axis
    ax.grid(axis="x", visible=False)


def plotCurves(history: dict, title: str = "", val_label: str = "test", fig=None,
               total_epochs: int | None = None, panels=None):
    """One panel per metric, training vs validation, from a dict of per-epoch lists.

    Keys follow the pattern train_<metric> / val_<metric> (a bare <metric> key is drawn as a
    single line): train_loss, val_acc, val_ppl, val_map50, ... Panels appear for every
    metric found, loss first. lr, time and epoch are recorded but not drawn unless named in
    `panels` (e.g. panels=["loss", "lr"]). Pass `fig` to redraw into an existing figure (that
    is how the live window updates) and `total_epochs` to fix the x axis so the curve fills
    in left to right. Returns the figure.
    """
    names = [n for n in _panelNames(history, panels) if _seriesOf(history, n)]
    if not names:
        raise ValueError("history has no train/val values to plot")
    rows, cols = _gridShape(len(names))
    splits = {"train": ("train", BLUE), "val": (val_label, ORANGE), "other": ("", BLUE)}

    with plt.rc_context(_STYLE):  # type: ignore
        if fig is None:
            fig = plt.figure(layout="constrained")
        fig.set_size_inches(5.4 * cols, 4.2 * rows)
        fig.clf()
        axes = fig.subplots(rows, cols, squeeze=False).ravel()

        for ax, name in zip(axes, names):
            series = _seriesOf(history, name)
            for split, values in series.items():
                label, colour = splits[split]
                if split == "val" and "train" not in series and "other" not in series:
                    label = label or "val"
                shown = _fmt(name, values[-1])
                ax.plot(range(1, len(values) + 1), values, color=colour, linewidth=1.6,
                        marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        clip_on=False, label=f"{label}  {shown}".strip())
            last = max(max(len(v) for v in series.values()), total_epochs or 0)
            _formatPanel(ax, name, last)
            ax.legend(loc="best", fontsize=9)
        for ax in axes[len(names):]:
            ax.set_visible(False)

        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    return fig


def plotComparison(histories: dict, panels=None, title: str = "", run_name: str | None = None,
                   val_label: str = "val", out_dir: Path = FIGURES_DIR, save: bool = True):
    """Several runs on the same panels: one colour per run, validation solid, training dashed.

    `histories` is {name: history}, from loadHistories([...]) or the dicts fit() returned.
    The legend shows each run's final validation value. Four runs fit the validated colour
    set; beyond that colours repeat and the marker shape changes. Saved to <run_name>_comparison.png
    when run_name is given.
    """
    names = [n for n in _panelNames({k: v for h in histories.values() for k, v in h.items()}, panels)
             if any(_seriesOf(h, n) for h in histories.values())]
    if not names:
        raise ValueError("none of the histories has values to plot")
    rows, cols = _gridShape(len(names))
    markers = "osD^v"

    with plt.rc_context(_STYLE):  # type: ignore
        fig = plt.figure(figsize=(5.4 * cols, 4.2 * rows), layout="constrained")
        axes = fig.subplots(rows, cols, squeeze=False).ravel()
        for ax, name in zip(axes, names):
            longest = 1
            for i, (run, history) in enumerate(histories.items()):
                series = _seriesOf(history, name)
                colour, marker = SERIES[i % len(SERIES)], markers[(i // len(SERIES)) % len(markers)]
                if "train" in series:
                    ax.plot(range(1, len(series["train"]) + 1), series["train"], color=colour,
                            linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.75)
                    longest = max(longest, len(series["train"]))
                main = series.get("val") or series.get("other")
                if main:
                    ax.plot(range(1, len(main) + 1), main, color=colour, linewidth=1.8,
                            marker=marker, markersize=6, markeredgecolor=SURFACE,
                            markeredgewidth=1.5, clip_on=False,
                            label=f"{run}  {_fmt(name, main[-1])}")
                    longest = max(longest, len(main))
            _formatPanel(ax, name, longest)
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([], [], color=INK_2, linewidth=1.2, linestyle=(0, (4, 3))))
            labels.append("training")
            ax.legend(handles, labels, loc="best", fontsize=9)
        for ax in axes[len(names):]:
            ax.set_visible(False)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=13)

    if save and run_name:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{run_name}_comparison.png", dpi=150)
    return fig


class TrainingMonitor:
    """Collects per-epoch numbers and keeps a chart up to date.

    update() takes any metrics by keyword, with the train_ / val_ prefix deciding the line
    (see plotCurves). It redraws the window (when the matplotlib backend can show one) and
    always rewrites <run_name>_curves.png and <run_name>_history.json, so an interrupted or
    headless run still leaves its curves behind. Every key must be reported on every epoch
    so the lists stay the same length. Call finish() after the last epoch; it turns
    interactive mode back off so a later plt.show() blocks and keeps the window open.
    """

    def __init__(self, run_name: str = "run", total_epochs: int | None = None,
                 val_label: str = "test", live: bool = True, out_dir: Path = FIGURES_DIR,
                 panels=None):
        self.run_name = run_name
        self.total_epochs = total_epochs
        self.val_label = val_label
        self.out_dir = Path(out_dir)
        self.panels = panels
        self.history = {}
        self.fig = None
        self.live = live and _canShowWindows()
        self._was_interactive = plt.isinteractive()
        if self.live:
            plt.ion()

    def update(self, **metrics):
        """Log one epoch, e.g. update(train_loss=0.31, train_acc=0.91, val_loss=0.28, val_acc=0.92)."""
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(float(value))

        epoch = max(len(v) for v in self.history.values())
        of = f" of {self.total_epochs}" if self.total_epochs else ""
        self.fig = plotCurves(self.history, f"{self.run_name} - epoch {epoch}{of}",
                              self.val_label, self.fig, self.total_epochs, self.panels)
        self._save()
        if self.live:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.05)

    def _save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_dir / f"{self.run_name}_curves.png", dpi=150)  # type: ignore
        historyPath(self.run_name, self.out_dir).write_text(json.dumps(self.history, indent=2))

    def best(self, key: str = "val_loss", mode: str | None = None):
        """(epoch, value) of the best epoch so far for `key`."""
        return bestEpoch(self.history, key, mode)

    def finish(self) -> dict:
        if self.fig is not None:
            self._save()
        if self.live and not self._was_interactive:
            plt.ioff()
        return self.history


def _epochLine(epoch, epochs, row, val_label):
    groups = {"train": [], "val": [], "other": []}
    for key, value in row.items():
        if key in _HIDDEN:
            continue
        split = "train" if key.startswith("train_") else "val" if key.startswith("val_") else "other"
        name = key.split("_", 1)[1] if split != "other" else key
        groups[split].append(f"{name.replace('_', ' ')} {_fmt(name, value)}")
    parts = [f"{label} " + "  ".join(items) for label, items in
             (("train", groups["train"]), (val_label, groups["val"]), ("", groups["other"])) if items]
    return f"Epoch {epoch}/{epochs}  " + "  |  ".join(p.strip() for p in parts) + f"  [{row['time']:.0f}s]"


def _rowFrom(prefix: str, result: dict) -> dict:
    return {f"{prefix}_{_HISTORY_NAME.get(k, k)}": float(v) for k, v in result.items()
            if k != "num_classes" and isinstance(v, (int, float, np.number))}


def saveCheckpoint(path, model, /, **meta) -> Path:
    """torch.save({'state_dict': ..., **meta}). Extra keyword arguments (vocab, args, epoch)
    ride along so a checkpoint can be rebuilt without the training script."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)
    return path


def loadCheckpoint(path, model=None, device="cpu") -> dict:
    """Load a saveCheckpoint() file; if `model` is given its weights are restored. Returns
    the whole dict so the metadata is available."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs: int,
        run_name: str = "run", val_label: str = "test", from_logits: bool = False,
        live: bool = True, out_dir: Path = FIGURES_DIR, *, step=None, scheduler=None,
        scheduler_per_batch: bool = False, clip: float | None = None, extra_eval=None,
        checkpoint=None, checkpoint_meta: dict | None = None, monitor_key: str = "val_loss",
        mode: str | None = None, patience: int | None = None,
        restore_best: bool | None = None, panels=None) -> dict:
    """Train for `epochs`, score `val_loader` after each one and chart it live.

    val_loader may be None to chart training numbers only (monitor_key then falls back to
    train_loss). Returns the history dict. Everything after out_dir is keyword-only:

        step            how a batch is run and scored; default ClassificationStep(from_logits)
        scheduler       stepped once per epoch (ReduceLROnPlateau gets monitor_key's value), or
                        once per optimizer step with scheduler_per_batch=True (OneCycleLR, ...)
        clip            gradient-norm clipping
        extra_eval      extra_eval(model, epoch) -> {"val_meteor": 0.21, ...}; those numbers are
                        added to the history and the chart (use the train_/val_ prefixes)
        checkpoint      path; the model is saved whenever monitor_key improves
        checkpoint_meta dict stored in the checkpoint beside the weights (vocab, args, ...)
        monitor_key     the history key that decides "best"; mode 'min'/'max' overrides the
                        guess (losses, perplexity and timings are minimised)
        patience        stop after this many epochs without improvement
        restore_best    reload the best weights at the end (default: when checkpoint is set)
        panels          which metrics to chart, see plotCurves

    lr and time per epoch are recorded in the history too.
    """
    step = step or ClassificationStep(from_logits)
    if val_loader is None and monitor_key == "val_loss":
        monitor_key = "train_loss"
    if restore_best is None:
        restore_best = checkpoint is not None
    per_batch = scheduler.step if scheduler is not None and scheduler_per_batch else None
    plateau = scheduler is not None and isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    monitor = TrainingMonitor(run_name, epochs, val_label, live, out_dir, panels)
    best_value, best_epoch, stale = None, 0, 0
    try:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            row = _rowFrom("train", runEpoch(model, train_loader, loss_fn, device, optimizer,
                                             step=step, clip=clip, on_batch=per_batch))
            if val_loader is not None:
                row.update(_rowFrom("val", runEpoch(model, val_loader, loss_fn, device, step=step)))
            if extra_eval is not None:
                row.update({k: float(v) for k, v in extra_eval(model, epoch).items()})

            if scheduler is not None and not scheduler_per_batch:
                scheduler.step(row[monitor_key]) if plateau else scheduler.step()
            row["lr"] = lr
            row["time"] = time.perf_counter() - started
            print(_epochLine(epoch, epochs, row, val_label))
            monitor.update(**row)

            if checkpoint is not None or patience is not None:
                if monitor_key not in row:
                    raise ValueError(f"monitor_key '{monitor_key}' is not in the history; "
                                     f"available: {sorted(row)}")
                value = row[monitor_key]
                higher = _bestIsMax(monitor_key, mode)
                improved = best_value is None or (value > best_value if higher else value < best_value)
                if improved:
                    best_value, best_epoch, stale = value, epoch, 0
                    if checkpoint is not None:
                        saveCheckpoint(checkpoint, model, epoch=epoch, monitor_key=monitor_key,
                                       value=value, run_name=run_name, **(checkpoint_meta or {}))
                else:
                    stale += 1
                    if patience is not None and stale >= patience:
                        print(f"Early stop: {monitor_key} has not improved for {stale} epochs "
                              f"(best {best_value:.4f} at epoch {best_epoch})")
                        break
    finally:
        monitor.finish()

    if best_value is not None:
        print(f"Best {monitor_key} {best_value:.4f} at epoch {best_epoch}")
        if restore_best and checkpoint is not None:
            loadCheckpoint(checkpoint, model, device)
    return monitor.history


# ---------------------------------------------------------------------------------------
# Comparing finished models: tables and bar charts
# ---------------------------------------------------------------------------------------

def _scalarMetrics(results: dict, metrics=None) -> list:
    """Metric names to show: the given ones, else numeric entries present for every model."""
    if metrics:
        return list(metrics)
    shared = None
    for entry in results.values():
        keys = [k for k, v in entry.items()
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)]
        shared = keys if shared is None else [k for k in shared if k in keys]
    return shared or []


def resultsTable(results: dict, metrics=None, markdown: bool = False, label: str = "model") -> str:
    """{model: {metric: value}} -> an aligned text table (or a markdown one for the report).

    Prints nothing; print() the return value. The best value in each column is marked with *
    (plain) or bold (markdown): the largest, or the smallest for losses and timings.
    """
    metrics = _scalarMetrics(results, metrics)
    names = list(results)
    best = {}
    for m in metrics:
        values = [results[n][m] for n in names if m in results[n]]
        if len(values) > 1:
            best[m] = max(values) if _bestIsMax(m) else min(values)

    def cell(n, m):
        if m not in results[n]:
            return "-"
        text = _plain(m, results[n][m])
        if best.get(m) == results[n][m]:
            return f"**{text}**" if markdown else f"{text}*"
        return text

    if markdown:
        lines = [f"| {label} | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
        lines += [f"| {n} | " + " | ".join(cell(n, m) for m in metrics) + " |" for n in names]
        return "\n".join(lines)

    width = max([len(label)] + [len(n) for n in names]) + 2
    cols = [max(len(m), *(len(cell(n, m)) for n in names)) + 2 for m in metrics]
    lines = [f"{label:<{width}}" + "".join(f"{m:>{w}}" for m, w in zip(metrics, cols))]
    lines += [f"{n:<{width}}" + "".join(f"{cell(n, m):>{w}}" for m, w in zip(metrics, cols))
              for n in names]
    return "\n".join(lines)


def plotMetricBars(results: dict, metrics=None, title: str = "", run_name: str | None = None,
                   out_dir: Path = FIGURES_DIR, save: bool = True):
    """Final numbers of several models: one small panel per metric, one bar per model.

    `results` is {model: {metric: value}}. Panels have their own scale, so fractions (mAP,
    mIoU) and timings (ms / image) can sit side by side. Fractions are shown as percentages;
    the best bar's label is bold (the smallest for losses and timings). Model colours are
    the same in every panel. Saved to <run_name>_models.png when run_name is given.
    """
    metrics = _scalarMetrics(results, metrics)
    if not metrics:
        raise ValueError("no numeric metric is shared by all models")
    models = list(results)
    rows, cols = _gridShape(len(metrics), 4)
    ypos = np.arange(len(models))[::-1]

    with plt.rc_context(_STYLE):  # type: ignore
        fig = plt.figure(figsize=(3.9 * cols, max(2.4, 0.55 * len(models) + 1.3) * rows),
                         layout="constrained")
        axes = fig.subplots(rows, cols, squeeze=False).ravel()
        for ax, metric in zip(axes, metrics):
            values = [float(results[m].get(metric, np.nan)) for m in models]
            finite = [v for v in values if not np.isnan(v)]
            best = (max(finite) if _bestIsMax(metric) else min(finite)) if len(finite) > 1 else None
            ax.barh(ypos, np.nan_to_num(values), height=0.55,
                    color=[SERIES[i % len(SERIES)] for i in range(len(models))])
            top = max(finite) if finite else 1.0
            for y, value in zip(ypos, values):
                if np.isnan(value):
                    continue
                ax.text(value + 0.015 * max(top, 1e-9), y, _fmt(metric, value) if _isPercent(metric)
                        else _plain(metric, value), va="center", ha="left", fontsize=9, color=INK,
                        fontweight="bold" if value == best else "normal")
            ax.set_yticks(ypos, models if ax in axes[::cols] else [""] * len(models))
            ax.set_xlim(0, 1.15 if _isPercent(metric) else max(top, 1e-9) * 1.25)
            if _isPercent(metric):
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            note = "  (lower is better)" if _lowerIsBetter(metric) else ""
            ax.set_title(_title(metric) + note, loc="left", color=INK_2, fontsize=11)
        for ax in axes[len(metrics):]:
            ax.set_visible(False)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=13)

    if save and run_name:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{run_name}_models.png", dpi=150)
    return fig


def plotPerClass(values: dict, class_names, metric: str = "IoU", title: str = "",
                 run_name: str | None = None, out_dir: Path = FIGURES_DIR, save: bool = True):
    """Grouped bars, one group per class and one bar per model: per-class IoU, Dice, F1, AP...

    `values` is {model: [one number per class]} (for example the per_class_IoU list from
    scoresFromConfusion). The legend carries each model's mean over classes. Saved to
    <run_name>_per_class_<metric>.png when run_name is given.
    """
    models = list(values)
    classes = list(class_names)
    x = np.arange(len(classes))
    width = 0.8 / len(models)

    with plt.rc_context(_STYLE):  # type: ignore
        fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(classes)), 4.6), layout="constrained")
        for i, model in enumerate(models):
            numbers = list(values[model])
            ax.bar(x + (i - (len(models) - 1) / 2) * width, numbers, width=width * 0.92,
                   color=SERIES[i % len(SERIES)], zorder=3, linewidth=0,
                   label=f"{model}  (mean {np.mean(numbers):.3f})")
        ax.set_xticks(x, classes, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
        ax.legend(loc="upper center", ncol=min(len(models), 3), fontsize=9, bbox_to_anchor=(0.5, 1.14))
        if title:
            ax.set_title(title, loc="left", color=INK_2, fontsize=11, pad=34)

    if save and run_name:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{run_name}_per_class_{_metricName(metric)}.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------------------
# Evaluation charts
# ---------------------------------------------------------------------------------------

def _drawConfusion(fig, ax, cm, names):
    n = cm.shape[0]
    row_totals = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row_totals, out=np.zeros(cm.shape), where=row_totals > 0)

    image = ax.imshow(share, cmap=BLUE_RAMP, vmin=0, vmax=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    if n <= 30:
        ax.set_xticks(range(n), names, rotation=45 if n > 10 else 0, ha="right" if n > 10 else "center")
        ax.set_yticks(range(n), names)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if n <= 15:
        # thin surface-coloured gaps between cells, counts printed in each cell
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10 if n <= 6 else 8,
                        color="white" if share[i, j] > 0.5 else INK)

    ax.tick_params(length=0)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix", loc="left", color=INK_2, fontsize=11)
    bar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.03)
    bar.set_label("share of actual class")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def _drawMetricBars(ax, results, average, names):
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [results[k] for k in ("accuracy", "precision", "recall", "f1")]
    ypos = np.arange(len(labels))[::-1]

    ax.barh(ypos, values, height=0.22, color=BLUE)
    for y, value in zip(ypos, values):
        ax.text(value + 0.015, y, f"{value:.1%}", va="center", ha="left", fontsize=10, color=INK)

    ax.set_yticks(ypos, labels)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    note = f"positive class = {names[1]}" if average == "binary" else "macro average"
    ax.set_title(f"Metrics ({note})", loc="left", color=INK_2, fontsize=11)


def plotEvaluation(results: dict, class_names=None, run_name: str = "run",
                   split_label: str = "test", out_dir: Path = FIGURES_DIR, save: bool = True):
    """Confusion matrix + accuracy / precision / recall / F1 bars for one evaluated model.

    `results` is what evaluateLoader() returns. A dict from your own evaluateModel() works
    too as long as it has accuracy, precision, recall, f1 and confusion_matrix.
    """
    cm = np.asarray(results["confusion_matrix"])
    n = cm.shape[0]
    names = list(class_names) if class_names is not None else [str(i) for i in range(n)]
    if len(names) != n:
        raise ValueError(f"class_names has {len(names)} entries but the confusion matrix has {n} classes")
    average = results.get("average") or ("binary" if n == 2 else "macro")

    with plt.rc_context(_STYLE):  # type: ignore
        fig, (ax_cm, ax_bar) = plt.subplots(
            1, 2, figsize=(11.5, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1]})
        _drawConfusion(fig, ax_cm, cm, names)
        _drawMetricBars(ax_bar, results, average, names)

        title = f"{run_name} - {split_label} set"
        if results.get("model") and results["model"] not in run_name:
            title = f"{run_name} ({results['model']}) - {split_label} set"
        if "loss" in results:
            title += f"  |  loss {results['loss']:.4f}"
        fig.suptitle(title, x=0.01, ha="left", fontsize=13)

    if save:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{run_name}_evaluation.png", dpi=150)
    return fig


def evaluateAndPlot(model, loader, loss_fn, device, run_name: str = "run", class_names=None,
                    split_label: str = "test", from_logits: bool = False,
                    out_dir: Path = FIGURES_DIR, step=None) -> dict:
    """Evaluate `model` on `loader`, print the numbers, draw and save the charts.

    With a classifier that is the confusion matrix and metric bars. With a custom `step`
    (SequenceStep, ...) there is no confusion matrix: the numbers are printed and saved to
    <run_name>_results.json instead.
    """
    results = evaluateLoader(model, loader, loss_fn, device, from_logits, step)
    if "confusion_matrix" not in results:
        shown = "  ".join(f"{k} {_fmt(k, v)}" for k, v in results.items()
                          if isinstance(v, (int, float, np.number)))
        print(f"{run_name} {split_label}: {shown}")
        saveResults(results, run_name, out_dir)
        return results
    print(f"{run_name} {split_label}: loss {results['loss']:.4f}  acc {results['accuracy']:.4f}  "
          f"precision {results['precision']:.4f}  recall {results['recall']:.4f}  "
          f"F1 {results['f1']:.4f}  ({results['average']})")
    plotEvaluation(results, class_names, run_name, split_label, out_dir)
    return results


# ---------------------------------------------------------------------------------------
# Per-group breakdown (e.g. one row per target word in word sense disambiguation)
#
# With hundreds of classes the full confusion matrix is unreadable (labels are hidden past
# 30 classes). Splitting the test set by a group instead - the target word - gives one small,
# readable result per word: accuracy, macro-F1 and, for the words you care about, a confusion
# matrix over just that word's own senses.
# ---------------------------------------------------------------------------------------

def collectPredictions(model, loader, device, group_of=None, from_logits: bool = False) -> dict:
    """Run `loader` once in eval mode and return numpy y_true / y_pred (and groups).

    group_of(X) -> LongTensor of one group id per row of the batch. For the WSD LSTM the
    target word's index is the last column of X, so group_of=lambda X: X[:, -1].
    """
    model.eval()
    y_true, y_pred, groups = [], [], []
    with torch.no_grad():
        for X, y in loader:
            if group_of is not None:
                groups.append(group_of(X).long().cpu())
            preds, _ = _predict(model(X.to(device)), from_logits)
            y_true.append(y.long().cpu())
            y_pred.append(preds.cpu())

    out = {"y_true": torch.cat(y_true).numpy(), "y_pred": torch.cat(y_pred).numpy()}
    if group_of is not None:
        out["groups"] = torch.cat(groups).numpy()
    return out


def _macroF1(y_true, y_pred) -> float:
    """Macro-F1 over the classes that occur in y_true or y_pred (same rule as computeMetrics)."""
    f1s = []
    for c in np.union1d(y_true, y_pred):
        tp = float(((y_true == c) & (y_pred == c)).sum())
        predicted, actual = float((y_pred == c).sum()), float((y_true == c).sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s))


def groupBreakdown(y_true, y_pred, groups, group_names, baseline_pred=None) -> list:
    """One dict per group: support, #senses, accuracy, macro-F1 (+ baseline accuracy).

    `baseline_pred` is an optional second set of predictions (e.g. always the most frequent
    sense) scored on the same rows, so every group shows how much the model adds.
    """
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    rows = []
    for g, name in enumerate(group_names):
        pick = groups == g
        n = int(pick.sum())
        if n == 0:
            continue
        row = {"group": name, "n": n, "classes": int(len(np.unique(y_true[pick]))),
               "accuracy": float((y_true[pick] == y_pred[pick]).mean()),
               "f1": _macroF1(y_true[pick], y_pred[pick])}
        if baseline_pred is not None:
            row["baseline_accuracy"] = float((y_true[pick] == np.asarray(baseline_pred)[pick]).mean())
        rows.append(row)
    return rows


def printGroupTable(rows: list, label: str = "word"):
    has_base = any("baseline_accuracy" in r for r in rows)
    head = f"{label:<12}{'n':>5}{'senses':>8}{'acc':>8}{'macro-F1':>10}" + (f"{'baseline':>10}" if has_base else "")
    print(head)
    for r in rows:
        line = f"{r['group']:<12}{r['n']:>5}{r['classes']:>8}{r['accuracy']:>8.3f}{r['f1']:>10.3f}"
        if has_base:
            line += f"{r.get('baseline_accuracy', float('nan')):>10.3f}"
        print(line)


def plotGroupBreakdown(rows: list, run_name: str = "run", split_label: str = "test", label: str = "word",
                       focus=None, out_dir: Path = FIGURES_DIR, save: bool = True):
    """Horizontal bars: accuracy per group, with the baseline beside it when there is one.

    `focus` groups (e.g. the three required target words) come first and in bold, the rest
    follow ordered by how many test examples they have. n is printed in the row label because
    a group with 5 examples says much less than one with 100.
    """
    focus = [f for f in (focus or []) if any(r["group"] == f for r in rows)]
    first = [next(r for r in rows if r["group"] == f) for f in focus]
    rest = sorted((r for r in rows if r["group"] not in focus), key=lambda r: -r["n"])
    ordered = first + rest
    has_base = any("baseline_accuracy" in r for r in ordered)

    ypos = np.arange(len(ordered))[::-1]
    h = 0.34 if has_base else 0.5
    with plt.rc_context(_STYLE):  # type: ignore
        fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.5 * len(ordered) + 1.6)), layout="constrained")
        offset = h / 2 + 0.02 if has_base else 0.0
        ax.barh(ypos + offset, [r["accuracy"] for r in ordered], height=h, color=BLUE, label="model")
        if has_base:
            ax.barh(ypos - offset, [r.get("baseline_accuracy", 0.0) for r in ordered], height=h,
                    color=AXIS, label="most-frequent-sense baseline")

        for y, r in zip(ypos, ordered):
            ax.text(r["accuracy"] + 0.012, y + offset, f"{r['accuracy']:.0%}", va="center", ha="left",
                    fontsize=9, color=INK)
            if has_base:
                ax.text(r.get("baseline_accuracy", 0.0) + 0.012, y - offset, f"{r.get('baseline_accuracy', 0.0):.0%}",
                        va="center", ha="left", fontsize=9, color=INK_2)

        ax.set_yticks(ypos, [f"{r['group']}  (n={r['n']})" for r in ordered])
        for tick, r in zip(ax.get_yticklabels(), ordered):
            if r["group"] in focus:
                tick.set_fontweight("bold")
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("accuracy")
        ax.set_title(f"{run_name}: accuracy per {label} ({split_label} set)", loc="left", color=INK_2, fontsize=11)
        if has_base:
            ax.legend(loc="lower right", fontsize=9)

    if save:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{run_name}_by_{label}.png", dpi=150)
    return fig


def _shortName(name: str, width: int = 16) -> str:
    return name if len(name) <= width else name[:width - 6] + "…" + name[-5:]


def plotGroupConfusions(y_true, y_pred, groups, group_names, class_names, focus,
                        run_name: str = "run", split_label: str = "test", label: str = "word",
                        out_dir: Path = FIGURES_DIR, save: bool = True):
    """One small confusion matrix per focus group, over only the classes that group uses.

    Rows / columns are the classes that appear in that group's y_true or y_pred, so a word with
    6 senses gets a 6x6 matrix instead of a slice of a 550x550 one.
    """
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    focus = [f for f in focus if f in group_names]
    if not focus:
        return None

    with plt.rc_context(_STYLE):  # type: ignore
        fig, axes = plt.subplots(1, len(focus), figsize=(6.2 * len(focus), 5.4), layout="constrained",
                                 squeeze=False)
        for ax, name in zip(axes[0], focus):
            pick = groups == group_names.index(name)
            classes = np.union1d(y_true[pick], y_pred[pick])
            index = {int(c): i for i, c in enumerate(classes)}
            cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
            for t, p in zip(y_true[pick], y_pred[pick]):
                cm[index[int(t)], index[int(p)]] += 1
            names = [_shortName(class_names[int(c)]) if class_names is not None else str(int(c)) for c in classes]
            _drawConfusion(fig, ax, cm, names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")     # sense names are long: never side by side
            ax.set_title(f"{name}  (n={int(pick.sum())}, accuracy {np.trace(cm) / cm.sum():.0%})",
                         loc="left", color=INK_2, fontsize=11)
        fig.suptitle(f"{run_name} - {split_label} set, per-{label} confusion", x=0.01, ha="left", fontsize=13)

    if save:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{run_name}_{label}_confusions.png", dpi=150)
    return fig


def evaluatePerGroup(y_true, y_pred, groups, group_names, class_names=None, baseline_pred=None,
                     focus=None, run_name: str = "run", split_label: str = "test", label: str = "word",
                     out_dir: Path = FIGURES_DIR) -> list:
    """Per-group table + accuracy bars + small confusion matrices for the focus groups.

    Works from plain arrays, so it fits any model: use collectPredictions() for a DataLoader
    model, or pass predictions you computed yourself (e.g. the BERT sense ranking).
    Returns the per-group rows. Figures are saved, never shown - call plt.show() once at the end.
    """
    rows = groupBreakdown(y_true, y_pred, groups, group_names, baseline_pred)
    print(f"\n{run_name}: per-{label} results ({split_label} set)")
    printGroupTable(rows, label)
    plotGroupBreakdown(rows, run_name, split_label, label, focus, out_dir)
    if focus:
        plotGroupConfusions(y_true, y_pred, groups, list(group_names), class_names, list(focus),
                            run_name, split_label, label, out_dir)
    return rows


# ---------------------------------------------------------------------------------------
# Demo: python src/model_eval.py   (synthetic data, shows what the charts look like)
# ---------------------------------------------------------------------------------------

def _demo():
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    classes = ["red", "green", "blue", "yellow"]
    centres = torch.randn(len(classes), 2) * 1.6          # overlapping blobs so errors happen

    def blobs(n):
        y = torch.randint(0, len(classes), (n,))
        return TensorDataset(centres[y] + torch.randn(n, 2), y)

    train_loader = DataLoader(blobs(2000), batch_size=64, shuffle=True)
    test_loader = DataLoader(blobs(600), batch_size=64)

    device = "cpu"
    loss_fn = nn.CrossEntropyLoss()
    histories, results = {}, {}
    for name, hidden in (("small", 4), ("wide", 32)):
        model = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, len(classes))).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
        histories[name] = fit(model, train_loader, test_loader, loss_fn, optimizer, device, epochs=8,
                              run_name=f"demo_{name}", scheduler=scheduler, patience=4,
                              checkpoint=FIGURES_DIR / f"demo_{name}.pt")
        results[name] = evaluateAndPlot(model, test_loader, loss_fn, device, run_name=f"demo_{name}",
                                        class_names=classes)

    plotComparison(histories, title="demo: small vs wide", run_name="demo")
    plotMetricBars(results, metrics=["accuracy", "precision", "recall", "f1"], run_name="demo")
    print(resultsTable(results, ["accuracy", "precision", "recall", "f1"]))
    plt.show()


if __name__ == "__main__":
    _demo()
