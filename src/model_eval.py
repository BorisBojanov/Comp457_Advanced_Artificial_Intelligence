"""
model_eval.py - training curves and evaluation charts for the COMP 457 assignments.

Works with any of the assignment layouts because it only needs a model, a DataLoader
and a loss function. Three levels of use, pick whichever fits the file you are in:

1. Drop into an existing loop (Assignment 1 / 2 style). You keep your own loop and just
   report numbers once per epoch. The window redraws live and a PNG is saved every epoch:

        monitor = TrainingMonitor("a2_rnn", total_epochs=EPOCHS)
        for epoch in range(EPOCHS):
            ...                                     # your training + test code
            monitor.update(train_loss=train_loss, train_acc=train_acc,
                           val_loss=test_loss, val_acc=test_acc)
        monitor.finish()
        plt.show()

2. Replace the loop. fit() trains, scores the test loader every epoch and feeds the monitor:

        history = fit(model, train_loader, test_loader, loss_fn, optimizer, device,
                      epochs=EPOCHS, run_name="a1_cnn")

3. Evaluate. Runs the loader once, prints metrics and draws a confusion matrix plus a
   metrics bar chart. Also works with the dict your own evaluateModel() returns:

        results = evaluateAndPlot(model, test_loader, loss_fn, device,
                                  run_name="a1_cnn", class_names=["0", "1", ...])
        plotEvaluation(myEvaluateModelDict, class_names=["neg", "pos"], run_name="a2_rnn")

Model output conventions (detected from the output shape, no flag needed):
    [B, C] with C > 1 -> multi-class logits, loss_fn(output, y), prediction = argmax
    [B, 1] or [B]     -> binary, loss_fn(output.reshape(-1), y), prediction = output > 0.5
                         (the models in Assignment 2 end in a sigmoid; if yours returns
                         raw logits for BCEWithLogitsLoss pass from_logits=True)

Plot functions never call plt.show() - call it once at the end of main(), like the
assignments already do. Everything is saved under data/figures/<run_name>_*.png.
A finished run can be re-plotted later, without retraining, with
plotCurves(loadHistory("a1_cnn")).
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

FIGURES_DIR = Path(__file__).resolve().parent.parent / "data" / "figures"

# Chart look. Colours are the validated light-mode palette: categorical slot 1 (blue) is
# the training series, slot 2 (orange) is the test/validation series.
SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
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
# Running a loader: one function for training an epoch and for evaluating
# ---------------------------------------------------------------------------------------

def _predict(output, from_logits: bool):
    """Raw model output -> (integer class predictions, is_binary)."""
    if output.dim() == 1 or output.shape[-1] == 1:
        threshold = 0.0 if from_logits else 0.5
        return (output.reshape(-1) > threshold).long(), True
    return output.argmax(dim=1), False


def runEpoch(model, loader, loss_fn, device, optimizer=None, from_logits: bool = False,
             collect: bool = False) -> dict:
    """One pass over `loader`. Trains when an optimizer is given, otherwise evaluates.

    Loss and accuracy are averaged over samples. During training they are running values
    (measured while the weights change), the same convention Keras uses for its curves.
    With collect=True the dict also holds y_true / y_pred as numpy arrays for the
    confusion matrix.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen, num_classes = 0.0, 0, 0, 2
    y_true, y_pred = [], []

    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            preds, binary = _predict(output, from_logits)
            if binary:
                loss = loss_fn(output.reshape(-1), y.float())
            else:
                loss = loss_fn(output, y)
                num_classes = output.shape[-1]

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = y.size(0)
            total_loss += loss.item() * n
            correct += (preds == y.long()).sum().item()
            seen += n
            if collect:
                y_true.append(y.long().cpu())
                y_pred.append(preds.cpu())

    result = {"loss": total_loss / seen, "accuracy": correct / seen, "num_classes": num_classes}
    if collect:
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


def evaluateLoader(model, loader, loss_fn, device, from_logits: bool = False) -> dict:
    """Run `loader` once in eval mode and return loss + all metrics in one dict."""
    out = runEpoch(model, loader, loss_fn, device, from_logits=from_logits, collect=True)
    metrics = computeMetrics(out["y_true"], out["y_pred"], out["num_classes"])
    return {"model": type(model).__name__, "loss": out["loss"], **metrics}


# ---------------------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------------------

def historyPath(run_name: str, out_dir: Path = FIGURES_DIR) -> Path:
    return Path(out_dir) / f"{run_name}_history.json"


def loadHistory(run_name: str, out_dir: Path = FIGURES_DIR) -> dict:
    """Reload the numbers a TrainingMonitor saved, e.g. to plot after loading model.pth."""
    return json.loads(historyPath(run_name, out_dir).read_text())


def _curveKinds(history: dict) -> list:
    return [kind for kind in ("loss", "acc")
            if any(history.get(f"{split}_{kind}") for split in ("train", "val"))]


def plotCurves(history: dict, title: str = "", val_label: str = "test", fig=None,
               total_epochs: int | None = None):
    """Loss vs epoch and accuracy vs epoch, from a dict of per-epoch lists.

    Keys: train_loss, val_loss, train_acc, val_acc - any subset. Panels with no data are
    left out. Pass `fig` to redraw into an existing figure (that is how the live window
    updates) and `total_epochs` to fix the x axis so the curve fills in left to right.
    Returns the figure.
    """
    kinds = _curveKinds(history)
    if not kinds:
        raise ValueError("history has no train/val loss or accuracy values to plot")

    panels = {"loss": ("Loss", None), "acc": ("Accuracy", PercentFormatter(1.0))}
    splits = (("train", "train", BLUE), ("val", val_label, ORANGE))

    with plt.rc_context(_STYLE): # type: ignore
        if fig is None:
            fig = plt.figure(figsize=(5.4 * len(kinds), 4.2), layout="constrained")
        elif len(fig.axes) != len(kinds):
            fig.set_size_inches(5.4 * len(kinds), 4.2)
        fig.clf()
        axes = fig.subplots(1, len(kinds), squeeze=False)[0]

        for ax, kind in zip(axes, kinds):
            name, formatter = panels[kind]
            for split, label, colour in splits:
                values = history.get(f"{split}_{kind}")
                if not values:
                    continue
                shown = f"{values[-1]:.1%}" if kind == "acc" else f"{values[-1]:.4f}"
                ax.plot(range(1, len(values) + 1), values, color=colour, linewidth=1.6,
                        marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        clip_on=False, label=f"{label}  {shown}")
            ax.set_title(name, loc="left", color=INK_2, fontsize=11)
            ax.set_xlabel("epoch")
            drawn = max(len(history.get(f"{split}_{kind}") or []) for split, _, _ in splits)
            last = max(drawn, total_epochs or 0)
            if last <= 12:
                ax.set_xticks(range(1, last + 1))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(0.5, last + 0.5)
            if formatter:
                ax.yaxis.set_major_formatter(formatter)
                ax.set_ylim(top=min(ax.get_ylim()[1], 1.01))
            else:
                ax.set_ylim(bottom=max(ax.get_ylim()[0], 0))
            # the grid is only for reading values off the y axis
            ax.grid(axis="x", visible=False)
            ax.legend(loc="best", fontsize=9)

        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    return fig


class TrainingMonitor:
    """Collects per-epoch numbers and keeps a loss / accuracy chart up to date.

    update() redraws the window (when the matplotlib backend can show one) and always
    rewrites <run_name>_curves.png and <run_name>_history.json, so an interrupted or
    headless run still leaves its curves behind. Call finish() after the last epoch;
    it turns interactive mode back off so a later plt.show() blocks and keeps the
    window open.
    """

    def __init__(self, run_name: str = "run", total_epochs: int | None = None,
                 val_label: str = "test", live: bool = True, out_dir: Path = FIGURES_DIR):
        self.run_name = run_name
        self.total_epochs = total_epochs
        self.val_label = val_label
        self.out_dir = Path(out_dir)
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

        epoch = len(next(iter(self.history.values())))
        of = f" of {self.total_epochs}" if self.total_epochs else ""
        self.fig = plotCurves(self.history, f"{self.run_name} - epoch {epoch}{of}",
                              self.val_label, self.fig, self.total_epochs)
        self._save()
        if self.live:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.05)

    def _save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.out_dir / f"{self.run_name}_curves.png", dpi=150) # type: ignore
        historyPath(self.run_name, self.out_dir).write_text(json.dumps(self.history, indent=2))

    def finish(self) -> dict:
        if self.fig is not None:
            self._save()
        if self.live and not self._was_interactive:
            plt.ioff()
        return self.history


def fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs: int,
        run_name: str = "run", val_label: str = "test", from_logits: bool = False,
        live: bool = True, out_dir: Path = FIGURES_DIR) -> dict:
    """Train for `epochs`, score `val_loader` after each one and chart it live.

    val_loader may be None to chart training numbers only. Returns the history dict.
    """
    monitor = TrainingMonitor(run_name, epochs, val_label, live, out_dir)
    try:
        for epoch in range(1, epochs + 1):
            train = runEpoch(model, train_loader, loss_fn, device, optimizer, from_logits)
            row = {"train_loss": train["loss"], "train_acc": train["accuracy"]}
            line = f"Epoch {epoch}/{epochs}  train loss {train['loss']:.4f} acc {train['accuracy']:.1%}"

            if val_loader is not None:
                val = runEpoch(model, val_loader, loss_fn, device, from_logits=from_logits)
                row.update(val_loss=val["loss"], val_acc=val["accuracy"])
                line += f"  |  {val_label} loss {val['loss']:.4f} acc {val['accuracy']:.1%}"

            print(line)
            monitor.update(**row)
    finally:
        monitor.finish()
    return monitor.history


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

    with plt.rc_context(_STYLE): # type: ignore
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
                    out_dir: Path = FIGURES_DIR) -> dict:
    """Evaluate `model` on `loader`, print the numbers, draw and save the charts."""
    results = evaluateLoader(model, loader, loss_fn, device, from_logits)
    print(f"{run_name} {split_label}: loss {results['loss']:.4f}  acc {results['accuracy']:.4f}  "
          f"precision {results['precision']:.4f}  recall {results['recall']:.4f}  "
          f"F1 {results['f1']:.4f}  ({results['average']})")
    plotEvaluation(results, class_names, run_name, split_label, out_dir)
    return results


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
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, len(classes))).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    fit(model, train_loader, test_loader, loss_fn, optimizer, device, epochs=8, run_name="demo")
    evaluateAndPlot(model, test_loader, loss_fn, device, run_name="demo", class_names=classes)
    plt.show()


if __name__ == "__main__":
    _demo()
