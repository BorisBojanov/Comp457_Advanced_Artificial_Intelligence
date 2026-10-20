"""
COMP 457 - Assignment 4, Part 2, Problem 1
Next-word prediction on WikiText-2 with an LSTM and a Transformer.

Pipeline (each step maps to a task in the assignment):
  1. Load WikiText-2 (Hugging Face `datasets`; GitHub fallback)
  2. Preprocess: lowercase -> strip unwanted characters -> tokenize ->
     build vocab -> turn the token stream into fixed-length n-gram windows
     (context of SEQ_LEN words -> next word)
  3. Train / validation / test splits (WikiText-2 ships with all three)
  4. Pre-trained GloVe embeddings (100-d, via gensim) to initialise nn.Embedding
  5. Two models: LSTMLanguageModel and TransformerLanguageModel (causal mask)
  6. Train with cross-entropy + AdamW; track loss, perplexity, top-1 / top-5 accuracy
  7. Test-set evaluation, top-k predictions for sample prompts,
     train-vs-validation curves saved to data/figures/
  8. Interactive prompt: type "the weather is" -> predicted next word(s)

Usage (run from the repo root):
  python src/Assignment4Part2Problem1.py                        # train both models
  python src/Assignment4Part2Problem1.py --model lstm --epochs 10
  python src/Assignment4Part2Problem1.py --quick                # 2-minute smoke test
  python src/Assignment4Part2Problem1.py --interactive          # load checkpoints, chat
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------
# Paths & configuration
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "wikitext2"
FIG_DIR = ROOT / "data" / "figures"
CKPT_DIR = ROOT / "checkpoints" / "a4p1"

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# 1. Load WikiText-2
# --------------------------------------------------------------------------
GITHUB_FALLBACK = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/{split}.txt"
)


def load_wikitext2() -> dict[str, list[str]]:
    """Return {'train': [lines], 'validation': [lines], 'test': [lines]}."""
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        print("Loaded WikiText-2 (raw) from Hugging Face.")
        return {s: list(ds[s]["text"]) for s in ("train", "validation", "test")}
    except Exception as e:  # no internet to HF, old datasets version, ...
        print(f"Hugging Face load failed ({type(e).__name__}); using GitHub copy.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = {}
        for split, fname in (("train", "train"), ("validation", "valid"), ("test", "test")):
            path = DATA_DIR / f"{fname}.txt"
            if not path.exists():
                urllib.request.urlretrieve(GITHUB_FALLBACK.format(split=fname), path)
            out[split] = path.read_text(encoding="utf-8").splitlines()
        return out


# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------
HEADING_RE = re.compile(r"^\s*(=\s*)+[^=]+(\s*=)+\s*$")  # " = = Gameplay = = "
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]")  # words + key punctuation


def clean_line(line: str) -> str:
    line = line.lower()
    line = line.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    line = line.replace("<unk>", " unkword ")  # keep the token in the fallback copy
    return line


def tokenize(text: str) -> list[str]:
    """Lowercase + regex tokenizer. Anything not matched (quotes, brackets,
    symbols, accents) is dropped - this is the 'remove unwanted characters' step."""
    toks = TOKEN_RE.findall(clean_line(text))
    return [UNK if t == "unkword" else t for t in toks]


def lines_to_tokens(lines: list[str]) -> list[str]:
    tokens: list[str] = []
    for line in lines:
        if not line.strip() or HEADING_RE.match(line):
            continue  # skip blank lines and section headings
        toks = tokenize(line)
        if toks:
            tokens.extend(toks + [EOS])  # <eos> marks paragraph boundaries
    return tokens


class Vocab:
    def __init__(self, tokens: list[str], min_freq: int = 3, max_size: int = 30000):
        counts = Counter(tokens)
        specials = [PAD, UNK, EOS]
        words = [w for w, c in counts.most_common() if c >= min_freq and w not in specials]
        self.itos = specials + words[: max_size - len(specials)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids) -> list[str]:
        return [self.itos[int(i)] for i in ids]


class NGramWindows(Dataset):
    """Slice the token-id stream into windows of SEQ_LEN+1 tokens.

    Each window gives SEQ_LEN input->output n-gram pairs at once:
        x = w[t : t+L],  y = w[t+1 : t+L+1]
    so position i of y is the 'next word' after the (i+1)-gram x[0..i].
    The causal LSTM / masked Transformer only ever sees the left context,
    so this is exactly the n-gram (context -> next word) task, just batched.
    """

    def __init__(self, ids: list[int], seq_len: int, stride: int | None = None):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.starts = list(range(0, len(ids) - seq_len - 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        chunk = self.ids[s : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# --------------------------------------------------------------------------
# 3. Pre-trained embeddings (GloVe)
# --------------------------------------------------------------------------
def load_glove_matrix(vocab: Vocab, dim: int = 100) -> tuple[torch.Tensor, float]:
    """Build a |V| x dim matrix: GloVe vector where available, small random otherwise."""
    import gensim.downloader as api

    print(f"Loading glove-wiki-gigaword-{dim} (first run downloads ~130 MB)...")
    kv = api.load(f"glove-wiki-gigaword-{dim}")
    rng = np.random.default_rng(0)
    mat = rng.normal(0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    hits = 0
    for i, w in enumerate(vocab.itos):
        if w in kv.key_to_index:
            mat[i] = kv[w]
            hits += 1
    mat[vocab.stoi[PAD]] = 0.0
    coverage = hits / len(vocab)
    print(f"GloVe coverage: {hits}/{len(vocab)} = {coverage:.1%}")
    return torch.from_numpy(mat), coverage


# --------------------------------------------------------------------------
# 4. Models
# --------------------------------------------------------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, emb: torch.Tensor, hidden=512, layers=2, dropout=0.3, freeze_emb=False):
        super().__init__()
        vocab_size, emb_dim = emb.shape
        self.embedding = nn.Embedding.from_pretrained(emb.clone(), freeze=freeze_emb, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):  # x: (B, T) -> logits (B, T, V)
        h, _ = self.lstm(self.drop(self.embedding(x)))
        return self.fc(self.drop(h))


class PositionalEncoding(nn.Module):
    """Sinusoidal positions (Vaswani et al., 2017; D2L Sec 11.6)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class TransformerLanguageModel(nn.Module):
    """Decoder-only (GPT-style) transformer: encoder layers + causal mask."""

    def __init__(self, emb: torch.Tensor, d_model=256, heads=4, layers=4, ff=1024,
                 dropout=0.2, freeze_emb=False, max_len=512):
        super().__init__()
        vocab_size, emb_dim = emb.shape
        self.embedding = nn.Embedding.from_pretrained(emb.clone(), freeze=freeze_emb, padding_idx=0)
        self.proj = nn.Linear(emb_dim, d_model)  # GloVe 100-d -> model width
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        T = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.pos(self.proj(self.embedding(x)))
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.fc(self.norm(h))


def build_model(name: str, emb: torch.Tensor, args) -> nn.Module:
    if name == "lstm":
        return LSTMLanguageModel(emb, hidden=args.hidden, layers=args.layers,
                                 dropout=args.dropout, freeze_emb=args.freeze_emb)
    return TransformerLanguageModel(emb, d_model=args.d_model, heads=args.heads,
                                    layers=args.tf_layers, ff=args.d_model * 4,
                                    dropout=args.dropout, freeze_emb=args.freeze_emb,
                                    max_len=max(512, args.seq_len))


# --------------------------------------------------------------------------
# 5. Training / evaluation
# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, clip=1.0):
    """One pass. Returns mean loss, perplexity, top-1 and top-5 accuracy."""
    train = optimizer is not None
    model.train(train)
    tot_loss, tot_tok, top1, top5 = 0.0, 0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)  # tame exploding grads
                optimizer.step()
            n = y.numel()
            tot_loss += loss.item() * n
            tot_tok += n
            k5 = logits.topk(5, dim=-1).indices
            top1 += (k5[..., 0] == y).sum().item()
            top5 += (k5 == y.unsqueeze(-1)).any(-1).sum().item()
    mean = tot_loss / tot_tok
    return {"loss": mean, "ppl": math.exp(min(mean, 20)), "acc": top1 / tot_tok, "top5": top5 / tot_tok}


def train_model(name, model, loaders, device, args, meta):
    criterion = nn.CrossEntropyLoss()  # categorical cross-entropy over the vocabulary
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    history = {k: [] for k in ("train_loss", "val_loss", "train_ppl", "val_ppl",
                               "train_acc", "val_acc", "val_top5", "lr")}
    best, bad = float("inf"), 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{name}_best.pt"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Training {name.upper()} ({n_params/1e6:.1f}M trainable params) on {device} ===")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va = run_epoch(model, loaders["validation"], criterion, device)
        scheduler.step(va["loss"])
        for k, v in (("train_loss", tr["loss"]), ("val_loss", va["loss"]),
                     ("train_ppl", tr["ppl"]), ("val_ppl", va["ppl"]),
                     ("train_acc", tr["acc"]), ("val_acc", va["acc"]),
                     ("val_top5", va["top5"]), ("lr", optimizer.param_groups[0]["lr"])):
            history[k].append(v)
        print(f"[{name}] epoch {epoch:2d}/{args.epochs} | train loss {tr['loss']:.3f} "
              f"ppl {tr['ppl']:.1f} | val loss {va['loss']:.3f} ppl {va['ppl']:.1f} "
              f"acc {va['acc']:.3f} top5 {va['top5']:.3f} | {time.time()-t0:.0f}s")
        if va["loss"] < best:  # checkpoint the best validation model
            best, bad = va["loss"], 0
            torch.save({"state_dict": model.state_dict(), "meta": meta, "model": name,
                        "args": vars(args), "epoch": epoch}, ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{name}] early stopping (no val improvement for {bad} epochs)")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    return history


def plot_curves(histories: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], "-o", ms=3, label=f"{name} train")
        axes[0].plot(ep, h["val_loss"], "--s", ms=3, label=f"{name} val")
        axes[1].plot(ep, h["train_ppl"], "-o", ms=3, label=f"{name} train")
        axes[1].plot(ep, h["val_ppl"], "--s", ms=3, label=f"{name} val")
        axes[2].plot(ep, h["val_acc"], "-o", ms=3, label=f"{name} val top-1")
        axes[2].plot(ep, h["val_top5"], "--s", ms=3, label=f"{name} val top-5")
    for ax, title, yl in zip(axes, ("Cross-entropy loss", "Perplexity", "Validation accuracy"),
                             ("loss", "perplexity", "accuracy")):
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel(yl)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[1].set_yscale("log")
    fig.suptitle("Assignment 4 P1 - next-word prediction on WikiText-2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved curves -> {path}")


# --------------------------------------------------------------------------
# 6. Prediction helpers (top-k + generation + interactive)
# --------------------------------------------------------------------------
@torch.no_grad()
def topk_next(model, vocab, prompt: str, device, k=5, seq_len=35, hide_unk=True):
    model.eval()
    ids = vocab.encode(tokenize(prompt))[-seq_len:] or [vocab.stoi[EOS]]
    logits = model(torch.tensor([ids], device=device))[0, -1]
    logits[vocab.stoi[PAD]] = -float("inf")
    if hide_unk:  # <unk> is a valid but useless suggestion for a human reader
        logits[vocab.stoi[UNK]] = -float("inf")
    probs = logits.softmax(-1)
    p, i = probs.topk(k)
    return [(vocab.itos[j], float(q)) for j, q in zip(i.tolist(), p.tolist())]


@torch.no_grad()
def generate(model, vocab, prompt: str, device, n_words=10, temperature=0.8, top_k=20, seq_len=35,
             rep_penalty=1.5):
    """Autoregressive continuation with temperature + top-k sampling.
    temperature <= 0 means greedy decoding. rep_penalty > 1 discourages
    re-using words from the last 10 positions (stops 'the the the ...' loops)."""
    model.eval()
    ids = vocab.encode(tokenize(prompt)) or [vocab.stoi[EOS]]
    banned = [vocab.stoi[PAD], vocab.stoi[UNK]]
    out = []
    for _ in range(n_words):
        logits = model(torch.tensor([ids[-seq_len:]], device=device))[0, -1]
        logits[banned] = -float("inf")
        for j in set(ids[-10:]):  # CTRL-style repetition penalty
            logits[j] = logits[j] / rep_penalty if logits[j] > 0 else logits[j] * rep_penalty
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            v, ix = (logits / temperature).topk(top_k)
            nxt = int(ix[torch.multinomial(v.softmax(-1), 1)])
        ids.append(nxt)
        out.append(vocab.itos[nxt])
        if out[-1] == EOS:
            break
    return " ".join(out)


def interactive(models: dict, vocab, device, seq_len):
    print("\nType a prompt (e.g. 'the weather is'); empty line or 'quit' to exit.")
    while True:
        try:
            prompt = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in {"quit", "exit", "q"}:
            break
        for name, m in models.items():
            preds = topk_next(m, vocab, prompt, device, k=5, seq_len=seq_len)
            print(f"  [{name:11s}] top-5: " + ", ".join(f"{w} ({p:.2f})" for w, p in preds))
            print(f"  [{name:11s}] greedy : {prompt} {generate(m, vocab, prompt, device, 8, 0, seq_len=seq_len)}")
            print(f"  [{name:11s}] sampled: {prompt} {generate(m, vocab, prompt, device, 12, 0.8, seq_len=seq_len)}")


def load_checkpoints(device):
    models, vocab, seq_len = {}, None, 35
    for name in ("lstm", "transformer"):
        path = CKPT_DIR / f"{name}_best.pt"
        if not path.exists():
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        vocab = Vocab([])
        vocab.itos = ck["meta"]["itos"]
        vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
        a = argparse.Namespace(**ck["args"])
        seq_len = a.seq_len
        dummy = torch.zeros(len(vocab.itos), ck["meta"]["emb_dim"])
        m = build_model(name, dummy, a).to(device)
        m.load_state_dict(ck["state_dict"])
        models[name] = m
    if not models:
        raise SystemExit(f"No checkpoints in {CKPT_DIR}. Train first.")
    return models, vocab, seq_len


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["lstm", "transformer", "both"], default="both")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=35, help="context length (n-1 of the n-gram)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512, help="LSTM hidden size")
    p.add_argument("--d-model", type=int, default=256, help="Transformer width")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2, help="LSTM layers")
    p.add_argument("--tf-layers", type=int, default=4, help="Transformer layers")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--emb-dim", type=int, default=100, choices=[50, 100, 200, 300])
    p.add_argument("--freeze-emb", action="store_true", help="keep GloVe vectors fixed")
    p.add_argument("--min-freq", type=int, default=3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="tiny subset, 2 epochs - smoke test")
    p.add_argument("--interactive", action="store_true", help="load checkpoints and chat")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()
    device = get_device()

    if args.interactive:
        models, vocab, seq_len = load_checkpoints(device)
        interactive(models, vocab, device, seq_len)
        return

    # --- data -------------------------------------------------------------
    raw = load_wikitext2()
    tokens = {s: lines_to_tokens(lines) for s, lines in raw.items()}
    if args.quick:
        tokens = {"train": tokens["train"][:60000], "validation": tokens["validation"][:8000],
                  "test": tokens["test"][:8000]}
        args.epochs = min(args.epochs, 2)
    vocab = Vocab(tokens["train"], min_freq=args.min_freq)
    for s, t in tokens.items():
        unk_rate = sum(1 for w in t if w not in vocab.stoi) / max(len(t), 1)
        print(f"{s:10s}: {len(t):,} tokens | OOV->unk rate {unk_rate:.2%}")
    print(f"Vocab size: {len(vocab):,}")
    print("Sample tokens:", " ".join(tokens["train"][:30]))

    datasets_ = {s: NGramWindows(vocab.encode(t), args.seq_len) for s, t in tokens.items()}
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == "train"), drop_last=(s == "train"))
               for s, d in datasets_.items()}
    print({s: len(d) for s, d in datasets_.items()}, "windows of", args.seq_len, "tokens")

    emb, coverage = load_glove_matrix(vocab, args.emb_dim)
    meta = {"itos": vocab.itos, "emb_dim": args.emb_dim}

    # --- train ------------------------------------------------------------
    names = ["lstm", "transformer"] if args.model == "both" else [args.model]
    histories, results, models = {}, {}, {}
    criterion = nn.CrossEntropyLoss()
    for name in names:
        set_seed()
        model = build_model(name, emb, args).to(device)
        histories[name] = train_model(name, model, loaders, device, args, meta)
        test = run_epoch(model, loaders["test"], criterion, device)
        results[name] = test
        models[name] = model
        print(f"[{name}] TEST loss {test['loss']:.3f} | ppl {test['ppl']:.1f} | "
              f"top-1 acc {test['acc']:.3f} | top-5 acc {test['top5']:.3f}")

    # --- report artefacts --------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if args.quick else ""
    plot_curves(histories, FIG_DIR / f"assignment4part2p1_curves{tag}.png")

    # top-k predictions for a few test inputs (real test-set contexts + custom prompts)
    test_ids = vocab.encode(tokens["test"])
    rng = np.random.default_rng(1)
    samples = []
    for s in rng.choice(len(test_ids) - 20, 4, replace=False):
        ctx = " ".join(vocab.decode(test_ids[s : s + 10]))
        samples.append((ctx, vocab.itos[test_ids[s + 10]]))
    samples += [("the weather is", None), ("the united states", None), ("he was born in", None)]
    topk_report = []
    print("\n=== Top-5 next-word predictions ===")
    for ctx, truth in samples:
        print(f"\ncontext: '{ctx}'" + (f"  | true next: '{truth}'" if truth else ""))
        row = {"context": ctx, "truth": truth}
        for name, m in models.items():
            preds = topk_next(m, vocab, ctx, device, 5, args.seq_len)
            row[name] = preds
            print(f"  {name:11s}: " + ", ".join(f"{w} ({p:.2f})" for w, p in preds))
        topk_report.append(row)

    with open(FIG_DIR / f"assignment4part2p1_results{tag}.json", "w") as f:
        json.dump({"args": vars(args), "vocab_size": len(vocab), "glove_coverage": coverage,
                   "test": results, "history": histories, "topk": topk_report}, f, indent=2)
    print(f"Saved results -> {FIG_DIR / f'assignment4part2p1_results{tag}.json'}")

    if not args.quick and os.isatty(0):
        interactive(models, vocab, device, args.seq_len)


if __name__ == "__main__":
    main()
