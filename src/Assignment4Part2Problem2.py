"""
COMP 457 - Assignment 4, Part 2, Problem 2
Image captioning on Flickr8k: {ResNet50, InceptionV3} encoders x {LSTM, Transformer} decoders.

Pipeline (each step maps to a task in the assignment):
  1. Download Flickr8k (images + captions + official 6000/1000/1000 split)
  2. Preprocess captions: lowercase, strip punctuation/digits, add <start>/<end>,
     build vocab (min_freq), pad/truncate to MAX_LEN
     Preprocess images: resize (224 for ResNet50, 299 for InceptionV3) + ImageNet normalisation
  3. Frozen pre-trained CNNs as feature extractors. Features are computed ONCE and cached:
     a GxG grid of 2048-d region vectors per image (ResNet50 'layer4', InceptionV3 'Mixed_7c')
  4. Decoders
       - LSTM with soft (additive) attention  (Show, Attend and Tell - Xu et al., 2015;
         the "PyTorch Tutorial to Image Captioning" from Unit 13)
       - Transformer decoder (masked self-attention + cross-attention to image regions)
  5. Teacher-forced training with cross-entropy, checkpoint every epoch + best model
  6. Beam-search captions for the test set; METEOR + ROUGE-L (+ BLEU-1..4 as extra);
     example figure with generated vs ground-truth captions; training curves

Usage (run from the repo root):
  python src/Assignment4Part2Problem2.py                          # everything, 4 models
  python src/Assignment4Part2Problem2.py --encoder resnet50 --decoder lstm --epochs 15
  python src/Assignment4Part2Problem2.py --stage eval             # re-evaluate saved checkpoints
  python src/Assignment4Part2Problem2.py --caption path/to/photo.jpg   # caption your own image
  python src/Assignment4Part2Problem2.py --quick                  # small smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flickr8k"
IMG_DIR = DATA / "Flicker8k_Dataset"  # (sic) - the archive's own spelling
FEAT_DIR = DATA / "features"
FIG_DIR = ROOT / "data" / "figures"
CKPT_DIR = ROOT / "checkpoints" / "a4p2"

URLS = {
    "Flickr8k_text.zip": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
    "Flickr8k_Dataset.zip": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
}
PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
ENCODERS = {"resnet50": 224, "inceptionv3": 299}  # name -> input resolution


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ==========================================================================
# 1. Data download + caption preprocessing
# ==========================================================================
def download_flickr8k():
    DATA.mkdir(parents=True, exist_ok=True)
    for fname, url in URLS.items():
        zpath = DATA / fname
        marker = IMG_DIR if "Dataset" in fname else DATA / "Flickr8k.token.txt"
        if marker.exists():
            continue
        if not zpath.exists():
            print(f"Downloading {fname} (~{'1.1 GB' if 'Dataset' in fname else '2 MB'}) ...")
            urllib.request.urlretrieve(url, zpath)
        print(f"Extracting {fname} ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATA)
    print(f"Flickr8k ready: {len(list(IMG_DIR.glob('*.jpg')))} images")


def clean_caption(text: str) -> list[str]:
    """lowercase -> keep letters only -> drop 1-letter tokens except 'a'."""
    text = re.sub(r"[^a-z ]", " ", text.lower())
    return [w for w in text.split() if len(w) > 1 or w == "a"]


def load_captions() -> dict[str, list[list[str]]]:
    caps: dict[str, list[list[str]]] = {}
    for line in (DATA / "Flickr8k.token.txt").read_text().splitlines():
        if "\t" not in line:
            continue
        key, text = line.split("\t", 1)
        img = key.split("#")[0]
        if (IMG_DIR / img).exists():  # a few entries point to missing files
            caps.setdefault(img, []).append(clean_caption(text))
    return caps


def load_split(name: str) -> list[str]:
    f = {"train": "Flickr_8k.trainImages.txt", "val": "Flickr_8k.devImages.txt",
         "test": "Flickr_8k.testImages.txt"}[name]
    return [l.strip() for l in (DATA / f).read_text().splitlines() if (IMG_DIR / l.strip()).exists()]


class Vocab:
    def __init__(self, captions: list[list[str]] | None = None, min_freq=5):
        self.itos = [PAD, START, END, UNK]
        if captions:
            c = Counter(w for cap in captions for w in cap)
            self.itos += sorted(w for w, n in c.items() if n >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, words: list[str], max_len: int) -> list[int]:
        """<start> w1 .. wn <end> <pad> ... (truncated / padded to max_len)."""
        ids = [self.stoi[START]] + [self.stoi.get(w, self.stoi[UNK]) for w in words][: max_len - 2]
        ids.append(self.stoi[END])
        return ids + [self.stoi[PAD]] * (max_len - len(ids))

    def decode(self, ids) -> list[str]:
        out = []
        for i in ids:
            w = self.itos[int(i)]
            if w == END:
                break
            if w not in (PAD, START):
                out.append(w)
        return out


# ==========================================================================
# 2-3. Image preprocessing + frozen CNN feature extraction (cached)
# ==========================================================================
def image_transform(size: int):
    from torchvision import transforms as T

    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
    ])


class ImageFolderList(Dataset):
    def __init__(self, names, size):
        self.names, self.tf = names, image_transform(size)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.tf(Image.open(IMG_DIR / self.names[i]).convert("RGB"))


def build_encoder(name: str, pretrained=True) -> nn.Module:
    """Pre-trained CNN with the classifier removed; returns the last conv feature map."""
    from torchvision import models
    from torchvision.models.feature_extraction import create_feature_extractor

    if name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net, node = models.resnet50(weights=w), "layer4"  # (B, 2048, 7, 7) for 224x224
    else:
        w = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.inception_v3(weights=w, aux_logits=True, init_weights=not pretrained)
        node = "Mixed_7c"  # (B, 2048, 8, 8) for 299x299
    enc = create_feature_extractor(net, return_nodes={node: "feat"}).eval()
    for p in enc.parameters():
        p.requires_grad = False  # frozen feature extractor
    return enc


@torch.no_grad()
def encode_images(enc, x, grid):
    fmap = enc(x)["feat"]                                  # (B, 2048, h, w)
    fmap = F.adaptive_avg_pool2d(fmap, grid)               # (B, 2048, G, G)
    return fmap.flatten(2).transpose(1, 2)                 # (B, G*G, 2048) region vectors


def extract_features(encoder_name, split, names, device, grid, pretrained=True, batch=32, tag=""):
    path = FEAT_DIR / f"{encoder_name}_{split}_g{grid}{tag}.pt"
    if path.exists():
        cached = torch.load(path)
        if cached["names"] == names:
            print(f"Using cached features {path.name}")
            return cached["feats"]
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    enc = build_encoder(encoder_name, pretrained).to(device)
    loader = DataLoader(ImageFolderList(names, ENCODERS[encoder_name]), batch_size=batch,
                        num_workers=min(4, os.cpu_count() or 1))
    out, t0 = [], time.time()
    for i, x in enumerate(loader):
        out.append(encode_images(enc, x.to(device), grid).float().cpu())
        if i % 20 == 0:
            print(f"  [{encoder_name}] {min((i+1)*batch, len(names))}/{len(names)} images "
                  f"({time.time()-t0:.0f}s)")
    feats = torch.cat(out)
    if not torch.isfinite(feats).all():
        raise RuntimeError(f"{encoder_name} produced non-finite features - check the CNN weights")
    if feats.abs().max() < 6e4:
        feats = feats.half()  # fp16 halves disk/RAM; safe because values are bounded
    torch.save({"names": names, "feats": feats}, path)
    print(f"Saved {tuple(feats.shape)} features -> {path}")
    return feats


class CaptionDataset(Dataset):
    """One item per (image, caption) pair - 5 per image."""

    def __init__(self, feats, img_names, captions, vocab, max_len):
        self.feats = feats
        self.items = [(i, vocab.encode(c, max_len))
                      for i, n in enumerate(img_names) for c in captions[n]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, ids = self.items[idx]
        return self.feats[i].float(), torch.tensor(ids)


# ==========================================================================
# 4. Decoders  - both expose forward(feats (B,R,2048), tokens (B,T)) -> logits (B,T,V)
# ==========================================================================
class AdditiveAttention(nn.Module):
    """Bahdanau attention: score(h, v_i) = w^T tanh(W_v v_i + W_h h)."""

    def __init__(self, feat_dim, hidden, attn_dim):
        super().__init__()
        self.v = nn.Linear(feat_dim, attn_dim)
        self.h = nn.Linear(hidden, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, feats_proj, feats, h):
        e = self.w(torch.tanh(feats_proj + self.h(h).unsqueeze(1))).squeeze(-1)  # (B, R)
        alpha = e.softmax(-1)
        return (alpha.unsqueeze(-1) * feats).sum(1), alpha               # context (B, F)


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, feat_dim=2048, emb=256, hidden=512, attn=256, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.attn = AdditiveAttention(feat_dim, hidden, attn)
        self.init_h = nn.Linear(feat_dim, hidden)  # h0, c0 from the mean image vector
        self.init_c = nn.Linear(feat_dim, hidden)
        self.gate = nn.Linear(hidden, feat_dim)    # "f_beta" gate on the attended context
        self.cell = nn.LSTMCell(emb + feat_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab_size)
        self.last_alphas = None

    def forward(self, feats, tokens):
        B, T = tokens.shape
        mean = feats.mean(1)
        h, c = torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))
        feats_proj = self.attn.v(feats)            # pre-compute W_v v_i once
        emb = self.embed(tokens)
        logits, alphas = [], []
        for t in range(T):
            ctx, a = self.attn(feats_proj, feats, h)
            ctx = torch.sigmoid(self.gate(h)) * ctx
            h, c = self.cell(torch.cat([emb[:, t], ctx], -1), (h, c))
            logits.append(self.fc(self.drop(h)))
            alphas.append(a)
        self.last_alphas = torch.stack(alphas, 1)  # (B, T, R) - used for the attention regulariser
        return torch.stack(logits, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, feat_dim=2048, d_model=512, heads=8, layers=3,
                 dropout=0.1, regions=49, max_len=100):
        super().__init__()
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, d_model), nn.LayerNorm(d_model))
        self.region_pos = nn.Parameter(torch.zeros(1, regions, d_model))  # learned 2-D slot embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerDecoderLayer(d_model, heads, d_model * 4, dropout,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, feats, tokens):
        memory = self.feat_proj(feats) + self.region_pos[:, : feats.size(1)]
        x = self.drop(self.pos(self.embed(tokens) * self.scale))
        T = tokens.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        pad_mask = tokens.eq(0)
        pad_mask[:, 0] = False  # never fully mask a row
        h = self.decoder(x, memory, tgt_mask=causal, tgt_is_causal=True,
                         tgt_key_padding_mask=pad_mask)
        return self.fc(self.norm(h))


def build_decoder(name, vocab_size, regions, args):
    if name == "lstm":
        return LSTMDecoder(vocab_size, hidden=args.hidden, dropout=0.5)
    return TransformerCaptionDecoder(vocab_size, d_model=args.d_model, heads=args.heads,
                                     layers=args.tf_layers, regions=regions)


# ==========================================================================
# Decoding: batched beam search (beam=1 == greedy)
# ==========================================================================
@torch.no_grad()
def beam_search(decoder, feats, vocab, max_len=25, beam=3, alpha=0.7):
    decoder.eval()
    B, V = feats.size(0), len(vocab)
    start, end, pad, unk = vocab.stoi[START], vocab.stoi[END], vocab.stoi[PAD], vocab.stoi[UNK]
    fk = feats.repeat_interleave(beam, 0)
    seqs = torch.full((B * beam, 1), start, device=feats.device)
    scores = torch.zeros(B, beam, device=feats.device)
    scores[:, 1:] = -1e9                         # only one live beam at t=0
    done = torch.zeros(B * beam, dtype=torch.bool, device=feats.device)
    lengths = torch.zeros(B * beam, device=feats.device)
    for _ in range(max_len):
        logp = decoder(fk, seqs)[:, -1].float().log_softmax(-1)  # (B*k, V)
        logp[:, pad] = -1e9
        logp[:, start] = -1e9
        logp[:, unk] = -1e9                      # never emit <unk>
        logp[done] = -1e9
        logp[done, end] = 0.0                    # finished beams just carry their score
        cand = (scores.view(-1, 1) + logp).view(B, beam * V)
        top, idx = cand.topk(beam, -1)
        src = (torch.arange(B, device=feats.device).unsqueeze(1) * beam + idx // V).view(-1)
        tok = (idx % V).view(-1, 1)
        seqs = torch.cat([seqs[src], tok], 1)
        lengths = lengths[src] + (~done[src]).float()
        done = done[src] | tok.squeeze(1).eq(end)
        scores = top
        if done.all():
            break
    norm = scores.view(-1) / lengths.clamp(min=1).pow(alpha)  # length normalisation
    best = norm.view(B, beam).argmax(-1) + torch.arange(B, device=feats.device) * beam
    return [vocab.decode(s[1:]) for s in seqs[best].tolist()]


def caption_images(decoder, feats, vocab, device, beam=3, batch=50):
    out = []
    for i in range(0, len(feats), batch):
        out += beam_search(decoder, feats[i:i + batch].float().to(device), vocab, beam=beam)
    return out


# ==========================================================================
# 6. Metrics: METEOR, ROUGE-L, BLEU
# ==========================================================================
def ensure_nltk():
    import nltk
    from nltk.corpus import wordnet

    try:
        wordnet.ensure_loaded()  # METEOR uses WordNet synonyms
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def caption_metrics(hyps: list[list[str]], refs: list[list[list[str]]]) -> dict:
    """METEOR: unigram F-mean with stemming/synonym matching + fragmentation penalty.
    ROUGE-L: longest-common-subsequence F-score (best over the 5 references).
    BLEU-n: n-gram precision with brevity penalty (corpus level)."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    ensure_nltk()
    meteor = np.mean([meteor_score(r, h) for h, r in zip(hyps, refs)])
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    rouge = np.mean([max(scorer.score(" ".join(x), " ".join(h))["rougeL"].fmeasure for x in r)
                     for h, r in zip(hyps, refs)])
    sm = SmoothingFunction().method1
    bleu = {f"bleu{n}": corpus_bleu(refs, hyps, weights=tuple([1 / n] * n), smoothing_function=sm)
            for n in (1, 2, 3, 4)}
    vocab_used = len({w for h in hyps for w in h})
    return {"meteor": float(meteor), "rougeL": float(rouge), **bleu,
            "avg_len": float(np.mean([len(h) for h in hyps])), "distinct_words": vocab_used}


# ==========================================================================
# 5. Training
# ==========================================================================
def train_one(tag, enc_name, dec_name, data, vocab, device, args):
    feats_tr, feats_va = data["train_feats"], data["val_feats"]
    regions = feats_tr.size(1)
    train_ds = CaptionDataset(feats_tr, data["train"], data["caps"], vocab, args.max_len)
    val_ds = CaptionDataset(feats_va, data["val"], data["caps"], vocab, args.max_len)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size)

    set_seed()
    model = build_decoder(dec_name, len(vocab), regions, args).to(device)
    lr = args.lr if args.lr else (4e-4 if dec_name == "lstm" else 2e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = args.epochs * len(tl)
    warm = len(tl)  # 1 epoch linear warm-up, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    crit = nn.CrossEntropyLoss(ignore_index=vocab.stoi[PAD], label_smoothing=0.1)

    # small fixed val subset for a per-epoch METEOR curve (cheap greedy decoding)
    n_mon = min(200, len(data["val"]))
    mon_refs = [data["caps"][n] for n in data["val"][:n_mon]]

    hist = {k: [] for k in ("train_loss", "val_loss", "val_meteor", "val_bleu4", "lr")}
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best = -1.0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {tag}: {n_params/1e6:.1f}M params | {len(train_ds):,} train pairs | lr {lr} | {device} ===")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for f, cap in tl:
            f, cap = f.to(device), cap.to(device)
            inp, tgt = cap[:, :-1], cap[:, 1:]  # teacher forcing: feed w_<t, predict w_t
            logits = model(f, inp)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            if dec_name == "lstm" and args.att_reg > 0:  # doubly-stochastic attention
                loss = loss + args.att_reg * ((1 - model.last_alphas.sum(1)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * f.size(0); n += f.size(0)
        tr_loss = tot / n
        if not math.isfinite(tr_loss):
            raise RuntimeError(f"[{tag}] training loss is {tr_loss} - lower --lr or check features")

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for f, cap in vl:
                f, cap = f.to(device), cap.to(device)
                logits = model(f, cap[:, :-1])
                vtot += crit(logits.reshape(-1, logits.size(-1)), cap[:, 1:].reshape(-1)).item() * f.size(0)
                vn += f.size(0)
        va_loss = vtot / vn
        hyps = caption_images(model, feats_va[:n_mon], vocab, device, beam=1)
        m = caption_metrics(hyps, mon_refs)
        for k, v in (("train_loss", tr_loss), ("val_loss", va_loss), ("val_meteor", m["meteor"]),
                     ("val_bleu4", m["bleu4"]), ("lr", opt.param_groups[0]["lr"])):
            hist[k].append(v)
        print(f"[{tag}] ep {ep:2d}/{args.epochs} | train {tr_loss:.3f} | val {va_loss:.3f} | "
              f"val METEOR {m['meteor']:.3f} BLEU-4 {m['bleu4']:.3f} | {time.time()-t0:.0f}s "
              f"| e.g. '{' '.join(hyps[0])}'")

        ck = {"state_dict": model.state_dict(), "vocab": vocab.itos, "encoder": enc_name,
              "decoder": dec_name, "regions": regions, "args": vars(args), "epoch": ep, "history": hist}
        torch.save(ck, CKPT_DIR / f"{tag}_last.pt")          # resume / inspect
        if m["meteor"] > best:                                 # model selection on val METEOR
            best = m["meteor"]
            torch.save(ck, CKPT_DIR / f"{tag}_best.pt")
    return hist


def load_decoder(tag, device):
    ck = torch.load(CKPT_DIR / f"{tag}_best.pt", map_location=device, weights_only=False)
    vocab = Vocab()
    vocab.itos = ck["vocab"]
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    a = argparse.Namespace(**ck["args"])
    model = build_decoder(ck["decoder"], len(vocab), ck["regions"], a).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), vocab, ck


# ==========================================================================
# Figures
# ==========================================================================
def plot_curves(histories, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for tag, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        line, = ax[0].plot(ep, h["train_loss"], "-o", ms=3, label=f"{tag} train")
        ax[0].plot(ep, h["val_loss"], "--", color=line.get_color(), label=f"{tag} val")
        ax[1].plot(ep, h["val_meteor"], "-o", ms=3, color=line.get_color(), label=tag)
    ax[0].set(title="Cross-entropy loss (label-smoothed)", xlabel="epoch", ylabel="loss")
    ax[1].set(title="Validation METEOR (greedy, 200 images)", xlabel="epoch", ylabel="METEOR")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved {path}")


def plot_examples(names, preds: dict, caps, path, n=6):
    import textwrap

    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6.2 * rows))
    for ax, name in zip(axes.flat, names[:n]):
        ax.imshow(Image.open(IMG_DIR / name).convert("RGB"))
        ax.axis("off")
        lines = [f"GT: {' '.join(caps[name][0])}", f"GT: {' '.join(caps[name][1])}"]
        lines += [f"{tag}: {' '.join(p[name])}" for tag, p in preds.items()]
        txt = "\n".join(textwrap.fill(l, 60) for l in lines)
        ax.set_title(txt, fontsize=7.5, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"Saved {path}")


# ==========================================================================
# main
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", choices=["resnet50", "inceptionv3", "both"], default="both")
    p.add_argument("--decoder", choices=["lstm", "transformer", "both"], default="both")
    p.add_argument("--stage", choices=["all", "train", "eval"], default="all")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="default: 4e-4 LSTM, 2e-4 Transformer")
    p.add_argument("--grid", type=int, default=7, help="GxG image regions kept per image")
    p.add_argument("--max-len", type=int, default=24, help="caption length incl. <start>/<end>")
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--tf-layers", type=int, default=3)
    p.add_argument("--att-reg", type=float, default=1.0, help="LSTM attention regulariser weight")
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--caption", type=str, default=None, help="caption one image file and exit")
    p.add_argument("--quick", action="store_true", help="300 train / 100 val / 100 test images, 2 epochs")
    p.add_argument("--no-pretrained", action="store_true", help="random CNN weights (offline testing only)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()
    device = get_device()
    encs = list(ENCODERS) if args.encoder == "both" else [args.encoder]
    decs = ["lstm", "transformer"] if args.decoder == "both" else [args.decoder]
    pretrained = not args.no_pretrained
    tag_sfx = "_quick" if args.quick else ""

    if args.caption:  # caption a user image with every trained model
        for e in encs:
            enc = build_encoder(e, pretrained).to(device)
            x = image_transform(ENCODERS[e])(Image.open(args.caption).convert("RGB"))[None].to(device)
            f = encode_images(enc, x, args.grid)
            for d in decs:
                if (CKPT_DIR / f"{e}_{d}{tag_sfx}_best.pt").exists():
                    m, vocab, _ = load_decoder(f"{e}_{d}{tag_sfx}", device)
                    print(f"{e}+{d}: {' '.join(beam_search(m, f, vocab, beam=args.beam)[0])}")
        return

    download_flickr8k()
    caps = load_captions()
    splits = {s: load_split(s) for s in ("train", "val", "test")}
    if args.quick:
        splits = {"train": splits["train"][:300], "val": splits["val"][:100], "test": splits["test"][:100]}
        args.epochs = min(args.epochs, 2)
    lens = [len(c) for n in splits["train"] for c in caps[n]]
    vocab = Vocab([c for n in splits["train"] for c in caps[n]], args.min_freq)
    print({s: len(v) for s, v in splits.items()}, "images |", f"vocab {len(vocab)} |",
          f"caption length mean {np.mean(lens):.1f}, 95th pct {np.percentile(lens, 95):.0f}")
    print("Example:", splits["train"][0], caps[splits["train"][0]][0])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    histories, results, preds = {}, {}, {}
    for e in encs:
        fsfx = tag_sfx + ("_rand" if not pretrained else "")
        data = {"caps": caps, **splits,
                "train_feats": extract_features(e, "train", splits["train"], device, args.grid, pretrained, tag=fsfx),
                "val_feats": extract_features(e, "val", splits["val"], device, args.grid, pretrained, tag=fsfx)}
        test_feats = extract_features(e, "test", splits["test"], device, args.grid, pretrained, tag=fsfx)
        for d in decs:
            tag = f"{e}_{d}{tag_sfx}"
            if args.stage in ("all", "train"):
                histories[tag] = train_one(tag, e, d, data, vocab, device, args)
            if args.stage in ("all", "eval"):
                model, vocab_m, ck = load_decoder(tag, device)
                histories.setdefault(tag, ck["history"])
                hyps = caption_images(model, test_feats, vocab_m, device, beam=args.beam)
                refs = [caps[n] for n in splits["test"]]
                results[tag] = caption_metrics(hyps, refs)
                preds[tag] = dict(zip(splits["test"], hyps))
                print(f"[{tag}] TEST (beam {args.beam}): " +
                      " | ".join(f"{k} {v:.3f}" for k, v in results[tag].items()))

    plot_curves(histories, FIG_DIR / f"assignment4part2p2_curves{tag_sfx}.png")
    if preds:
        rng = np.random.default_rng(7)
        pick = list(rng.choice(splits["test"], 6, replace=False))
        plot_examples(pick, preds, caps, FIG_DIR / f"assignment4part2p2_examples{tag_sfx}.png")
        print("\n=== Results table (paste into your report) ===")
        print("| Model | METEOR | ROUGE-L | BLEU-1 | BLEU-4 | avg len |\n|---|---|---|---|---|---|")
        for t, r in results.items():
            print(f"| {t} | {r['meteor']:.3f} | {r['rougeL']:.3f} | {r['bleu1']:.3f} | "
                  f"{r['bleu4']:.3f} | {r['avg_len']:.1f} |")
        with open(FIG_DIR / f"assignment4part2p2_results{tag_sfx}.json", "w") as f:
            json.dump({"args": vars(args), "test": results, "history": histories,
                       "examples": {n: {"gt": [" ".join(c) for c in caps[n]],
                                        **{t: " ".join(p[n]) for t, p in preds.items()}} for n in pick}},
                      f, indent=2)


if __name__ == "__main__":
    main()
