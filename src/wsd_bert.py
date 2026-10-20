"""BERT gloss-pair model for Problem 2 (WiC / GlossBERT-style fine-tuning).

Instead of one output per sense, BERT is asked a yes/no question about a PAIR:

    [CLS] he deposited money in the " bank " [SEP] bank : a financial institution ... [SEP]
                       sentence, target marked                 target : WordNet gloss

and outputs ONE number: how well this gloss fits the target word in this sentence.
To disambiguate a word, score every WordNet sense (gloss) of it and pick the best one.
That is why the output size is 1 no matter how many senses a word has, and why it can rank the
senses of a word it never saw in training.

Training:   1 right gloss (label 1) + NEG_PER_POS wrong glosses of the same word (label 0) per example.
Prediction: all glosses of the word, softmax over their scores.

Batches are one tensor [B, 3, L] (input_ids, attention_mask, token_type_ids) so that fit() and
runEpoch() in model_eval.py - which expect `model(X)` with a single tensor - work unchanged.
"""

import random
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from nltk.corpus import wordnet as wn

from model_eval import fit

BERT_NAME = "bert-base-uncased"
MAX_LEN = 96            # tokens per (sentence, gloss) pair
SENT_WINDOW = 25        # words kept either side of the target, so the target is never truncated away
NEG_PER_POS = 4         # wrong glosses sampled per correct one in training
BATCH_SIZE = 32
LR = 2e-5               # BERT fine-tuning needs a small learning rate (SGD 1e-3 would wreck it)


class BertGlossPair(nn.Module):
    """BERT encoder + one linear layer on the [CLS] vector -> a single match logit per pair."""

    def __init__(self, name: str = BERT_NAME, config=None):
        super().__init__()
        from transformers import AutoModel, BertModel
        # `config` builds a randomly initialised model (used by tests, no download)
        self.bert = AutoModel.from_pretrained(name) if config is None else BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.score = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, x):
        # x: [B, 3, L] -> match logits [B]
        ids, mask, types = x[:, 0], x[:, 1], x[:, 2]
        cls = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=types).last_hidden_state[:, 0]
        return self.score(self.dropout(cls)).squeeze(-1)


def loadTokenizer(name: str = BERT_NAME):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


# ---------------------------------------------------------------------------------------
# Building the (sentence, gloss) pairs
# ---------------------------------------------------------------------------------------

def markTarget(tokens, target_idx, window: int = SENT_WINDOW) -> str:
    """The sentence around the target, with the target wrapped in quotes: ... the " bank " ..."""
    left = tokens[max(0, target_idx - window):target_idx]
    right = tokens[target_idx + 1:target_idx + 1 + window]
    return " ".join(left + ['"', tokens[target_idx], '"'] + right)


@lru_cache(maxsize=None)
def glossText(lemma: str, synset_name: str) -> str:
    return f"{lemma} : {wn.synset(synset_name).definition()}"


def candidateSynsets(lemma: str) -> list:
    # same inventory as buildSenseSpace in wsd_data.py (all POS), so labels line up
    return [s.name() for s in wn.synsets(lemma)]


def encodePairs(tokenizer, sentences, glosses) -> torch.Tensor:
    enc = tokenizer(sentences, glosses, truncation=True, max_length=MAX_LEN,
                    padding="max_length", return_tensors="pt")
    return torch.stack([enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]], dim=1)


def buildTrainLoader(examples, tokenizer, neg_per_pos: int = NEG_PER_POS, batch_size: int = BATCH_SIZE,
                     seed: int = 0) -> DataLoader:
    """Positive + sampled negative pairs for every training example.

    The negatives are drawn once here (seeded), not re-drawn each epoch, to keep this simple.
    """
    rng = random.Random(seed)
    sentences, glosses, labels = [], [], []
    for e in examples:
        sentence = markTarget(e["tokens"], e["target_idx"])
        wrong = [c for c in candidateSynsets(e["target"]) if c != e["synset"]]
        chosen = [e["synset"]] + rng.sample(wrong, min(neg_per_pos, len(wrong)))
        for synset in chosen:
            sentences.append(sentence)
            glosses.append(glossText(e["target"], synset))
            labels.append(1.0 if synset == e["synset"] else 0.0)

    X = encodePairs(tokenizer, sentences, glosses)
    y = torch.tensor(labels, dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------------------

def trainBert(train_examples, device, epochs: int = 3, run_name: str = "bert", name: str = BERT_NAME,
              model=None, tokenizer=None):
    """Fine-tune on the gloss pairs. fit() charts the pair-level loss / accuracy live.

    Pair accuracy is inflated (4 of 5 pairs are negatives), so judge the model by the
    sense-level numbers from predictSenseIdx(), not by this curve.
    """
    tokenizer = tokenizer or loadTokenizer(name)
    model = (model or BertGlossPair(name)).to(device)
    loader = buildTrainLoader(train_examples, tokenizer)
    fit(model, loader, None, nn.BCEWithLogitsLoss(), torch.optim.AdamW(model.parameters(), lr=LR),
        device, epochs, run_name=run_name, from_logits=True)
    return model, tokenizer


@torch.no_grad()
def rankSenses(model, tokenizer, device, tokens, target_idx: int, lemma: str, batch_size: int = 64) -> list:
    """Score every WordNet sense of `lemma` for this occurrence.

    Returns [(synset_name, probability), ...], best first. The probabilities are a softmax over
    this word's candidate glosses only.
    """
    model.eval()
    candidates = candidateSynsets(lemma)
    if not candidates:
        return []

    sentence = markTarget(tokens, target_idx)
    scores = []
    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i:i + batch_size]
        X = encodePairs(tokenizer, [sentence] * len(chunk), [glossText(lemma, c) for c in chunk])
        scores.append(model(X.to(device)).float().cpu())
    probs = torch.softmax(torch.cat(scores), dim=0).tolist()
    return sorted(zip(candidates, probs), key=lambda pair: -pair[1])


def predictSenseIdx(model, tokenizer, device, examples, sense2idx: dict) -> np.ndarray:
    """Best-scoring sense (as a class index) for each example, for the shared metrics code."""
    preds = []
    for e in examples:
        ranked = rankSenses(model, tokenizer, device, e["tokens"], e["target_idx"], e["target"])
        preds.append(sense2idx[ranked[0][0]])
    return np.array(preds)
