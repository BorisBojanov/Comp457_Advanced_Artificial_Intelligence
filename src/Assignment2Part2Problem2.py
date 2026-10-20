

from pathlib import Path

import math
import random
from typing import Any
import matplotlib.pyplot as plt

# Model
import torch 
from torch import nn

# Data Pipeline
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Tokenize Data
import re
from collections import Counter

import os
import glob

# SemCor data
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
from nltk.corpus.reader.wordnet import Lemma

# Live training curves + evaluation charts (see model_eval.py)
import argparse
import json
import sys
import numpy as np

from model_eval import fit, evaluateAndPlot, computeMetrics, collectPredictions, evaluatePerGroup

# SemCor extraction, train/test split and WordNet sense space (see wsd_data.py)
from wsd_data import TARGET_WORDS, REQUIRED_WORDS, semcor_dataset, buildSenseSpace, mostFrequentSense

# BERT gloss-pair model (see wsd_bert.py)
import wsd_bert

DOC_FIGS_NAME ="assignment2part2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# own file: model.pth belongs to the Problem 1 / Assignment 1 models and would be overwritten
MODEL_PATH = Path(__file__).resolve().parent.parent / "wsd_lstm.pth"
VOCAB_PATH = Path(__file__).resolve().parent.parent / "wsd_lstm_vocab.json"   # the LSTM needs its vocab to predict later
BERT_PATH = Path(__file__).resolve().parent.parent / "wsd_bert.pth"
BERT_EPOCHS = 3

# Hyperparameters
VOCAB_SIZE = 5000       # Size of each dictionary / vocabulary
EMBEDDING_DIM = 100     # Dimension of word vectors
HIDDEN_DIM = 128        # Dimensions of RNN internal hidden state
BATCH_SIZE = 32
EPOCHS = 5

MAX_LEN = 200
K = 10                  # window half-width: the model sees K words either side of the target
PAD, UNK = 0, 1

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WSDLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_sense, k, sense_mask=None):
        super().__init__()
        self.k = k
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True) # bidirectional=True for ±k words
        self.fc = nn.Linear(2 * hidden_dim, num_sense)    # 2 * hidden dimension because of bidirectional
        # [num_targets, num_senses] bool: which senses belong to which target word (None = no masking).
        # A buffer moves with .to(device) and is saved in the state_dict, but is not trained.
        self.register_buffer("sense_mask", sense_mask)

    def forward(self, x):
        # x: [B, 2k+2] = the 2k+1 word-id window, then the target word's index in the last column.
        # The index rides along in x so the loaders can stay (X, y) pairs, which fit() needs.
        text, target_id = x[:, :-1], x[:, -1]
        embedded = self.embedding(text)
        # out: [B, 2k+1, 2*hidden]
        out, (hidden, c_n) = self.lstm(embedded)
        # state at the target word -> logits [B, num_senses]
        logits = self.fc(out[:, self.k, :])

        if self.sense_mask is not None:
            # IF there is an invalid senses (mask is False)
            #   fill with -1e9 wherever the mask is False
            # -1e9: Big negative number keeps the loss finite whatever happens
            logits = logits.masked_fill(~self.sense_mask[target_id], -1e9) # type: ignore
        return logits

# Helper functions for saving and loading models
def _saveModel(model: WSDLSTM, path):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")

def _loadModel(model: WSDLSTM, path: Path, device: str = "mps") -> WSDLSTM:
    model.load_state_dict(torch.load(path, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded PyTorch Model State from {path}")
    return model

# def tokenize(text):
#     text = re.sub(r"<br\s*/?>", " ", text.lower())   # IMDB reviews contain <br /> tags
#     return re.findall(r"[a-z']+", text)

def buildVocab(examples, vocab_size=VOCAB_SIZE) -> dict:
    counts = Counter()
    common = []
    vocabdict = {}

    for example in examples:
        # count only the +-K window the model will actually see (tokens are already lowercased)
        i = example['target_idx']
        for token in example['tokens'][max(0, i - K):i + K + 1]:
            counts[token] += 1

    # reserve 0=<pad>, 1=<unk>
    mostcommonCounts = counts.most_common(vocab_size - 2)
    for w, _ in mostcommonCounts:
        common.append(w)

    for i, w in enumerate(common):
            vocabdict[w] = i +2                                 # +2 because 0 and 1 are taken by <pad> and <unk>

    return vocabdict

# def encode(text, vocab, max_len=MAX_LEN):
#     ids = []
#     tokenized = tokenize(text)
#     for t in tokenized:
#         ids.append(vocab.get(t, UNK))
#     # ---
#     ids = ids[:max_len]                                            # truncate
#     padding = [PAD] * (max_len - len(ids))                         # how many pads are needed
#     return padding + ids                                           # left-pad

def toDataLoader(examples, vocab, sense_space, shuffle: bool =False):

    # --- inputs: each row = the 2K+1 window of word ids, then the target word's index (last column)
    texts = []
    for example in examples:
        window = encodeWindow(example['tokens'], example['target_idx'], vocab, K)
        texts.append(window + [sense_space['target2idx'][example['target']]])
    textTensors = torch.tensor(texts, dtype=torch.long)

    # --- labels: integer index of the WordNet sense (torch.long, as CrossEntropyLoss needs)
    labels = []
    for example in examples:
        labels.append(sense_space['sense2idx'][example['synset']])
    labelTensors = torch.tensor(labels, dtype=torch.long)
    # ---

    dataset = TensorDataset(textTensors, labelTensors)
    data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    return data

def loadDataset(trainOrtest: bool = False):
    """
    Needs to return integer sense labels
    SemCor (sense-tagged corpus based on WordNet) dataset for this assignment.

    Ensure you have downloaded the semcor dataset (semcor_dataset does this itself)

    Returns one dict per occurrence of a target word, not one per sentence:
        {"target": "bank", "tokens": [...], "target_idx": 4, "synset": "bank.n.01", "sent_id": 812}
    True -> the training split (~80% of the sentences), False -> the test split.
    The split is by sentence and deterministic, so the two calls never overlap.
    """
    return semcor_dataset(TARGET_WORDS, train=trainOrtest)

def prepareData(data: bool = False):
    """prepareData
    This is the main data pipeline.
    Load the data apply the steps below and return the tokenized version
        Tokenize the text, 
        - (optional) remove stop words and punctuation 
        - convert tokens to integer sequences using a vocabulary, 
        - pad sequences to the same length.

    Keyword arguments:
    argument -- description
    Return: should return the DataLoader
    """
    train = loadDataset(True)
    test = loadDataset(False)

    # convert tokens to integer sequences using a vocabulary
    # pad sequences to the same length
    vocab = buildVocab(train)            # build from TRAIN only, reuse for test

    # label space + candidate mask come from WordNet, so train and test share them
    sense_space = buildSenseSpace(TARGET_WORDS)

    # Apply batch_size and shuffle them
    traindata = toDataLoader(train, vocab, sense_space, True)
    testdata = toDataLoader(test, vocab, sense_space, False)

    return traindata, testdata, vocab, sense_space


def initializeModel(model_cls: type[WSDLSTM], sense_space: dict):

    # Set the correct hardware
    device = get_device()
        # criterion = nn.BCEWithLogitsLoss()
    # output size = number of WordNet senses; K = window half-width (target sits at index K);
    # the mask limits each target word to its own senses
    sense_mask = torch.tensor(sense_space['mask'], dtype=torch.bool)
    model = model_cls(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, len(sense_space['senses']), K, sense_mask).to(device)

    # WSD is multi-class: raw logits [B, num_senses] + integer sense labels, so CrossEntropyLoss (not BCELoss)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, device, criterion, optimizer


def trainingLoop(model_cls: type[WSDLSTM]):
    """sumary_line
    
    list of dicts → (texts, labels) as tensors → TensorDataset → DataLoader → iterate in batches

    fit() (model_eval.py) runs the epochs: it trains, scores the test loader once after every
    epoch and redraws the loss / accuracy curves live. The curves are also saved to
    data/figures/<run_name>_curves.png.

    Keyword arguments:
    model_cls -- the type of the model that we want to train
    Return: model, test_loader, device, criterion, vocab, sense_space
    """
    train_loader, test_loader, vocab, sense_space = prepareData()

    # --- WSDLSTM ---
    model, device, criterion, optimizer = initializeModel(model_cls, sense_space)

    # Training Loop (CrossEntropyLoss on the sense logits, same loss for training and the test curve)
    fit(model, train_loader, test_loader, criterion, optimizer, device, EPOCHS,
        run_name=f"{DOC_FIGS_NAME}_{model_cls.__name__}")

    # Return model, test_loader, device and the loss function used for evaluation,
    # plus the vocab and sense space that are needed to use the model later
    return model, test_loader, device, criterion, vocab, sense_space

# def test(dataloader, model, loss_fn, device):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0.0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test:  accuracy {100 * correct:>5.1f}%  avg loss {test_loss:>8f}")
#     return correct, test_loss
       

def evaluateModel(model: WSDLSTM, test_loader: DataLoader, device: str):
    """evaluateModel
        
        Evaluate the model on the test set.
        Reports accuracy, confusion matrix, precision, recall and F1 (positive class = 'pos' = 1).
        Returns the metrics as a dict so Problem1 can compare models.

        Evaluate the model on the test set
        Report: accuracy, confusion matrix, precision, recall, and F1 score

        Perplexity:
            We use perplexity as a metric to evaluate the performance of our RNN/LSTM model for sentiment analysis.
            perplexity of sequence X:
            - Perplexity = 1: Perfect model (always predicts the right token with probability 1)
            - Perplexity = ∞ (infinity): Worst case (assigns probability 0 to true tokens)
            - Perplexity = |vocabulary| (size of vocabulary): Baseline (uniform distribution over all words)
            - Lower perplexity = better model


        wrap it in model.eval() and with torch.no_grad(): 
        Threshold the output with (model(x).squeeze(1) > 0.5).float() to get 0/1 predictions.

    Keyword arguments:
    argument -- description
    Return: return_description
    """
    model.eval()                                  # inference mode (apperently)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # torch.inference_mode()
    # torch.enable_grad()
    """
    counting tp/tn/fp/fn per batch then compute the metrices once at the end
        saves my sanity and memory   
    """
    with torch.no_grad():                         # no gradients needed, saves time and memory
        correct = 0
        total = 0
        for batchText, batchLabels in test_loader:
            batchText = batchText.to(device)
            batchLabels = batchLabels.to(device)

            probs = model(batchText).squeeze(1)  # [B, 1] -> [B], values in (0, 1)
            preds = (probs > 0.5).float()         # threshold -> hard 0.0 / 1.0

            
            tp += ((preds == 1) & (batchLabels == 1)).sum().item()
            tn += ((preds == 0) & (batchLabels == 0)).sum().item()
            fp += ((preds == 1) & (batchLabels == 0)).sum().item()
            fn += ((preds == 0) & (batchLabels == 1)).sum().item()

            # ---
            # Add an additional dimension for the sequence length
            batchText2 = batchText.unsqueeze(1)
            outputs = model(batchText2)
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            total += batchLabels.size(0)
            correct += (predicted == batchLabels).sum().item()
            # ---
        # ---
        accuracy = correct / total
        print(f'Acuracy: {100*accuracy:.4f} %')
        # ---
    
    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    modelname = type(model).__name__

    fancyString = f"""
        --- {modelname} test results ---
            Accuracy  : {accuracy:.4f}
            Precision : {precision:.4f}
            Recall    : {recall:.4f}
            F1 score  : {f1:.4f}

            Confusion matrix (rows = actual, cols = predicted):
                        neg   |   pos
                neg   {tn:6d} | {fp:6d}
                pos   {fn:6d} | {tp:6d}
    """
    
    print(fancyString)
    fancydict = {
                "model": modelname, 
                "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1, 
                "confusion_matrix": [[tn, fp], [fn, tp]]
    }
    
    return fancydict

def encodeWindow(tokens, target_idx, vocab, k=10):
    ids = lambda ws: [vocab.get(w, UNK) for w in ws]
    left  = ids(tokens[max(0, target_idx - k):target_idx])
    right = ids(tokens[target_idx + 1:target_idx + 1 + k])
    left  = [PAD] * (k - len(left)) + left
    right = right + [PAD] * (k - len(right))
    return left + ids([tokens[target_idx]]) + right      # length 2k+1, target at index k

def wordSensDisambiguation(model_cls: type[WSDLSTM]):
    """wordSensDisambiguation
    
    Keyword arguments:
    model_cls -- of type model class of WSDLSTM 
    Return: return_description
    """
    # 1-3. Data preparation, model architecture and training all happen inside trainingLoop
    model, test_loader, device, criterion, vocab, sense_space = trainingLoop(model_cls)

    _saveModel(model, MODEL_PATH)
    VOCAB_PATH.write_text(json.dumps(vocab))      # predicting later needs the same word -> id table

    # 4. Evaluate: one pass over the test set. Not evaluateAndPlot(): with 550 senses its confusion
    #    matrix is unreadable, so the per-word charts (evaluatePerGroup in Problem2) replace it.
    out = collectPredictions(model, test_loader, device, group_of=lambda X: X[:, -1])
    results = {"model": model_cls.__name__,
               **computeMetrics(out["y_true"], out["y_pred"], len(sense_space['senses']))}
    print(f"{results['model']} test: acc {results['accuracy']:.4f}  precision {results['precision']:.4f}  "
          f"recall {results['recall']:.4f}  F1 {results['f1']:.4f}  ({results['average']})")

    return results, out, {"model": model, "vocab": vocab, "device": device}


def printComparison(results: list):
    """The assignment's model comparison table: one row per model, baseline included."""
    print(f"\n{'Model':<16} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
    for r in results:
        print(f"{r['model']:<16} | {r['accuracy']:7.4f} | {r['precision']:7.4f} | {r['recall']:7.4f} | {r['f1']:7.4f}")
    print("(Prec / Rec / F1 are macro averages over the senses that occur; the baseline needs no training)")


# ---Requirement 5: predict the sense of a word in a sentence the user types---

def loadLSTM(sense_space: dict, device: str):
    """Rebuild the saved LSTM: same architecture + the vocab it was trained with."""
    vocab = json.loads(VOCAB_PATH.read_text())
    sense_mask = torch.tensor(sense_space['mask'], dtype=torch.bool)
    model = WSDLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, len(sense_space['senses']), K, sense_mask)
    return _loadModel(model, MODEL_PATH, device), vocab


def loadBert(device: str):
    model = wsd_bert.BertGlossPair().to(device)
    model.load_state_dict(torch.load(BERT_PATH, weights_only=True))
    model.eval()
    print(f"Loaded PyTorch Model State from {BERT_PATH}")
    return model, wsd_bert.loadTokenizer()


def tokenizeSentence(text: str) -> list:
    # lowercase words (keeping don't) and single punctuation marks, like the SemCor tokens
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text.lower())


def findTarget(tokens: list, target: str):
    """(index of the target in tokens, its WordNet lemma, how many times it occurs) or None.

    The exact word wins; otherwise any inflection of it ('banks', 'ran' for target 'run').
    """
    target = target.lower().strip()
    lemma = wn.morphy(target) or target
    hits = [i for i, t in enumerate(tokens) if t == target]
    if not hits:
        hits = [i for i, t in enumerate(tokens) if (wn.morphy(t) or t) == lemma]
    return (hits[0], lemma, len(hits)) if hits else None


def predictSense(sentence: str, target: str, device: str, lstm=None, bert=None, sense_space=None) -> dict:
    """Rank the WordNet senses of `target` in `sentence` with whichever models are given.

    lstm = (model, vocab), bert = (model, tokenizer). Returns {"found": False} when the target
    word is not in the sentence, else the tokens, index, lemma and a ranked
    [(synset, probability), ...] list per model (None when that model can't handle the word).
    """
    tokens = tokenizeSentence(sentence)
    found = findTarget(tokens, target)
    if found is None:
        return {"found": False, "tokens": tokens}
    idx, lemma, count = found
    out = {"found": True, "tokens": tokens, "idx": idx, "lemma": lemma, "count": count,
           "senses": len(wn.synsets(lemma)), "bert": None, "lstm": None}

    if bert is not None and out["senses"]:
        out["bert"] = wsd_bert.rankSenses(bert[0], bert[1], device, tokens, idx, lemma)

    # the LSTM only has outputs for the words it was trained on
    if lstm is not None and sense_space is not None and lemma in sense_space['target2idx']:
        model, vocab = lstm
        row = encodeWindow(tokens, idx, vocab, K) + [sense_space['target2idx'][lemma]]
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor([row], dtype=torch.long, device=device))[0], dim=0).cpu()
        own = sense_space['mask'][sense_space['target2idx'][lemma]]
        ranked = [(sense_space['senses'][i], probs[i].item()) for i, ok in enumerate(own) if ok]
        out["lstm"] = sorted(ranked, key=lambda p: -p[1])
    return out


def showPrediction(label: str, ranked, alternatives: int = 2):
    """Print the best sense with its WordNet gloss and example, then the runners-up."""
    if not ranked:
        print(f"\n{label}: (no prediction - this model has no output for that word)")
        return
    best, prob = ranked[0]
    synset = wn.synset(best)
    examples = synset.examples()
    print(f"\n{label}: {best}  ({prob:.0%} confident)")
    print(f"    gloss   : {synset.definition()}")
    print(f"    example : {examples[0] if examples else '(WordNet has no example sentence for this sense)'}")
    for name, p in ranked[1:1 + alternatives]:
        print(f"    also    : {name} ({p:.0%}) - {wn.synset(name).definition()}")


def interactive(device: str, lstm=None, bert=None, sense_space=None):
    """Prompt for a sentence and a target word until the sentence is left blank."""
    print("\n--- Word sense disambiguation: type a sentence and the word to disambiguate ---")
    print("(leave the sentence blank to quit)  e.g.  'She sat on the bank of the river.'  /  bank")
    while True:
        try:
            sentence = input("\nSentence: ").strip()
            if not sentence:
                break
            target = input("Target word: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        result = predictSense(sentence, target, device, lstm, bert, sense_space)
        if not result["found"]:
            print(f"'{target}' is not in that sentence (words seen: {', '.join(result['tokens'])})")
            continue
        if not result["senses"]:
            print(f"WordNet has no senses for '{result['lemma']}'.")
            continue

        note = f" ('{target}' appears {result['count']} times; using the first)" if result["count"] > 1 else ""
        print(f"\n'{result['tokens'][result['idx']]}' -> lemma '{result['lemma']}', "
              f"{result['senses']} WordNet senses{note}")
        if bert is not None:
            showPrediction("BERT", result["bert"])
        if lstm is not None:
            showPrediction("LSTM", result["lstm"])


# ---Problem2---
def Problem2(predict_only: bool = False, use_bert: bool = True):
    """
    Build a system that can determine the correct sense (meaning) of a word in context. 
        Words often have multiple meanings, and their correct interpretation depends on context. 
        For example, in the sentence, “He deposited money in the bank,” the word “bank” means a financial institution. 
        In the sentence, “She sat on the bank of the river,” the word “bank” means the river shore. 
        Your goal is to create a model that can automatically identify the correct sense of a word given its context in a sentence. 
        Use the SemCor (sense-tagged corpus based on WordNet) dataset for this assignment.
    
    1. Choose a set of ambiguous target words (“bank,” “bat,” “charge”). 
        Load the dataset and extract sentence-level examples for those words. 
        Tokenize, lowercase, and (optionally) lemmatize the texts as needed.
            Need to add extra target words

    2. Use WordNet to retrieve all possible senses of each target word. 
        Include glosses (definitions) and example usages for each sense.

    3. Use LSTM and pretrained contextual embeddings (BERT/WiC-style fine-tuning) models.

    4. Use accuracy, F1-score, or precision/recall depending on class distribution. 
        Compare the models' performances.

    5. Let the user input a sentence with a target word. 
        Display the predicted word sense with 'gloss' and 'example'.

    WSD needs the representation at the target word, and we need both sides of it. 
        ±k words around the target. Padding the edges makes target always at index k.


    multi-class classification
    given: sentenct "s" and target word "w" pick one of that word's WordNet senses (bank.n.01, bank.n.09, ...)
    no sigmoid
    Output layer: Linear(hidden, num_senses) with raw logits and no Sigmoid.
    Loss: CrossEntropyLoss instead of BCELoss. It applies softmax internally, so it needs raw logits.
    Labels: integer class indices (torch.long) instead of 0.0/1.0 floats.
    Metrics: argmax instead of > 0.5, and an NxN confusion matrix. 
            SemCor is heavily skewed toward each word's first sense, 
                so use use macro-F1 and compare against a “always predict the most frequent sense” baseline. 
    
    Accuracy is expected to be much higher.

    """
    device = get_device()
    sense_space = buildSenseSpace(TARGET_WORDS)
    lstm = bert = None

    if not predict_only:
        train_examples, test_examples = loadDataset(True), loadDataset(False)
        sense2idx = sense_space['sense2idx']
        # test labels and each test example's word, in the same order as test_loader (shuffle=False)
        y_true = np.array([sense2idx[e['synset']] for e in test_examples])
        groups = np.array([sense_space['target2idx'][e['target']] for e in test_examples])

        # Baseline: always the sense most often seen for that word in the training split.
        base_pred = np.array([sense2idx[s] for s in mostFrequentSense(train_examples, test_examples)])
        results = [{"model": "MFS baseline", **computeMetrics(y_true, base_pred, len(sense_space['senses']))}]
        predictions = {}                                             # model name -> predicted sense index per test row

        # LSTM (output = one logit per sense)
        lstm_results, lstm_out, trained = wordSensDisambiguation(WSDLSTM)
        assert np.array_equal(y_true, lstm_out["y_true"]), "test_loader rows are not in test_examples order"
        results.insert(0, lstm_results)
        predictions["WSDLSTM"] = lstm_out["y_pred"]
        lstm = (trained["model"], trained["vocab"])

        # BERT gloss-pair (output = one match score per (sentence, gloss) pair)
        if use_bert:
            model, tokenizer = wsd_bert.trainBert(train_examples, device, BERT_EPOCHS,
                                                  run_name=f"{DOC_FIGS_NAME}_BertGlossPair")
            torch.save(model.state_dict(), BERT_PATH)
            print(f"Saved PyTorch Model State to {BERT_PATH}")
            bert_pred = wsd_bert.predictSenseIdx(model, tokenizer, device, test_examples, sense2idx)
            results.insert(1, {"model": "BertGlossPair", **computeMetrics(y_true, bert_pred, len(sense_space['senses']))})
            predictions["BertGlossPair"] = bert_pred
            bert = (model, tokenizer)

        # Requirement 4: comparison table, with the baseline as the row to beat
        printComparison(results)

        # Per-word breakdown (replaces the unreadable 550 x 550 confusion matrix). The three required
        # words are shown first and get their own small confusion matrices over their own senses.
        for name, y_pred in predictions.items():
            evaluatePerGroup(y_true, y_pred, groups, sense_space['targets'], class_names=sense_space['senses'],
                             baseline_pred=base_pred, focus=REQUIRED_WORDS, run_name=f"{DOC_FIGS_NAME}_{name}")

        # One blocking call for every figure: the training curves and the evaluation charts
        plt.show()
    else:
        if MODEL_PATH.exists() and VOCAB_PATH.exists():
            lstm = loadLSTM(sense_space, device)
        if use_bert and BERT_PATH.exists():
            bert = loadBert(device)
        if lstm is None and bert is None:
            print("No saved models found. Run without --predict first to train and save them.")
            return

    # Requirement 5: let the user type a sentence + target word (skipped when not run from a terminal)
    if sys.stdin.isatty():
        interactive(device, lstm, bert, sense_space)



def main():
    parser = argparse.ArgumentParser(description="COMP 457 Problem 2: word sense disambiguation on SemCor")
    parser.add_argument("--predict", action="store_true",
                        help="skip training: load the saved models and go straight to the interactive prompt")
    parser.add_argument("--no-bert", action="store_true",
                        help="LSTM only (skips the BERT download and fine-tuning)")
    args = parser.parse_args()
    Problem2(predict_only=args.predict, use_bert=not args.no_bert) # type: igone
if __name__ == "__main__":
    main()