

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

from torchnlp.download import download_file_maybe_extract

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pth"

# Hyperparameters
VOCAB_SIZE = 1000       # Size of each dictionary / vocabulary
EMBEDDING_DIM = 100     # Dimension of word vectors
HIDDEN_DIM = 128        # Dimensions of RNN internal hidden state
BATCH_SIZE = 32
EPOCHS = 5

MAX_LEN = 200
PAD, UNK = 0, 1

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WSDLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_sense, k):
        super().__init__()
        self.k = k
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True) # bidirectional=True for ±k words
        self.fc = nn.Linear(2 * hidden_dim, num_sense)    # 2 * hidden dimension because of bidirectional
    
    def forward(self, text):
        # text: [B, 2k+1]
        embedded = self.embedding(text)
        # out: [B, 2k+1, 2*hidden]
        out, (hidden, c_n) = self.lstm(embedded)
        # state at the target word -> logits [B, num_senses]
        targetwordState = self.fc(out[:,self.k, :])
        return targetwordState

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

def tokenize(text):
    text = re.sub(r"<br\s*/?>", " ", text.lower())   # IMDB reviews contain <br /> tags
    return re.findall(r"[a-z']+", text)

def buildVocab(examples, vocab_size=VOCAB_SIZE) -> dict:
    counts = Counter()
    common = []
    vocabdict = {}

    for example in examples:
        tokens = tokenize(example['text'])
        for token in tokens:
            counts[token] += 1
    
    # reserve 0=<pad>, 1=<unk>
    mostcommonCounts = counts.most_common(vocab_size - 2)
    for w, _ in mostcommonCounts:
        common.append(w)

    for i, w in enumerate(common):
            vocabdict[w] = i +2                                 # +2 because 0 and 1 are taken by <pad> and <unk>

    return vocabdict

def encode(text, vocab, max_len=MAX_LEN):
    ids = []
    tokenized = tokenize(text)
    for t in tokenized:
        ids.append(vocab.get(t, UNK))
    # ---
    ids = ids[:max_len]                                            # truncate
    padding = [PAD] * (max_len - len(ids))                         # how many pads are needed
    return padding + ids                                           # left-pad

def toDataLoader(examples, vocab, shuffle: bool =False):

    # --- texts: list of dicts -> list of padded int lists -> tensor
    texts = []
    for example in examples:
        texts.append(encode(example['text'], vocab))
    textTensors = torch.tensor(texts,dtype=torch.long)

    # --- labels: 'pos' -> 1.0, 'neg' -> 0.0
    labels = []
    for example in examples:
        if example['sentiment'] == 'pos':
            labels.append(1.0)
        else:
            labels.append(0.0) 
    labelTensors = torch.tensor(labels, dtype=torch.float32)
    # ---

    dataset = TensorDataset(textTensors, labelTensors)
    data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    return data

def loadDataset(trainOrtest: bool = False):
    from torchnlp.datasets import imdb_dataset
    # Load the IMDB dataset for training then prints some examples
    listOfDicts=[{}]

    if trainOrtest:
        listOfDicts = imdb_dataset(directory=str(DATA_DIR), train=True)
    else:
        listOfDicts = imdb_dataset(directory=str(DATA_DIR), test=True)
    
    # print(dictionaries[0:2])
    """format of each test[i] is list containing i dictionaries
    {
          'text': 'For a movie that gets no respect there sure are a lot of memorable quotes...',
          'sentiment': 'pos'
        }
    """
    return listOfDicts

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

    # Apply batch_size and shuffle them
    traindata = toDataLoader(train, vocab, True)
    testdata = toDataLoader(test, vocab, False)

    return traindata, testdata, vocab


def initializeModel(model_cls: type[WSDLSTM]):

    # Set the correct hardware
    device = get_device()
        # criterion = nn.BCEWithLogitsLoss()
    model = model_cls(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, MAX_LEN, UNK).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, device, criterion, optimizer


def trainingLoop(model_cls: type[WSDLSTM]):
    """sumary_line
    
    list of dicts → (texts, labels) as tensors → TensorDataset → DataLoader → iterate in batches

    Keyword arguments:
    argument -- description
    Return: return_description
    """
    train_loader, test_loader, vocab = prepareData()

    # --- WSDLSTM ---
    model, device, criterion, optimizer = initializeModel(model_cls)
    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_text, batch_labels in train_loader:
            batch_text, batch_labels = batch_text.to(device), batch_labels.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            # predictions = model(batch_text)
            predictions = model(batch_text).squeeze(1)   # [B, 1] -> [B]

            # Calculate loss
            loss = criterion(predictions, batch_labels)
            
            # Backward pass and weight updates
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {total_loss / len(train_loader):.4f}")

    # Return model, test_loader and device
    return model, test_loader, device

       

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
    model, test_loader, device = trainingLoop(model_cls)
    
    path = MODEL_PATH
    _saveModel(model, path)

    # 4. Evaluate Model (returns the metrics dict so Problem1 can compare models)
    return evaluateModel(model, test_loader, device)


# ---Problem2---
def Problem2():
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
    results = []
    model_cls = (WSDLSTM)
    results.append(wordSensDisambiguation(model_cls))
    
    # Comparison table, printed once after all three models have been trained and evaluated
    print(f"{'Model':<6} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
    for result in results:
        print(f"{result['model']:<6} | {result['accuracy']:7.4f} | {result['precision']:7.4f} | {result['recall']:7.4f} | {result['f1']:7.4f}")



def example():
    """Source code for torchnlp.datasets.imdb
        https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/imdb.html
        
        Prerequisite:pip3 install pytorch-nlp

    """

    def imdb_dataset(directory='data/', train=False, test=False, train_directory='train', test_directory='test',extracted_name='aclImdb',
                    check_files=['aclImdb/README'], url='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', sentiments=['pos', 'neg']):
        """
        Load the IMDB dataset (Large Movie Review Dataset v1.0).

        This is a dataset for binary sentiment classification containing substantially more data than
        previous benchmark datasets. Provided a set of 25,000 highly polar movie reviews for
        training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text
        and already processed bag of words formats are provided.

        Note:
            The order examples are returned is not guaranteed due to ``iglob``.

        Reference: http://ai.stanford.edu/~amaas/data/sentiment/

        Args:
            directory (str, optional): Directory to cache the dataset.
            train (bool, optional): If to load the training split of the dataset.
            test (bool, optional): If to load the test split of the dataset.
            train_directory (str, optional): The directory of the training split.
            test_directory (str, optional): The directory of the test split.
            extracted_name (str, optional): Name of the extracted dataset directory.
            check_files (str, optional): Check if these files exist, then this download was successful.
            url (str, optional): URL of the dataset ``tar.gz`` file.
            sentiments (list of str, optional): Sentiments to load from the dataset.

        Returns:
            :class:`tuple` of :class:`iterable` or :class:`iterable`:
            Returns between one and all dataset splits (train, dev and test) depending on if their
            respective boolean argument is ``True``.

        Example:
            >>> from torchnlp.datasets import imdb_dataset  # doctest: +SKIP
            >>> train = imdb_dataset(train=True)  # doctest: +SKIP
            >>> train[0:2]  # doctest: +SKIP
            [{
            'text': 'For a movie that gets no respect there sure are a lot of memorable quotes...',
            'sentiment': 'pos'
            }, {
            'text': 'Bizarre horror movie filled with famous faces but stolen by Cristina Raines...',
            'sentiment': 'pos'
            }]
        """
        download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

        ret = []
        splits = [
            dir_ for (requested, dir_) in [(train, train_directory), (test, test_directory)]
            if requested
        ]
        for split_directory in splits:
            full_path = os.path.join(directory, extracted_name, split_directory)
            examples = []
            for sentiment in sentiments:
                for filename in glob.iglob(os.path.join(full_path, sentiment, '*.txt')):
                    with open(filename, 'r', encoding="utf-8") as f:
                        text = f.readline()
                    examples.append({
                        'text': text,
                        'sentiment': sentiment,
                    })
            ret.append(examples)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    imdb_dataset(directory=str(DATA_DIR), train=True)



def main():
    Problem2()

if __name__ == "__main__":
    main()