"""
COMP 457 - Assignment 4, Part 2, Problem 1
Language Model for Text Generation (next-word prediction)

What this script does, step by step:
  Step 1. Load the WikiText-2 dataset (Wikipedia articles).
  Step 2. Preprocess the text: lowercase, remove unwanted characters,
          split into words, build a vocabulary.
  Step 3. Make n-gram pairs:  5 words in  ->  the next word out.
  Step 4. Load pre-trained GloVe word embeddings.
  Step 5. Build two models: an LSTM and a Transformer.
  Step 6. Train both with cross-entropy loss and the Adam optimizer,
          tracking accuracy and perplexity.
  Step 7. Evaluate on the test set, show top-5 predictions, plot the
          training vs validation loss.
  Step 8. Interactive demo: you type a prompt, the models predict the next words.

How to run (from the repo root, with the Comp457 conda env active):
  python src/Assignment4_simple/problem1_text_generation.py
"""

import math
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # lets us save plots to a file without opening a window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Settings (change these to experiment)
# ---------------------------------------------------------------------------
CONTEXT_SIZE = 5          # how many words the model sees before guessing the next one
VOCAB_SIZE = 10000        # keep only the 10,000 most common words
EMBED_DIM = 100           # size of each GloVe word vector
HIDDEN_SIZE = 256         # size of the LSTM memory
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_TRAIN_WORDS = 1000000  # use the first 1 million training words so it trains in minutes

# Folders
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
REPO_FOLDER = os.path.dirname(os.path.dirname(THIS_FOLDER))
GLOVE_FILE = os.path.join(REPO_FOLDER, "data", "glove", "glove.6B.100d.txt")
OUTPUT_FOLDER = os.path.join(THIS_FOLDER, "outputs")

# Special tokens
PAD_TOKEN = "<pad>"   # used to fill up prompts that are shorter than CONTEXT_SIZE
UNK_TOKEN = "<unk>"   # used for any word that is not in our vocabulary
PAD_INDEX = 0
UNK_INDEX = 1

# Use the Apple GPU (mps) or an NVIDIA GPU (cuda) if one is available
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Step 1 + 2: Load and preprocess the text
# ---------------------------------------------------------------------------
def clean_text(text):
    """Lowercase the text, remove everything that is not a letter, split into words."""
    text = text.lower()
    # Replace any character that is NOT a-z or whitespace with a space.
    # This removes numbers, punctuation and symbols like "@-@" or "=".
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    return words


def load_words(split_name):
    """Load one split of WikiText-2 ('train', 'validation' or 'test') as one long list of words."""
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split_name)
    all_words = []
    for line in dataset["text"]:
        words = clean_text(line)
        all_words.extend(words)
    return all_words


def build_vocabulary(words):
    """Make two lookups: word -> number, and number -> word."""
    word_counts = Counter(words)
    most_common = word_counts.most_common(VOCAB_SIZE - 2)  # -2 leaves room for <pad> and <unk>

    index_to_word = [PAD_TOKEN, UNK_TOKEN]
    for word, count in most_common:
        index_to_word.append(word)

    word_to_index = {}
    for index, word in enumerate(index_to_word):
        word_to_index[word] = index

    return word_to_index, index_to_word


def words_to_indices(words, word_to_index):
    """Turn a list of words into a list of numbers. Unknown words become <unk>."""
    indices = []
    for word in words:
        if word in word_to_index:
            indices.append(word_to_index[word])
        else:
            indices.append(UNK_INDEX)
    return indices


# ---------------------------------------------------------------------------
# Step 3: Create n-gram input/output pairs
# ---------------------------------------------------------------------------
def make_ngram_pairs(indices):
    """
    Slide a window over the text.
    Example with CONTEXT_SIZE = 5:
        text:    the cat sat on the mat today
        input:   [the, cat, sat, on, the]   -> target: mat
        input:   [cat, sat, on, the, mat]   -> target: today
    """
    inputs = []
    targets = []
    for i in range(len(indices) - CONTEXT_SIZE):
        context = indices[i : i + CONTEXT_SIZE]
        next_word = indices[i + CONTEXT_SIZE]
        # Skip examples where the answer is <unk>: "predict an unknown word" teaches nothing useful
        if next_word != UNK_INDEX:
            inputs.append(context)
            targets.append(next_word)
    return torch.tensor(inputs), torch.tensor(targets)


# ---------------------------------------------------------------------------
# Step 4: Pre-trained GloVe embeddings
# ---------------------------------------------------------------------------
def load_glove_embeddings(index_to_word):
    """
    Build a matrix with one row per word in our vocabulary.
    If GloVe knows the word, use its pre-trained vector.
    If not, keep a small random vector (the model will learn it).
    """
    print("Loading GloVe vectors from", GLOVE_FILE)
    glove = {}
    with open(GLOVE_FILE, encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = []
            for number in parts[1:]:
                vector.append(float(number))
            glove[word] = vector

    embedding_matrix = torch.randn(len(index_to_word), EMBED_DIM) * 0.1
    found = 0
    for index, word in enumerate(index_to_word):
        if word in glove:
            embedding_matrix[index] = torch.tensor(glove[word])
            found = found + 1
    embedding_matrix[PAD_INDEX] = torch.zeros(EMBED_DIM)

    print("GloVe found", found, "of", len(index_to_word), "vocabulary words")
    return embedding_matrix


# ---------------------------------------------------------------------------
# Step 5: The two models
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """
    words -> GloVe embedding -> LSTM -> last hidden state -> Linear -> score for every word
    The LSTM reads the 5 words one at a time, left to right, and keeps a "memory".
    After the 5th word, its memory is used to guess the next word.
    """

    def __init__(self, embedding_matrix):
        super().__init__()
        vocab_size = embedding_matrix.shape[0]
        # freeze=False means the GloVe vectors are fine-tuned during training
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(input_size=EMBED_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(self, word_indices):
        # word_indices shape: (batch, 5)
        embedded = self.embedding(word_indices)   # (batch, 5, 100)
        lstm_output, _ = self.lstm(embedded)      # (batch, 5, 256)
        last_step = lstm_output[:, -1, :]         # (batch, 256) - memory after the last word
        scores = self.output_layer(last_step)     # (batch, 10000) - one score per vocab word
        return scores


class TransformerModel(nn.Module):
    """
    words -> GloVe embedding + position embedding -> Transformer encoder -> last position -> Linear
    Unlike the LSTM, the Transformer looks at all 5 words at once using self-attention.
    Attention has no sense of word order, so we add a learned "position embedding"
    that tells the model which word is 1st, 2nd, 3rd, ...
    """

    def __init__(self, embedding_matrix):
        super().__init__()
        vocab_size = embedding_matrix.shape[0]
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.position_embedding = nn.Embedding(CONTEXT_SIZE, EMBED_DIM)
        one_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,       # size of each word vector (100)
            nhead=4,                 # 4 attention heads (100 / 4 = 25 numbers per head)
            dim_feedforward=256,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(one_layer, num_layers=2)
        self.output_layer = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, word_indices):
        # word_indices shape: (batch, 5)
        positions = torch.arange(CONTEXT_SIZE, device=word_indices.device)  # [0, 1, 2, 3, 4]
        embedded = self.embedding(word_indices) + self.position_embedding(positions)  # (batch, 5, 100)
        transformer_output = self.transformer(embedded)  # (batch, 5, 100)
        last_step = transformer_output[:, -1, :]        # (batch, 100)
        scores = self.output_layer(last_step)           # (batch, 10000)
        return scores


# ---------------------------------------------------------------------------
# Step 6: Training and evaluation
# ---------------------------------------------------------------------------
def run_one_epoch(model, data_loader, loss_function, optimizer=None):
    """
    Go through the data once.
    If an optimizer is given, the model learns (training).
    If optimizer is None, we only measure (validation / test).
    Returns average loss, accuracy and top-5 accuracy.
    """
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_top5_correct = 0
    total_examples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        if optimizer is not None:
            scores = model(inputs)
            loss = loss_function(scores, targets)
            optimizer.zero_grad()   # clear old gradients
            loss.backward()         # compute new gradients
            optimizer.step()        # update the weights
        else:
            with torch.no_grad():   # no gradients needed when only measuring
                scores = model(inputs)
                loss = loss_function(scores, targets)

        batch_size = targets.shape[0]
        total_loss = total_loss + loss.item() * batch_size

        # Top-1 accuracy: was the highest-scoring word the right one?
        predictions = scores.argmax(dim=1)
        total_correct = total_correct + (predictions == targets).sum().item()

        # Top-5 accuracy: was the right word anywhere in the 5 highest scores?
        top5 = scores.topk(5, dim=1).indices                  # (batch, 5)
        target_column = targets.unsqueeze(1)                  # (batch, 1)
        total_top5_correct = total_top5_correct + (top5 == target_column).any(dim=1).sum().item()

        total_examples = total_examples + batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    top5_accuracy = total_top5_correct / total_examples
    return average_loss, accuracy, top5_accuracy


def train_model(model, model_name, train_loader, val_loader):
    """Train the model for EPOCHS epochs and remember the loss/accuracy/perplexity of each epoch."""
    loss_function = nn.CrossEntropyLoss()    # categorical cross-entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_perplexity": [], "val_perplexity": [],
    }

    print("\n========== Training", model_name, "==========")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, _ = run_one_epoch(model, train_loader, loss_function, optimizer)
        val_loss, val_acc, _ = run_one_epoch(model, val_loader, loss_function)

        # Perplexity = e^loss. Roughly: "how many words is the model unsure between?"
        train_perplexity = math.exp(train_loss)
        val_perplexity = math.exp(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_perplexity"].append(train_perplexity)
        history["val_perplexity"].append(val_perplexity)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train loss {train_loss:.3f}, acc {train_acc:.3f}, ppl {train_perplexity:.1f} | "
              f"val loss {val_loss:.3f}, acc {val_acc:.3f}, ppl {val_perplexity:.1f}")

    return history


# ---------------------------------------------------------------------------
# Step 7: Predictions and plots
# ---------------------------------------------------------------------------
def prompt_to_tensor(prompt_words, word_to_index):
    """Turn a list of words into a (1, CONTEXT_SIZE) tensor the model can read."""
    indices = words_to_indices(prompt_words, word_to_index)
    indices = indices[-CONTEXT_SIZE:]   # keep only the last 5 words
    # If the prompt is shorter than 5 words, add <pad> at the front
    while len(indices) < CONTEXT_SIZE:
        indices.insert(0, PAD_INDEX)
    return torch.tensor([indices]).to(DEVICE)


def top_k_words(model, prompt_words, word_to_index, index_to_word, k=5):
    """Return the k most likely next words and their probabilities."""
    model.eval()
    input_tensor = prompt_to_tensor(prompt_words, word_to_index)
    with torch.no_grad():
        scores = model(input_tensor)[0]               # (vocab_size,)
    probabilities = torch.softmax(scores, dim=0)      # turn scores into probabilities that sum to 1
    probabilities[PAD_INDEX] = 0                      # never suggest <pad> or <unk>
    probabilities[UNK_INDEX] = 0
    top = probabilities.topk(k)

    results = []
    for probability, index in zip(top.values, top.indices):
        word = index_to_word[index.item()]
        results.append((word, probability.item()))
    return results


def generate_text(model, prompt_words, word_to_index, index_to_word, number_of_words=5):
    """Predict one word, add it to the prompt, predict the next one, ... (greedy decoding)."""
    words = list(prompt_words)
    for step in range(number_of_words):
        best = top_k_words(model, words, word_to_index, index_to_word, k=1)
        next_word = best[0][0]
        words.append(next_word)
    return " ".join(words)


def plot_curves(lstm_history, transformer_history):
    """Save a picture with training vs validation loss, perplexity and accuracy for both models."""
    epochs = range(1, EPOCHS + 1)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, lstm_history["train_loss"], "b-o", label="LSTM train")
    axes[0].plot(epochs, lstm_history["val_loss"], "b--o", label="LSTM validation")
    axes[0].plot(epochs, transformer_history["train_loss"], "r-s", label="Transformer train")
    axes[0].plot(epochs, transformer_history["val_loss"], "r--s", label="Transformer validation")
    axes[0].set_title("Cross-entropy loss")

    axes[1].plot(epochs, lstm_history["train_perplexity"], "b-o", label="LSTM train")
    axes[1].plot(epochs, lstm_history["val_perplexity"], "b--o", label="LSTM validation")
    axes[1].plot(epochs, transformer_history["train_perplexity"], "r-s", label="Transformer train")
    axes[1].plot(epochs, transformer_history["val_perplexity"], "r--s", label="Transformer validation")
    axes[1].set_title("Perplexity (lower is better)")

    axes[2].plot(epochs, lstm_history["train_acc"], "b-o", label="LSTM train")
    axes[2].plot(epochs, lstm_history["val_acc"], "b--o", label="LSTM validation")
    axes[2].plot(epochs, transformer_history["train_acc"], "r-s", label="Transformer train")
    axes[2].plot(epochs, transformer_history["val_acc"], "r--s", label="Transformer validation")
    axes[2].set_title("Next-word accuracy")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, "problem1_training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Saved training curves to", save_path)


# ---------------------------------------------------------------------------
# Step 8: Interactive demo
# ---------------------------------------------------------------------------
def interactive_demo(models, word_to_index, index_to_word):
    print("\n========== Interactive demo ==========")
    print("Type a prompt like: the weather is")
    print("Type quit to stop.")
    while True:
        prompt = input("\nYour prompt: ")
        if prompt.strip().lower() == "quit":
            break
        prompt_words = clean_text(prompt)
        if len(prompt_words) == 0:
            continue
        for model_name, model in models.items():
            suggestions = top_k_words(model, prompt_words, word_to_index, index_to_word, k=5)
            sentence = generate_text(model, prompt_words, word_to_index, index_to_word, number_of_words=5)
            print(f"  {model_name} top-5 next words: {suggestions_to_text(suggestions)}")
            print(f"  {model_name} continues: {sentence}")


def suggestions_to_text(suggestions):
    """Format [('the', 0.12), ('a', 0.05)] as 'the (0.12), a (0.05)'."""
    parts = []
    for word, probability in suggestions:
        parts.append(f"{word} ({probability:.2f})")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    torch.manual_seed(42)
    print("Using device:", DEVICE)

    # Step 1 + 2: load and clean the text
    train_words = load_words("train")[:MAX_TRAIN_WORDS]
    val_words = load_words("validation")
    test_words = load_words("test")
    print("Words -> train:", len(train_words), "| validation:", len(val_words), "| test:", len(test_words))
    print("First 20 cleaned words:", " ".join(train_words[:20]))

    # The vocabulary is built from the training words only
    word_to_index, index_to_word = build_vocabulary(train_words)
    print("Vocabulary size:", len(index_to_word))

    # Step 3: n-gram pairs
    train_x, train_y = make_ngram_pairs(words_to_indices(train_words, word_to_index))
    val_x, val_y = make_ngram_pairs(words_to_indices(val_words, word_to_index))
    test_x, test_y = make_ngram_pairs(words_to_indices(test_words, word_to_index))
    print("N-gram pairs -> train:", len(train_y), "| validation:", len(val_y), "| test:", len(test_y))

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE)

    # Step 4: GloVe
    embedding_matrix = load_glove_embeddings(index_to_word)

    # Step 5 + 6: build and train both models
    lstm_model = LSTMModel(embedding_matrix.clone()).to(DEVICE)
    lstm_history = train_model(lstm_model, "LSTM", train_loader, val_loader)

    transformer_model = TransformerModel(embedding_matrix.clone()).to(DEVICE)
    transformer_history = train_model(transformer_model, "Transformer", train_loader, val_loader)

    models = {"LSTM": lstm_model, "Transformer": transformer_model}

    # Step 7a: test-set results
    print("\n========== Test set results ==========")
    loss_function = nn.CrossEntropyLoss()
    for model_name, model in models.items():
        test_loss, test_acc, test_top5 = run_one_epoch(model, test_loader, loss_function)
        print(f"{model_name}: loss {test_loss:.3f} | perplexity {math.exp(test_loss):.1f} | "
              f"top-1 accuracy {test_acc:.3f} | top-5 accuracy {test_top5:.3f}")

    # Step 7b: top-5 predictions for a few real test inputs
    print("\n========== Top-5 predictions on test examples ==========")
    example_positions = [100, 2000, 15000, 40000]
    for position in example_positions:
        context = test_words[position : position + CONTEXT_SIZE]
        true_next_word = test_words[position + CONTEXT_SIZE]
        print(f"\nContext: '{' '.join(context)}'   (true next word: '{true_next_word}')")
        for model_name, model in models.items():
            suggestions = top_k_words(model, context, word_to_index, index_to_word, k=5)
            print(f"  {model_name}: {suggestions_to_text(suggestions)}")

    # Step 7c: plots
    plot_curves(lstm_history, transformer_history)

    # Step 8: interactive demo
    interactive_demo(models, word_to_index, index_to_word)


if __name__ == "__main__":
    main()
