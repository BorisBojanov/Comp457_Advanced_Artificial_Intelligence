"""
COMP 457 - Assignment 4, Part 2, Problem 2
Image Captioning on Flickr8k

What this script does, step by step:
  Step 1. Preprocess captions: lowercase, remove punctuation, add <start>/<end>,
          build a vocabulary, turn words into numbers, pad to the same length.
  Step 2. Preprocess images (resize + normalize) and run them through a
          pre-trained CNN (ResNet50 and InceptionV3) to get one 2048-number
          "feature vector" per image. This is done once and saved to disk.
  Step 3. Two decoders turn an image feature vector into a sentence:
          an LSTM decoder and a Transformer decoder.
  Step 4. Train all 4 combinations (2 CNNs x 2 decoders), then evaluate with
          BLEU scores and by looking at generated captions.
  Step 5. Compare LSTM vs Transformer in a results table.

Data needed (inside the repo's data/flickr8k folder):
  Flicker8k_Dataset/          <- the 8,091 .jpg images
  Flickr8k.token.txt          <- 5 captions per image
  Flickr_8k.trainImages.txt   <- 6,000 training image names
  Flickr_8k.devImages.txt     <- 1,000 validation image names
  Flickr_8k.testImages.txt    <- 1,000 test image names
  (from https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)

How to run (from the repo root, with the Comp457 conda env active):
  python src/Assignment4_simple/problem2_image_captioning.py
"""

import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MAX_CAPTION_LENGTH = 22   # <start> + up to 20 words + <end>
MIN_WORD_COUNT = 5        # words that appear fewer than 5 times become <unk>
EMBED_SIZE = 256          # size of each word vector
HIDDEN_SIZE = 512         # size of the LSTM memory
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# Folders
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
REPO_FOLDER = os.path.dirname(os.path.dirname(THIS_FOLDER))
DATA_FOLDER = os.path.join(REPO_FOLDER, "data", "flickr8k")
IMAGE_FOLDER = os.path.join(DATA_FOLDER, "Flicker8k_Dataset")  # (sic) the zip spells it "Flicker"
OUTPUT_FOLDER = os.path.join(THIS_FOLDER, "outputs")

# Special tokens
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
PAD_INDEX = 0
START_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Step 1: Caption preprocessing
# ---------------------------------------------------------------------------
def clean_caption(caption):
    """'A dog runs, fast!' -> ['a', 'dog', 'runs', 'fast']"""
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", " ", caption)   # keep only letters and spaces
    words = caption.split()
    return words


def load_all_captions():
    """Read Flickr8k.token.txt into a dictionary: image name -> list of 5 cleaned captions."""
    captions = {}
    token_file = os.path.join(DATA_FOLDER, "Flickr8k.token.txt")
    with open(token_file, encoding="utf-8") as file:
        for line in file:
            # Each line looks like: 1000268201_693b08cb0e.jpg#0<TAB>A child in a pink dress ...
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            image_name = parts[0].split("#")[0]
            words = clean_caption(parts[1])
            if image_name not in captions:
                captions[image_name] = []
            captions[image_name].append(words)
    return captions


def load_image_names(file_name):
    """Read one of the split files (train / dev / test) into a list of image names."""
    names = []
    with open(os.path.join(DATA_FOLDER, file_name), encoding="utf-8") as file:
        for line in file:
            name = line.strip()
            if name != "":
                names.append(name)
    return names


def build_vocabulary(train_names, all_captions):
    """Count words in the TRAINING captions and keep the ones seen at least MIN_WORD_COUNT times."""
    counts = Counter()
    for name in train_names:
        for words in all_captions[name]:
            counts.update(words)

    index_to_word = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
    for word, count in counts.most_common():
        if count >= MIN_WORD_COUNT:
            index_to_word.append(word)

    word_to_index = {}
    for index, word in enumerate(index_to_word):
        word_to_index[word] = index
    return word_to_index, index_to_word


def caption_to_indices(words, word_to_index):
    """
    ['a', 'dog', 'runs'] -> [<start>, a, dog, runs, <end>, <pad>, <pad>, ...] as numbers.
    Every caption ends up exactly MAX_CAPTION_LENGTH long (padding).
    """
    words = words[: MAX_CAPTION_LENGTH - 2]   # leave room for <start> and <end>
    indices = [START_INDEX]
    for word in words:
        if word in word_to_index:
            indices.append(word_to_index[word])
        else:
            indices.append(UNK_INDEX)
    indices.append(END_INDEX)
    while len(indices) < MAX_CAPTION_LENGTH:
        indices.append(PAD_INDEX)
    return indices


def make_caption_tensors(image_names, all_captions, word_to_index):
    """
    One training example per caption (so 5 per image).
    Returns two tensors:
      image_ids: which image (position in image_names) the caption belongs to
      captions:  the padded caption as numbers
    """
    image_ids = []
    captions = []
    for image_id, name in enumerate(image_names):
        for words in all_captions[name]:
            image_ids.append(image_id)
            captions.append(caption_to_indices(words, word_to_index))
    return torch.tensor(image_ids), torch.tensor(captions)


# ---------------------------------------------------------------------------
# Step 2: Image preprocessing + pre-trained CNN feature extraction
# ---------------------------------------------------------------------------
class ImageDataset(Dataset):
    """Loads images from disk one by one and applies resize + normalize."""

    def __init__(self, image_names, image_size):
        self.image_names = image_names
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # The CNNs were trained on ImageNet images normalized with these numbers
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        path = os.path.join(IMAGE_FOLDER, self.image_names[index])
        image = Image.open(path).convert("RGB")
        return self.transform(image)


def load_cnn(cnn_name):
    """
    Load a pre-trained CNN and remove its last layer (the 1000 ImageNet classes).
    What is left outputs a 2048-number vector that describes the image.
    """
    if cnn_name == "ResNet50":
        cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        image_size = 224
    else:
        cnn = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        image_size = 299
    cnn.fc = nn.Identity()   # "do nothing" layer instead of the classifier
    cnn.eval()               # evaluation mode: we do NOT train the CNN
    return cnn, image_size


def extract_features(cnn_name, image_names):
    """Run every image through the CNN once and save the features, so training is fast later."""
    save_path = os.path.join(DATA_FOLDER, f"features_{cnn_name}.pt")
    if os.path.exists(save_path):
        print("Loading saved", cnn_name, "features from", save_path)
        return torch.load(save_path)

    print("Extracting", cnn_name, "features for", len(image_names), "images (only happens once)...")
    cnn, image_size = load_cnn(cnn_name)
    cnn = cnn.to(DEVICE)
    loader = DataLoader(ImageDataset(image_names, image_size), batch_size=64)

    feature_batches = []
    with torch.no_grad():
        for batch_number, images in enumerate(loader):
            features = cnn(images.to(DEVICE))       # (batch, 2048)
            feature_batches.append(features.cpu())
            if batch_number % 20 == 0:
                print(f"  batch {batch_number}/{len(loader)}")

    all_features = torch.cat(feature_batches)      # (number of images, 2048)
    # Store as a dictionary so we can look features up by image name
    features_by_name = {}
    for index, name in enumerate(image_names):
        features_by_name[name] = all_features[index]
    torch.save(features_by_name, save_path)
    return features_by_name


def stack_features(features_by_name, image_names):
    """Make one (number of images, 2048) tensor in the same order as image_names."""
    rows = []
    for name in image_names:
        rows.append(features_by_name[name])
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# Step 3: The two decoders
# ---------------------------------------------------------------------------
# Both decoders use "teacher forcing" during training:
#   input  = caption without its last token:  <start> a dog runs
#   target = caption without its first token:  a dog runs <end>
# so at every position the model learns to predict the NEXT word.

class LSTMDecoder(nn.Module):
    """
    The image feature becomes the LSTM's starting memory (hidden state and cell state).
    The LSTM then reads the caption word by word and predicts the next word each time.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.image_to_hidden = nn.Linear(2048, HIDDEN_SIZE)
        self.image_to_cell = nn.Linear(2048, HIDDEN_SIZE)
        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE)
        self.lstm = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(self, image_features, caption_input):
        # image_features: (batch, 2048)   caption_input: (batch, length)
        # The LSTM wants its starting state shaped (num_layers=1, batch, hidden)
        hidden = torch.tanh(self.image_to_hidden(image_features)).unsqueeze(0)
        cell = torch.tanh(self.image_to_cell(image_features)).unsqueeze(0)

        embedded = self.embedding(caption_input)                    # (batch, length, 256)
        lstm_output, _ = self.lstm(embedded, (hidden, cell))        # (batch, length, 512)
        scores = self.output_layer(self.dropout(lstm_output))       # (batch, length, vocab)
        return scores


class TransformerDecoder(nn.Module):
    """
    The caption words attend to each other (masked self-attention: a word can only
    look at the words BEFORE it), and to the image (cross-attention).
    The image is given to the Transformer as a "memory" of length 1.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.image_to_memory = nn.Linear(2048, EMBED_SIZE)
        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE)
        self.position_embedding = nn.Embedding(MAX_CAPTION_LENGTH, EMBED_SIZE)
        one_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_SIZE,
            nhead=8,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(one_layer, num_layers=2)
        self.output_layer = nn.Linear(EMBED_SIZE, vocab_size)

    def forward(self, image_features, caption_input):
        length = caption_input.shape[1]

        # The image as a sequence of length 1: (batch, 1, 256)
        memory = self.image_to_memory(image_features).unsqueeze(1)

        # Word embedding + position embedding: (batch, length, 256)
        positions = torch.arange(length, device=caption_input.device)
        embedded = self.embedding(caption_input) + self.position_embedding(positions)

        # Causal mask: stops position i from looking at words i+1, i+2, ... (no cheating)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length, device=caption_input.device)

        output = self.transformer(tgt=embedded, memory=memory, tgt_mask=causal_mask)
        scores = self.output_layer(output)    # (batch, length, vocab)
        return scores


# ---------------------------------------------------------------------------
# Step 4: Training, caption generation and BLEU
# ---------------------------------------------------------------------------
def run_one_epoch(decoder, data_loader, features, loss_function, optimizer=None):
    """One pass through the captions. Trains if an optimizer is given, otherwise only measures."""
    if optimizer is not None:
        decoder.train()
    else:
        decoder.eval()

    total_loss = 0.0
    number_of_batches = 0
    for image_ids, captions in data_loader:
        image_features = features[image_ids].to(DEVICE)
        captions = captions.to(DEVICE)
        caption_input = captions[:, :-1]     # <start> a dog runs <end> <pad> ...  minus last
        caption_target = captions[:, 1:]     #         a dog runs <end> <pad> ...  minus first

        if optimizer is not None:
            scores = decoder(image_features, caption_input)
            # CrossEntropyLoss wants (N, vocab) and (N,), so flatten batch and length together
            loss = loss_function(scores.reshape(-1, scores.shape[2]), caption_target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                scores = decoder(image_features, caption_input)
                loss = loss_function(scores.reshape(-1, scores.shape[2]), caption_target.reshape(-1))

        total_loss = total_loss + loss.item()
        number_of_batches = number_of_batches + 1

    return total_loss / number_of_batches


def train_decoder(decoder, run_name, train_loader, val_loader, train_features, val_features):
    # ignore_index=PAD_INDEX: don't count the <pad> positions in the loss
    loss_function = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": []}

    print("\n========== Training", run_name, "==========")
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_one_epoch(decoder, train_loader, train_features, loss_function, optimizer)
        val_loss = run_one_epoch(decoder, val_loader, val_features, loss_function)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}/{EPOCHS} | train loss {train_loss:.3f} | val loss {val_loss:.3f}")
    return history


def generate_captions(decoder, features, index_to_word):
    """
    Greedy decoding for many images at once:
      start with <start>, ask the model for the most likely next word, add it,
      ask again, ... until MAX_CAPTION_LENGTH words. Then cut each caption at <end>.
    """
    decoder.eval()
    all_captions = []
    for start in range(0, len(features), 100):
        image_features = features[start : start + 100].to(DEVICE)
        batch_size = image_features.shape[0]
        words_so_far = torch.full((batch_size, 1), START_INDEX, device=DEVICE)  # (batch, 1)

        with torch.no_grad():
            for step in range(MAX_CAPTION_LENGTH - 1):
                scores = decoder(image_features, words_so_far)    # (batch, length, vocab)
                next_words = scores[:, -1, :].argmax(dim=1)       # best word at the last position
                words_so_far = torch.cat([words_so_far, next_words.unsqueeze(1)], dim=1)

        # Turn numbers back into words, stopping at <end>
        for row in words_so_far.tolist():
            caption = []
            for index in row[1:]:          # skip the <start> token
                if index == END_INDEX:
                    break
                caption.append(index_to_word[index])
            all_captions.append(caption)
    return all_captions


def compute_bleu(generated_captions, image_names, all_captions):
    """
    BLEU compares the generated caption with the 5 human captions for the same image.
    BLEU-1 counts matching single words, BLEU-4 counts matching runs of up to 4 words.
    """
    references = []
    for name in image_names:
        references.append(all_captions[name])   # the 5 human captions (lists of words)

    smoothing = SmoothingFunction().method1     # avoids a score of 0 when no 4-word match exists
    bleu1 = corpus_bleu(references, generated_captions, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, generated_captions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, generated_captions, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothing)
    return bleu1, bleu2, bleu4


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_loss_curves(histories):
    epochs = range(1, EPOCHS + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for run_name, history in histories.items():
        axes[0].plot(epochs, history["train_loss"], marker="o", label=run_name)
        axes[1].plot(epochs, history["val_loss"], marker="o", label=run_name)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss")
        axis.grid(True, alpha=0.3)
        axis.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, "problem2_training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Saved training curves to", save_path)


def plot_example_captions(test_names, all_captions, generated, number_of_examples=6):
    """Show some test images with one human caption and every model's caption (visual inspection)."""
    figure, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i in range(number_of_examples):
        image_index = i * 150      # spread the examples out over the test set
        name = test_names[image_index]
        image = Image.open(os.path.join(IMAGE_FOLDER, name)).convert("RGB")

        text = "Human: " + " ".join(all_captions[name][0])
        for run_name, captions in generated.items():
            text = text + "\n" + run_name + ": " + " ".join(captions[image_index])

        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(text, fontsize=8, loc="left")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, "problem2_example_captions.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print("Saved example captions to", save_path)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    torch.manual_seed(42)
    print("Using device:", DEVICE)

    # Step 1: captions
    all_captions = load_all_captions()
    train_names = load_image_names("Flickr_8k.trainImages.txt")
    val_names = load_image_names("Flickr_8k.devImages.txt")
    test_names = load_image_names("Flickr_8k.testImages.txt")
    print("Images -> train:", len(train_names), "| validation:", len(val_names), "| test:", len(test_names))

    word_to_index, index_to_word = build_vocabulary(train_names, all_captions)
    vocab_size = len(index_to_word)
    print("Vocabulary size:", vocab_size)

    example = all_captions[train_names[0]][0]
    print("Example caption:", " ".join(example))
    print("As numbers:     ", caption_to_indices(example, word_to_index))

    train_ids, train_caps = make_caption_tensors(train_names, all_captions, word_to_index)
    val_ids, val_caps = make_caption_tensors(val_names, all_captions, word_to_index)
    train_loader = DataLoader(TensorDataset(train_ids, train_caps), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_ids, val_caps), batch_size=BATCH_SIZE)

    every_image = train_names + val_names + test_names
    histories = {}
    generated = {}
    results = []

    for cnn_name in ["ResNet50", "InceptionV3"]:
        # Step 2: image features from the pre-trained CNN
        features_by_name = extract_features(cnn_name, every_image)
        train_features = stack_features(features_by_name, train_names)
        val_features = stack_features(features_by_name, val_names)
        test_features = stack_features(features_by_name, test_names)

        for decoder_name in ["LSTM", "Transformer"]:
            # Step 3: build the decoder
            torch.manual_seed(42)
            if decoder_name == "LSTM":
                decoder = LSTMDecoder(vocab_size).to(DEVICE)
            else:
                decoder = TransformerDecoder(vocab_size).to(DEVICE)
            run_name = cnn_name + " + " + decoder_name

            # Step 4: train, then caption the test images and score them
            histories[run_name] = train_decoder(decoder, run_name, train_loader, val_loader,
                                                train_features, val_features)
            test_captions = generate_captions(decoder, test_features, index_to_word)
            generated[run_name] = test_captions
            bleu1, bleu2, bleu4 = compute_bleu(test_captions, test_names, all_captions)
            results.append((run_name, histories[run_name]["val_loss"][-1], bleu1, bleu2, bleu4))
            print(f"{run_name} TEST  BLEU-1 {bleu1:.3f} | BLEU-2 {bleu2:.3f} | BLEU-4 {bleu4:.3f}")
            print("  first test caption:", " ".join(test_captions[0]))

    # Step 5: comparison table
    table_lines = []
    table_lines.append(f"{'Model':28s} {'Val loss':>9s} {'BLEU-1':>8s} {'BLEU-2':>8s} {'BLEU-4':>8s}")
    for run_name, val_loss, bleu1, bleu2, bleu4 in results:
        table_lines.append(f"{run_name:28s} {val_loss:9.3f} {bleu1:8.3f} {bleu2:8.3f} {bleu4:8.3f}")
    table = "\n".join(table_lines)
    print("\n========== Results on the test set ==========")
    print(table)
    with open(os.path.join(OUTPUT_FOLDER, "problem2_results.txt"), "w") as file:
        file.write(table + "\n")

    plot_loss_curves(histories)
    plot_example_captions(test_names, all_captions, generated)


if __name__ == "__main__":
    main()
