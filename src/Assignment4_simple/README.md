# Assignment 4, Part 2: simple versions

These two scripts each stand on their own (no `model_eval.py` needed). Every step is written
out with plain `for` loops and `if` statements so you can follow it from top to bottom.

```bash
conda activate Comp457
python src/Assignment4_simple/problem1_text_generation.py    # about 10-15 min on the Mac
python src/Assignment4_simple/problem2_image_captioning.py   # about 15-25 min the first time
```

Plots and results are written to `src/Assignment4_simple/outputs/`.

---

## Problem 1: next-word prediction

**The big idea:** give the model 5 words and it guesses the 6th. That's a language model.

| Step | Function | What happens |
|---|---|---|
| Clean | `clean_text` | lowercase, replace anything that isn't a-z with a space, split into words |
| Vocabulary | `build_vocabulary` | keep the 10,000 most common words. Every word gets a number; rare words become `<unk>` |
| N-grams | `make_ngram_pairs` | slide a 6-word window across the text: the first 5 words are the input, the 6th is the target |
| GloVe | `load_glove_embeddings` | each word starts with a pre-trained 100-number vector. Words with similar meanings have similar vectors |
| LSTM | `LSTMModel` | reads the 5 words one at a time and updates its memory. After the last word, a Linear layer scores all 10,000 words |
| Transformer | `TransformerModel` | looks at all 5 words at once with self-attention. Position embeddings tell it the word order |
| Train | `train_model` | cross-entropy loss + Adam, 5 epochs |
| Measure | `run_one_epoch` | accuracy (was the top guess right?), top-5 accuracy, perplexity = e^loss |
| Demo | `interactive_demo` | you type a prompt and get the top-5 next words plus a 5-word continuation |

**Exam points**
- **Cross-entropy loss** is `-log(probability the model gave to the correct word)`. It's small when the model is confident and right.
- **Perplexity** = `e^loss`. A perplexity of 200 means the model is about as unsure as if it were picking
  uniformly from 200 words. Random guessing over 10,000 words gives a perplexity of 10,000.
- **Why pre-trained embeddings?** The model starts out already "knowing" that *king* and *queen* are related,
  so it learns faster from less data. `freeze=False` lets training fine-tune them.
- **LSTM vs Transformer:** the LSTM processes words in order (one step at a time). The Transformer processes them
  in parallel and lets each word attend directly to every other word. Because it has no built-in sense of order,
  it needs position embeddings.
- **Greedy decoding** always picks the top word, which tends to repeat itself ("of the of the ...").
  Sampling or beam search give more varied text.

---

## Problem 2: image captioning

**The big idea:** encoder-decoder. A CNN turns the image into numbers (the encoder).
A language model turns those numbers into a sentence (the decoder).

```
image --resize/normalize--> ResNet50 or InceptionV3 (frozen) --> 2048 numbers
                                                                     |
                                     <start> a dog runs ...  --> LSTM or Transformer --> a dog runs on grass <end>
```

| Step | Function | What happens |
|---|---|---|
| Captions | `clean_caption`, `caption_to_indices` | lowercase, letters only, add `<start>`/`<end>`, pad every caption to length 22 with `<pad>` |
| Images | `ImageDataset` | resize (224 for ResNet50, 299 for InceptionV3) and normalize with the ImageNet mean/std |
| CNN features | `load_cnn`, `extract_features` | replace the CNN's final classification layer with `Identity` to get a 2048-number image vector. This is computed once and saved to disk |
| LSTM decoder | `LSTMDecoder` | the image vector becomes the LSTM's starting memory (h0, c0), then the LSTM predicts each next word |
| Transformer decoder | `TransformerDecoder` | masked self-attention over the caption words, plus cross-attention to the image vector (the "memory") |
| Train | `run_one_epoch` | **teacher forcing**: the input is `<start> a dog runs` and the target is `a dog runs <end>`. Loss ignores `<pad>` |
| Generate | `generate_captions` | greedy: start at `<start>`, repeatedly add the most likely word, then cut at `<end>` |
| Evaluate | `compute_bleu` | BLEU-1/2/4 against the 5 human captions, plus a picture of example captions |

**Exam points**
- **Feature extraction vs fine-tuning:** the CNN is frozen (`eval()` + `no_grad`) and only the decoder trains.
  This is fast and works well with only 6,000 training images.
- **Teacher forcing:** during training the decoder always sees the *true* previous word, not its own guess.
  At test time it only has its own guesses. That mismatch is called *exposure bias*.
- **Causal mask:** in the Transformer, word *i* may only attend to words 0..i. Without the mask it could peek
  at the answer.
- **BLEU** measures n-gram overlap (precision) with the reference captions, with a brevity penalty for
  captions that are too short. BLEU-1 measures word choice; BLEU-4 measures fluent phrases. BLEU doesn't
  understand meaning: "a dog runs" and "a puppy sprints" barely overlap.
- **Why do the captions look generic?** Greedy decoding plus a single global image vector leads to safe,
  common captions like "a man in a red shirt is ...". Attention over image regions and beam search would help.
