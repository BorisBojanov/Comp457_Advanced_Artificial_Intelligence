"""SemCor data extraction for Problem 2 (word sense disambiguation).

One example = one occurrence of a target word in a SemCor sentence:
    {"target": "bank", "tokens": [...lowercased sentence...], "target_idx": 5,
     "synset": "bank.n.01", "sent_id": 123}

Run this file directly to print per-word sense counts and the most-frequent-sense
baseline, which is what to look at when choosing target words.
"""

import nltk
from functools import lru_cache
from collections import Counter, defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
from nltk.corpus.reader.wordnet import Lemma

REQUIRED_WORDS = ["bank", "bat", "charge"]
EXTRA_WORDS = ["line", "play", "run", "head", "hand", "point", "case", "interest", "state",
               "power", "plant", "party", "field", "fence", "light", "face", "place", "work", "book"]
TARGET_WORDS = REQUIRED_WORDS + EXTRA_WORDS

TEST_EVERY = 5      # sentences with sent_id % 5 == 0 go to the test set (~20%)


# Ensure the semcor dataset is downloaded
def downloadDataset():
    # no-op when already downloaded
    for name in ("semcor", "wordnet", "omw-1.4"):
        nltk.download(name, quiet=True)


def extractExamples(targets):
    """Return a list of example dicts for every occurrence of each target lemma.

    Only single-word chunks tagged with a WordNet Lemma are used as targets
        (multiword expressions like 'bank_account' are skipped).
    Tokens are lowercased.
    """
    targets = {t.lower() for t in targets}

    examples = []

    for sent_id, sent in enumerate(semcor.tagged_sents(tag="sem")):
        tokens = []
        hits = []                                    # (index in tokens, lemma name, synset name)
        for chunk in sent:
            if isinstance(chunk, nltk.Tree):
                label = chunk.label()
                words = chunk.leaves()
            else:
                label = None
                words = list(chunk)                  # untagged chunks are plain lists like ['The']

            if isinstance(label, Lemma) and len(words) == 1:
                lemma = label.name().lower()
                if lemma in targets:
                    hits.append((len(tokens), lemma, label.synset().name()))

            # Build a lowercased token list for the sentence, one token per word.
            for w in words:
                tokens.append(str(w).lower())

        for idx, lemma, synset in hits:
            examples.append({"target": lemma,
                             "tokens": tokens,
                             "target_idx": idx,
                             "synset": synset,
                             "sent_id": sent_id})
    return examples


@lru_cache(maxsize=None)
def _extractCached(targets: tuple):
    # parsing SemCor takes a while and train + test both need it, so do it once per target list
    downloadDataset()
    return extractExamples(targets)


def splitExamples(examples):
    """Split by SENTENCE so a sentence never lands in both train and test.

    Deterministic (no random seed): the same call always gives the same split.
    """
    train = [e for e in examples if e["sent_id"] % TEST_EVERY != 0]
    test = [e for e in examples if e["sent_id"] % TEST_EVERY == 0]
    return train, test


def semcor_dataset(targets=TARGET_WORDS, train=True):
    """Per-occurrence examples for the target words: the train split, or the test split."""
    train_examples, test_examples = splitExamples(_extractCached(tuple(targets)))
    return train_examples if train else test_examples


def buildSenseSpace(targets=TARGET_WORDS):
    """Label space + candidate mask, taken from WordNet (requirement 2).

    senses    : every WordNet synset of every target word, in a fixed order = class index
    targets   : the target lemmas, in a fixed order = target index
    mask[t][s]: True when sense s is a WordNet sense of target t. The model sets every other
                logit to -inf so 'bank' can only be predicted as a sense of 'bank'.
    Built from WordNet only (not from the data), so train and test share it without leakage.
    """
    targets = [t.lower() for t in targets]
    senses, seen, candidates = [], set(), {}
    for t in targets:
        candidates[t] = [s.name() for s in wn.synsets(t)] # type: ignore
        for name in candidates[t]:
            if name not in seen:
                seen.add(name)
                senses.append(name)

    sense2idx = {s: i for i, s in enumerate(senses)}
    mask = [[False] * len(senses) for _ in targets]
    for ti, t in enumerate(targets):
        for name in candidates[t]:
            mask[ti][sense2idx[name]] = True

    return {"senses": senses, "sense2idx": sense2idx,
            "targets": targets, "target2idx": {t: i for i, t in enumerate(targets)},
            "mask": mask}


def mostFrequentSense(train_examples, test_examples):
    """Baseline: for every test example predict the sense most often seen for that word in TRAIN.

    A word with no training examples falls back to WordNet's first sense. No model, no context -
    anything that beats this is actually using the sentence.
    """
    best = {}
    for word in {e["target"] for e in train_examples}:
        counts = Counter(e["synset"] for e in train_examples if e["target"] == word)
        best[word] = counts.most_common(1)[0][0]
    return [best.get(e["target"]) or wn.synsets(e["target"])[0].name() for e in test_examples] # type: ignore


def senseInventory(word):
    """All WordNet senses of a word with gloss + example sentences (requirement 2)."""
    if word is None:
        return []

    results = []
    for s in wn.synsets(str(word)):
        if s is None:
            continue
        results.append({
            "synset": s.name(),
            "pos": s.pos(),
            "gloss": s.definition(),
            "examples": s.examples(),
        })
    return results


def countReport(examples, targets):
    by_word = defaultdict(Counter)
    for ex in examples:
        by_word[ex["target"]][ex["synset"]] += 1

    print(f"{'word':<10}{'examples':>9}{'senses used':>12}{'WN senses':>10}{'MFS acc':>9}   top senses")
    for word in targets:
        counts = by_word.get(word, Counter())
        total = sum(counts.values())
        mfs = counts.most_common(1)[0][1] / total if total else 0.0
        top = ", ".join(f"{s}:{n}" for s, n in counts.most_common(4))
        print(f"{word:<10}{total:>9}{len(counts):>12}{len(wn.synsets(word)):>10}{mfs:>9.2f}   {top}")


if __name__ == "__main__":
    downloadDataset()
    countReport(extractExamples(TARGET_WORDS), TARGET_WORDS)
