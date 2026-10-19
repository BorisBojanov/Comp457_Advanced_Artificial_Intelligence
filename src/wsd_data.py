"""SemCor data extraction for Problem 2 (word sense disambiguation).

One example = one occurrence of a target word in a SemCor sentence:
    {"target": "bank", "tokens": [...lowercased sentence...], "target_idx": 5,
     "synset": "bank.n.01", "sent_id": 123}

Run this file directly to print per-word sense counts and the most-frequent-sense
baseline, which is what to look at when choosing target words.
"""

import nltk
from collections import Counter, defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma


def ensureCorpora():
    # no-op when already downloaded
    for name in ("semcor", "wordnet", "omw-1.4"):
        nltk.download(name, quiet=True)


def extractExamples(targets):
    """Return a list of example dicts for every occurrence of each target lemma.

    Only single-word chunks tagged with a WordNet Lemma are used as targets
    (multiword expressions like 'bank_account' are skipped). Tokens are lowercased.
    """
    from nltk.corpus import semcor
    targets = {t.lower() for t in targets}
    examples = []

    for sent_id, sent in enumerate(semcor.tagged_sents(tag="sem")):
        tokens = []
        hits = []                                    # (index in tokens, lemma name, synset name)
        for chunk in sent:
            label = chunk.label() if hasattr(chunk, "label") else None
            words = chunk.leaves() if hasattr(chunk, "leaves") else list(chunk)
            if isinstance(label, Lemma) and len(words) == 1:
                lemma = label.name().lower()
                if lemma in targets:
                    hits.append((len(tokens), lemma, label.synset().name()))
            tokens.extend(w.lower() for w in words)

        for idx, lemma, synset in hits:
            examples.append({"target": lemma, "tokens": tokens, "target_idx": idx,
                             "synset": synset, "sent_id": sent_id})
    return examples


def senseInventory(word):
    """All WordNet senses of a word with gloss + example sentences (requirement 2)."""
    return [{"synset": s.name(), "pos": s.pos(), "gloss": s.definition(), "examples": s.examples()}
            for s in wn.synsets(word)]


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
    ensureCorpora()
    required = ["bank", "bat", "charge"]
    extras = ["line", "play", "run", "head", "hand", "point", "case", "interest", "state",
              "power", "plant", "party", "field", "light", "face", "place", "work", "book"]
    words = required + extras
    countReport(extractExamples(words), words)
