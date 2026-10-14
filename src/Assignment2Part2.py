"""
Comp457_Assignment2_Part2_Problem1_Sentiment_Analysis_with_RNN/LSTM

Assignment 2 coding problems in part 2 of assignment
Measuring Sentiment Analysis with RNN/LSTM

IMDB dataset for sentiment analysis: https://ai.stanford.edu/~amaas/data/sentiment/
    It ships with Pytorch-NLP.

We use perplexity as a metric to evaluate the performance of our RNN/LSTM model for sentiment analysis.
perplexity of sequence X:
- Perplexity = 1: Perfect model (always predicts the right token with probability 1)
- Perplexity = ∞ (infinity): Worst case (assigns probability 0 to true tokens)
- Perplexity = |vocabulary| (size of vocabulary): Baseline (uniform distribution over all words)
- Lower perplexity = better model

"""
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pth"
BATCH_SIZE = 64
EPOCHS = 5

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RNN:
    pass
class LSTM:
    pass

def example():
    """Source code for torchnlp.datasets.imdb
        https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/imdb.html
        
        Prerequisite:pip3 install pytorch-nlp

    """
    
    import os
    import glob

    from torchnlp.download import download_file_maybe_extract


    def imdb_dataset(directory='data/',
                    train=False,
                    test=False,
                    train_directory='train',
                    test_directory='test',
                    extracted_name='aclImdb',
                    check_files=['aclImdb/README'],
                    url='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                    sentiments=['pos', 'neg']):
        """
        Load the IMDB dataset (Large Movie Review Dataset v1.0).

        This is a dataset for binary sentiment classification containing substantially more data than
        previous benchmark datasets. Provided a set of 25,000 highly polar movie reviews for
        training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text
        and already processed bag of words formats are provided.

        Note:
            The order examples are returned is not guaranteed due to ``iglob``.

        **Reference:** http://ai.stanford.edu/~amaas/data/sentiment/

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







def main():
    pass

if __name__ == "__main__":
    main()