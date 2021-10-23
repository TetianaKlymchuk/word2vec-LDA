from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import string
import re
from tqdm import tqdm

from gensim.test.utils import get_tmpfile
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

from transformers import BertTokenizer, BertModel
import torch

from utils.utils import *


class Embeddings:
    """
    Class that handles word embeddings
    """

    def __init__(self, df=None, path=None, column='text', embedding_dim=100, embedding_window=5, min_count=100):
        """
        :param df:  dataframe with coluns specified
        :param path: path to pretrained embeddings.
        :param column: either abstract or text (default: text) <str>
        :param embedding_dim: word vectors dimension.
        :param embedding_window: window size to compute embeddings.
        :param min_count: minimum frequency of a word in the corpus to consider it.
        """
        if path is None and df is None:
            raise Exception('You must provide either a df or a path')
        if df is None:
            self.load(path)
        else:
            model = self._get_embeddings(df, column, embedding_dim, embedding_window, min_count)
            self.wv = model.wv

        self.embedding_dim = embedding_dim
        self.cluster_model = None

    def _get_embeddings(self, df, column='text', embedding_dim=200, embedding_window=3, min_count=30):
        """
        Returns gensim Word2Vec model trained on the column given.
        """

        l = []
        for des in df[column]:
            des = des.translate(des.maketrans({key: ' ' for key in string.punctuation}))
            des = des.lower()
            l.append(re.findall(r"[\w']+|[.,!?;]", des.strip()))
        model = Word2Vec(l, workers=4, size=embedding_dim, min_count=min_count, window=embedding_window,
                         sample=1e-3, sg=1, seed=1)

        return model

    def most_similar(self, word, k):
        """
        Get the k nearest neighbors to the word.
        :param word: word to find nearest neighbors.
        :param k: number of neighbors to return
        :return: list of (word, similarity)
        """
        return self.wv.most_similar(word, topn=k)

    def get_embedding(self, word):
        """
        Method to obtain the embedding vector of a word.
        :param word: word to obtain embedding.
        :return: word vector <np.array>
        """
        return np.array(self.wv[word])

    def get_sentence_embedding(self):
        pass

    def plot_embedding(self):
        pass

    def plot_cluster(self):
        pass