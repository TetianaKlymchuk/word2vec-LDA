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


class D2V:

    def __init__(self, df=None, path=None, column='text', embedding_dim=100, embedding_window=5, min_count=100):

        if path is None and df is None:
            raise Exception('You must provide either a df or a path')
        if df is None:
            self.load(path)
        else:
            self._get_embeddings(df, column, embedding_dim, embedding_window, min_count)

        self.embedding_dim = embedding_dim

    def _get_embeddings(self, df, column, embedding_dim, embedding_window, min_count):

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df[column])]
        model = Doc2Vec(documents, vector_size=embedding_dim, window=embedding_window, min_count=min_count, workers=4)

        self.model = model

    def get_vectors(self):

        return self.model.docvecs.vectors_docs

    def save(self, path):

        fname = get_tmpfile(path)
        self.model.save(fname)

    def load(self, path):

        fname = get_tmpfile(path)
        self.model = Doc2Vec.load(fname)