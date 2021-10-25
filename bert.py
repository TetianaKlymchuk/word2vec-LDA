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


class BertEmbeddings:

    def __init__(self, df=None, path=None, column='text'):

        if path:
            self.load(path)
        else:
            # INIT BERT AND THE TOKENIZER TO COMPUTE QUESTION EMBEDDING
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            embeddings = []
            for tweet in tqdm(df[column]):
            # GET BERT OUTPUT FOR THE QUESTION
                inputs = tokenizer(tweet, return_tensors="pt")
                outputs = model(**inputs)

                # WE WILL TAKE THE [CLS] EMBEDDING AS THE QUESTION EMBEDDING
                last_hidden_states = outputs[0]
                tweet_embedding = np.array(last_hidden_states[0][0].detach().numpy())
                embeddings.append(tweet_embedding)

            self.embeddings = np.array(embeddings)

    def get_vectors(self):

        return self.embeddings

    def save(self, path):

        *dir, file = path.split('/')
        dir = '/'.join(dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        np.save(path, self.embeddings)

    def load(self, path):

        self.embeddings = np.load(path)