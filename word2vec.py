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

    def get_sentence_embedding(self, sentence):
        """
        Get the average vectors of the words in the word2vec vocabulary.
        :param sentence: text to get embedding. (str)
        :return: embedding vector (np.array)
                If no word of the sentence is in vocabulary, return None
        """

        word_count = 0
        embedding = None
        for word in re.findall(r"[\w']+|[.,!?;]", sentence.strip()):
            if word in self.wv.vocab:
                word_count += 1
                if embedding is None:
                    embedding = self.get_embedding(word)
                else:
                    embedding += self.get_embedding(word)

        if embedding is None:
            return None

        return embedding / word_countç

    def get_vectors(self, original, column='text<gx:text>'):
        tweet_embeddings = []
        for text in original[column]:
            emb = self.get_sentence_embedding(text.strip())
            if emb is not None:
                tweet_embeddings.append(emb)
            if emb is None:
                tweet_embeddings.append([0] * self.embedding_dim)

        return np.array(tweet_embeddings)

    def save(self, path='Data/wordvectors.kv'):
        self.wv.save(path)

    def load(self, path='Data/wordvectors.kv'):
        self.wv = KeyedVectors.load(path)

    def plot_embedding(self):

        vocab = list(self.wv.vocab)
        X = self.wv[self.wv.vocab]

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)

        df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(df['x'], df['y'])

        for word, pos in df.iterrows():
            ax.annotate(word, pos)

        plt.show()

    def plot_sentence_embeddings(self, original, clusters=None):

        # GET SENTENCE EMBEDDINGS
        tweet_embeddings = []
        for text in original['text<gx:text>']:
            emb = self.get_sentence_embedding(text.strip())
            if emb is not None:
                tweet_embeddings.append(emb)
            if emb is None:
                tweet_embeddings.append([0] * self.embedding_dim)

        tweet_embeddings = np.array(tweet_embeddings)

        # REDUCE DIMENSIONALITY TO PLOT
        tsne = TSNE(n_components=2, random_state=0)
        X_tsne = tsne.fit_transform(tweet_embeddings)

        df = pd.DataFrame(X_tsne, columns=['x', 'y'])

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)

        # CHANGE COLOR OF THE DATA POINT
        tot_col = []
        if clusters is None:
            tot_col = ['blue'] * len(df)
        else:
            tot_col = clusters

        ax.scatter(df['x'], df['y'], c=tot_col)

        original = original.reset_index()  # Needed to match the df indices.

        def onclick(event):

            ax.clear()

            # CHECK POINT NEAREST TO CURSOR CLICK
            min = 3
            minx = 3
            miny = 4
            index = None
            for word, pos in df.iterrows():
                x_dif = pos.x - event.xdata
                y_dif = pos.y - event.ydata
                dif = np.sqrt(x_dif ** 2 + y_dif ** 2)
                if dif < min:
                    minx = x_dif
                    miny = y_dif
                    min = dif
                    index = word

            # GET THE TWEET AND SPLIT EVERY 150 TO FIT INTO THE PLOT
            old_tweet = original['text<gx:text>'][index]
            split_every = 150

            last_i = 0
            tweet = ''
            if len(old_tweet) <= split_every:
                tweet = old_tweet
            else:
                for i in range(split_every, len(old_tweet), split_every):
                    tweet += old_tweet[last_i:i] + '\n'
                    last_i = i

                tweet += old_tweet[last_i:] + '\n'

            tweet = tweet.strip()

            # SET THE TWEET AS THE TITLE
            fig.suptitle(tweet, fontsize=12)

            # CHANGE COLOR OF THE DATA POINT
            if clusters is None:
                tot_col = []
                for m in range(len(df)):
                    if m == index:
                        tot_col.append('orange')
                    else:
                        tot_col.append('blue')
            else:
                tot_col = clusters
                tot_col[index] = 213

            ax.scatter(df['x'], df['y'], c=tot_col)

            plt.gcf().canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', onclick)

        start = None
        end = None

        def on_key(event):
            print('you pressed', event.key, event.xdata, event.ydata)
            global start
            global end

            # GET THE SELECTED BOX DATA POINTS FROM START ('s') TO END ('e') AND ('r') TO RESET
            if event.key == 'c':
                start = [event.xdata, event.ydata]
                end = None
            elif event.key == 'e':
                end = [event.xdata, event.ydata]

                if start[0] > end[0]:
                    start_x = end[0]
                    end[0] = start[0]
                    start[0] = start_x
                if start[1] > end[1]:
                    start_y = end[1]
                    end[1] = start[1]
                    start[1] = start_y

                rect = []
                for index, pos in df.iterrows():
                    if start[0] < pos.x < end[0]:
                        if start[1] < pos.y < end[1]:
                            rect.append(index)

                tweets = ''
                tot_col = ['blue'] * len(original)
                for index in rect:
                    tweets += ' ' + original['text<gx:text>'][index].strip()
                    tot_col[index] = 'orange'

                tfidf = TFIDF(texts=original['text<gx:text>'])
                words = tfidf.get_relevant_words(text=tweets, k=10)

                fig.suptitle(words, fontsize=10)
                ax.scatter(df['x'], df['y'], c=tot_col)
                plt.gcf().canvas.draw_idle()
            elif event.key == 'r':
                start = None
                end = None

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

    def plot_clusters_words(self, list_list_words, tweets):
        vocab = list(self.wv.vocab)
        # X = self.wv.vocab

        X = self.wv[self.wv.vocab]
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)

        df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

        fig = plt.figure()

        if self.cluster_model is None:
            self.get_cluster_model(n_clusters=4)

        color_dict = {
            -1: 'blue',
            0: 'orange',
            1: 'red',
            2: 'green',
            3: 'yellow',
            4: 'purple',
            5: 'black',
            6: 'gray',
        }

        for tweet_number, list_words in enumerate(list_list_words):

            old_tweet = tweets[tweet_number]
            print(old_tweet)

            split_every = 75

            last_i = 0
            tweet = ''
            if len(old_tweet) <= split_every:
                tweet = old_tweet
            else:
                for i in range(split_every, len(old_tweet), split_every):
                    tweet += old_tweet[last_i:i] + '\n'
                    last_i = i

                tweet += old_tweet[i] + '\n'

            tweet = tweet.strip()

            ax = fig.add_subplot(1, 1, 1)
            print('TWEET {}'.format(tweet))
            fig.suptitle(tweet, fontsize=6)

            colors = {}
            clust = {}
            for i, word in enumerate(vocab):
                if word in list_words:
                    clust[word] = self.cluster_model.labels_[i]
                    colors[word] = color_dict[clust[word]]

            tot_col = []
            for word, pos in df.iterrows():
                if word in list_words:
                    ax.annotate(word, pos)
                    tot_col.append(color_dict[clust[word]])
                else:
                    tot_col.append(color_dict[-1])

            ax.scatter(df['x'], df['y'], c=tot_col)

            filename = 'plots/tweet_{}'.format(tweet_number)
            print('Saving tweet {} at {}'.format(tweet_number, filename))

            plt.savefig(filename, dpi=150)
            # CLOSE PLOTS
            ax.remove()
            # plt.close('all')

    def get_cluster_model(self, n_clusters):
        """
        Get cluster model.
        :return:
        """
        X = self.wv[self.wv.vocab]
        self.cluster_model = KMeans(n_clusters=n_clusters).fit(X)

    def get_cluster(self, x):

        x = np.array(x)
        x = x.astype(np.double)
        return self.cluster_model.predict(x)

    def plot_clusters(self, range_cluster=(0, 5)):
        # TODO
        if type(range_cluster) == int:
            min, max = (range_cluster, range_cluster + 1)
        else:
            min, max = range_cluster

        vocab = list(self.wv.vocab)
        list_words = vocab
        # X = self.wv.vocab

        X = self.wv[self.wv.vocab]
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)

        df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for n_clusters in range(min, max):
            self.get_cluster_model(n_clusters=n_clusters)

            clust = {}
            for i, word in enumerate(vocab):
                if word in list_words:
                    clust[word] = self.cluster_model.labels_[i]

            color_dict = {
                -1: 'blue',
                0: 'orange',
                1: 'red',
                2: 'green',
                3: 'yellow',
                4: 'purple',
                5: 'black',
                6: 'gray',
            }

            colors = {}
            for word in list_words:
                if word in clust:
                    colors[word] = color_dict[clust[word]]

            tot_col = []
            for word, _ in df.iterrows():
                tot_col.append(color_dict[clust[word]])

            ax.scatter(df['x'], df['y'], c=tot_col)

            filename = 'plots/clusters_{}'.format(n_clusters)
            print('Saving cluster {} at {}'.format(n_clusters, filename))

            plt.savefig(filename, dpi=150)
            # CLOSE PLOTS
            # plt.close('all')