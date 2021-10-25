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


def plot_data(original, data, clusters=None):

    global tot_col

    # REDUCE DIMENSIONALITY TO PLOT
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(data)

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

        global tot_col

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

        if clusters is None:
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
        else:
            cl = tot_col[index]

            tweets = ''
            for i, cluster in enumerate(clusters):
                if cluster == cl:
                    tweets += ' ' + original['text<gx:text>'][i].strip()

            tfidf = TFIDF(texts=original['text<gx:text>'])
            words = tfidf.get_relevant_words(text=tweets, k=10)

            fig.suptitle(words, fontsize=10)

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



selected = []