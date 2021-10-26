from gensim.models import LdaMulticore, TfidfModel
import pickle
import numpy as np
import gensim
from utils_utils_lda import *

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

sns.set_style('whitegrid')


class LDA:

    def __init__(self, path, delete_duplicates=False, clean_stop=True, stemming=True, column='text<gx:text>',
                 deeper_clean=True):

        if deeper_clean:
            self._clean_df(path, column, delete_duplicates)
        else:
            self._get_corpus(path, column, delete_duplicates)
            self._clean_corpus(path, clean_stop, stemming)

        # self.model = LdaMulticore(corpus=self.corpus,
        #                         num_topics=self.lda_parameters.num_topics,
        #                       id2word=self.token2word,
        #                        passes=self.lda_parameters.passes,
        #                      workers=self.lda_parameters.workers
        #                     )

    def _clean_df(self, path, column, delete_duplicates):

        df = pd.read_csv(path)

        if delete_duplicates:
            df = df.groupby([column]).size().reset_index(name='counts')
        df = df.reset_index()
        df['tokens'] = df[column].apply(give_emoji_free_text)
        df['tokens'] = df['tokens'].apply(url_free_text)
        clean_stop(df)
        self.df = df
        self.corpus = df['tokens'].values

    def _get_corpus(self, path, column, delete_duplicates):

        df = pd.read_csv(path)

        if delete_duplicates:
            df = df.groupby([column]).size().reset_index(name='counts')
        df = df.reset_index()
        self.df = df
        self.corpus = df[column].values

    def _clean_corpus(self, stemming):

        self.corpus = np.array([remove_emoji(xi) for xi in self.corpus])
        self.corpus = np.array([remove_users(xi) for xi in self.corpus])
        self.corpus = np.array([remove_punctuation(xi) for xi in self.corpus])
        self.corpus = np.array([remove_url(xi) for xi in self.corpus])
        self.corpus = np.array([clean_tweet(xi, stemming=stemming) for xi in self.corpus])

    def corpus_vectorize(self, min_words_vectorizer=10):

        self.vectorizer = CountVectorizer(max_df=0.9, min_df=min_words_vectorizer,
                                          token_pattern='\w+|\$[\d\.]+|\S+')
        self.tf = self.vectorizer.fit_transform(self.corpus)  # .toarray()
        self.tf_feature_names = self.vectorizer.get_feature_names()

    def gensimlda_topics(self, ifprint=True, topics=10, iterat=50, below=2, above=.99, pas=5):

        # Create a id2word dictionary
        self.id2word = Dictionary(self.df['tokens_splitted'])
        self.id2word.filter_extremes(no_below=below, no_above=above)
        self.gensim_corpus = [self.id2word.doc2bow(d) for d in self.df['tokens_splitted']]
        self.gensimlda_model = LdaMulticore(corpus=self.gensim_corpus, num_topics=topics,
                                            id2word=self.id2word, workers=12, passes=pas, iterations=iterat,
                                            random_state=42)
        self.topic_words = [re.findall(r'"([^"]*)"', t[1]) for t in self.gensimlda_model.print_topics()]

        # Create Topics
        self.topics = [' '.join(t[0:10]) for t in self.topic_words]

        # Getting the topics
        if ifprint:
            for id, t in enumerate(self.topics):
                print(f"------ Topic {id} ------")
                print(t, end="\n\n")

    def get_scores(self, ifprint=True):

        # Compute Perplexity
        self.base_perplexity = self.gensimlda_model.log_perplexity(self.gensim_corpus)
        if ifprint:
            print('\nPerplexity: ', self.base_perplexity)

            # Compute Coherence Score
        coherence_model = CoherenceModel(model=self.gensimlda_model, texts=self.df['tokens_splitted'],
                                         dictionary=self.id2word, coherence='c_v')
        self.coherence_lda_model_base = coherence_model.get_coherence()
        if ifprint:
            print('\nCoherence Score: ', self.coherence_lda_model_base)

    def plot_10_most_common_words(self):

        words = self.tf_feature_names
        total_counts = np.zeros(len(words))
        for t in self.tf:
            total_counts += t.toarray()[0]

        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

        plt.figure(2, figsize=(15, 15 / 1.6180))
        plt.subplot(title='10 most common words')
        sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
        sns.barplot(x_pos, counts, palette='husl')
        plt.xticks(x_pos, words, rotation=90)
        plt.xlabel('words')
        plt.ylabel('counts')
        plt.show()


class LDA_Parameters:

    def __init__(self, num_topics=30, passes=4, workers=4):
        self.num_topics = num_topics
        self.passes = passes
        self.workers = workers


def plot_data(data, clusters=None):

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

    plt.show()
    