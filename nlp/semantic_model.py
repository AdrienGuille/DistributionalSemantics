# coding: utf-8
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn import decomposition
from scipy import spatial, sparse, stats


__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def ppmi_transform(corpus, k=0):
    """
    This function transforms the raw co-occurrence frequency matrix with Positive Pointwise Mutual Information (PPMI)
    :param corpus: the corpus to be processed
    :param k: a constant positive value added to raw co-occurrence frequencies (Laplace smoothing), default to 0 (i.e. no smoothing)
    :return: a corpus with the updated X matrix
    """
    if k < 0:
        k = 0
    n_r = len(corpus.vocabulary)
    n_c = len(corpus.vocabulary)
    X = sparse.dok_matrix((n_r, n_c), dtype=np.float32)
    total_f_ij = 0
    for i in range(n_r):
        for j in range(n_c):
            total_f_ij += corpus.X[i, j] + k
    for i in range(n_r):
        for j in range(n_c):
            p_ij = (corpus.X[i, j] + k) / total_f_ij
            p_i = (corpus.X[i, :].sum() + n_c * k) / total_f_ij
            p_j = (corpus.X[:, j].sum() + n_r * k) / total_f_ij
            if p_ij / (p_i * p_j) > 0:
                pmi_ij = np.log10(p_ij / (p_i * p_j))
                if pmi_ij < 0:
                    pmi_ij = 0
            else:
                pmi_ij = 0
            X[i, j] = pmi_ij
    corpus.X = X
    return corpus


def jaccard_similarity(x, y):
    """
    This function calculates the Generalized Jaccard similarity measure of two real vectors of the same length
    :param x: a word vector
    :param y: a word vector
    :return: a real value in [0;1]
    """
    n = len(x)
    sum_min = .0
    sum_max = .0
    for i in range(0, n):
        sum_min += min(x[i], y[i])
        sum_max += max(x[i], y[i])
    return sum_min / sum_max


def cosine_similarity(x, y):
    """
    This function calculates the Cosine similarity measure of two real vectors of the same length
    :param x: a word vector
    :param y: a word vector
    :return: a real value in [0;1]
    """
    return 1 - spatial.distance.cosine(x, y)


def rand(a, b):
    return np.random.randn(a, b).astype('f')


class SemanticModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, corpus):
        self.corpus = corpus
        self.vector_space = None
        self.dimensions = None

    @abstractmethod
    def learn_vector_space(self, dimension=100):
        pass

    def vector_for_id(self, word_id):
        vector = self.vector_space[word_id, :]
        return vector

    def vector_for_word(self, word):
        if self.vector_space is not None:
            if self.corpus.vocabulary_map.get(word) is not None:
                word_id = self.corpus.vocabulary_map[word]
                return self.vector_for_id(word_id)
            else:
                raise ValueError("'%s' is not part of the vocabulary" % word)
        else:
            raise ValueError('Vector space undefined')

    def most_similar_words(self, word, nb_words=3, similarity_measure='cosine'):
        similarity = []
        if self.corpus.vocabulary_map.get(word) is not None:
            if similarity_measure not in ('cosine', 'jaccard'):
                raise ValueError("'similarity_measure' can only be either 'cosine or 'jaccard'")
            word_vector = self.vector_for_word(word)
            for i in range(len(self.corpus.vocabulary)):
                similarity.append(spatial.distance.cosine(word_vector, self.vector_for_id(i)))
            similar_word_ids = np.argsort(np.array(similarity)).tolist()
            similar_words = []
            for j in range(0, nb_words):
                this_word = self.corpus.vocabulary[similar_word_ids[j]]
                similar_words.append(this_word)
            return similar_words
        else:
            raise ValueError("'%s' is not part of the vocabulary" % word)


class PPMI_SVD(SemanticModel):

    def learn_vector_space(self, dimensions=100, k=0):
        self.dimensions = dimensions
        # apply PPMI transformation on X
        print('   Applying PPMI transformation on X...')
        self.corpus = ppmi_transform(self.corpus, k)
        # compute truncated SVD
        print('   Computing truncated SVD...')
        svd = decomposition.TruncatedSVD(n_components=dimensions)
        self.vector_space = svd.fit_transform(self.corpus.X)
