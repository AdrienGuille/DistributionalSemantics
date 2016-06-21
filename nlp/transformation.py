# coding: utf-8

# math
import numpy as np
from scipy import sparse, stats

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def ppsc(corpus):
    """
    This function transforms the raw co-occurrence frequency matrix with Positive Pairwise Squared Correlation (PPSC)
    :param corpus: the corpus to be processed
    :return: a corpus with the updated X matrix
    """
    n_r = len(corpus.vocabulary)
    n_c = len(corpus.vocabulary)
    X = sparse.dok_matrix((n_r, n_c), dtype=np.float32)
    for i in range(n_r):
        for j in range(n_c):
            v_i = corpus.X[i, :].toarray()
            v_j = corpus.X[j, :].toarray()
            correlation = stats.pearsonr(v_i[0, :], v_j[0, :])[0]
            if correlation < 0:
                correlation = 0
            else:
                correlation = np.sqrt(correlation)
            X[i, j] = correlation
    corpus.X = X
    return corpus


def ppmi(corpus, k=0):
    """
    This function transforms the raw co-occurrence frequency matrix with Positive Pointwise Mutual Information (PPMI)
    :param corpus: the corpus to be processed
    :param k: a constant positive value added to raw co-occurrence frequencies (Laplace smoothing), default to 0 (i.e. no smoothing)
    :return: a corpus with the updated X matrix
    """
    if k < 0:
        k = 0
    n_r = len(corpus.vocabulary)
    n_c = n_r
    X = sparse.dok_matrix((n_r, n_c), dtype=np.float32)
    total_f_ij = corpus.X.sum() + k * n_r * n_c
    for i in range(0, n_r):
        for j in range(0, i):
            p_ij = (corpus.X[i, j] + k) / total_f_ij
            p_i = (corpus.X[i, :].sum() + n_c * k) / total_f_ij
            p_j = (corpus.X[:, j].sum() + n_r * k) / total_f_ij
            if p_ij / (p_i * p_j) > 0:
                pmi_ij = np.log10(p_ij / (p_i * p_j))
                if pmi_ij < 0:
                    pmi_ij = 0
            else:
                pmi_ij = 0
            if pmi_ij > 0:
                X[i, j] = pmi_ij
    for i in range(0, n_r):
        for j in range(i, n_c):
            if X[j, i] > 0:
                X[i, j] = X[j, i]
    corpus.X = X
    return corpus
