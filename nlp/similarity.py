# coding: utf-8
from scipy import spatial


__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def jaccard(x, y):
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


def cosine(x, y):
    """
    This function calculates the Cosine similarity measure of two real vectors of the same length
    :param x: a word vector
    :param y: a word vector
    :return: a real value in [0;1]
    """
    return 1 - spatial.distance.cosine(x, y)
