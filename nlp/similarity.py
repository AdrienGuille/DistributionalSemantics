# coding: utf-8

# math
from scipy import spatial
import numpy as np

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
    :return: a real value in ]0;1]
    """
    return 1 - spatial.distance.cosine(x, y)


def euclidean(x, y):
    """
    This function calculates the similarity of two real vectors of the same length using the Euclidean distance
    :param x: a word vector
    :param y: a word vector
    :return: a real value in ]0;1]
    """
    if np.array_equal(x, y):
        return 1
    else:
        return 1 / spatial.distance.euclidean(x, y)


def get_binary_vector(x):
    """
    This functions transforms a real vector into a binary vector, such that negative values become 0 and positive value become 1
    :param x: a word vector
    :return: a binary vector
    """
    binary_x = []
    for value in x:
        binary_value = 1
        if value < 0:
            binary_value = 0
        binary_x.append(binary_value)
    return np.array(binary_x)

