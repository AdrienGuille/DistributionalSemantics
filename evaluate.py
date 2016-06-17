# coding: utf-8
from structure.corpus import Corpus
from nlp.semantic_model import PPMI_SVD, COALS, GloVe
from evaluation.human_judgment import word_similarity
import timeit
import csv

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

test = word_similarity()
