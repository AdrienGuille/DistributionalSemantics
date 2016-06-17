# coding: utf-8
from structure.corpus import Corpus
from nlp.semantic_model import PPMI_SVD, COALS, GloVe
import timeit
import pickle

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

print('Loading corpus...')
start_time = timeit.default_timer()
my_corpus = Corpus('input/messages1.csv', max_nb_features=50000, window_size=5, decreasing_weighting=True)
elapsed = timeit.default_timer() - start_time
print('Corpus loaded in %f seconds.' % elapsed)

print('Learning vector space with GloVe...')
start_time = timeit.default_timer()
my_semantic_model = GloVe(my_corpus)
my_semantic_model.learn_vector_space(dimensions=100)
elapsed = timeit.default_timer() - start_time
print('Vector space learned in %f seconds.' % elapsed)

pickle.dump(my_semantic_model, open('my_semantic_model.pickle', 'wb'))
