# coding: utf-8
from structure.corpus import Corpus
from nlp.semantic_model import PPMI_SVD
import timeit

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

print('Loading corpus...')
start_time = timeit.default_timer()
my_corpus = Corpus('input/messages1.csv', max_nb_features=20000, window_size=5)
elapsed = timeit.default_timer() - start_time
print(my_corpus.vocabulary)
print('Corpus loaded in %f seconds.' % elapsed)

print('Learning vector space...')
start_time = timeit.default_timer()
my_semantic_model = PPMI_SVD(my_corpus)
my_semantic_model.learn_vector_space(dimensions=100)
elapsed = timeit.default_timer() - start_time
print('Vector space learned in %f seconds.' % elapsed)

while True:
    a_word = str(input('Type a word: '))
    if a_word in my_corpus.vocabulary:
        if a_word != '"quit"':
            print('Most similar words to %s: %s' % (a_word, ', '.join(my_semantic_model.most_similar_words(a_word))))
        else:
            break
