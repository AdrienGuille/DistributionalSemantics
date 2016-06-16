# coding: utf-8
from structure.corpus import Corpus
from nlp.semantic_model import PPMI_SVD, COALS, GloVe
import timeit
import pickle

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

print('Loading corpus...')
start_time = timeit.default_timer()
my_corpus = Corpus('input/messages1.csv', max_nb_features=500, window_size=5, decreasing_weighting=True)
elapsed = timeit.default_timer() - start_time
print('Corpus loaded in %f seconds.' % elapsed)

method = input('Select a method (either PPMI+SVD, COALS or GloVe): ')
my_semantic_model = None

if method == 'PPMI+SVD':
    print('Learning vector space with PPMI+SVD...')
    start_time = timeit.default_timer()
    my_semantic_model = PPMI_SVD(my_corpus)
    my_semantic_model.learn_vector_space(dimensions=100)
    elapsed = timeit.default_timer() - start_time
    print('Vector space learned in %f seconds.' % elapsed)
elif method == 'COALS':
    print('Learning vector space with COALS...')
    start_time = timeit.default_timer()
    my_semantic_model = COALS(my_corpus)
    my_semantic_model.learn_vector_space(dimensions=100)
    elapsed = timeit.default_timer() - start_time
    print('Vector space learned in %f seconds.' % elapsed)
elif method == 'GloVe':
    print('Learning vector space with GloVe...')
    start_time = timeit.default_timer()
    my_semantic_model = GloVe(my_corpus)
    my_semantic_model.learn_vector_space(dimensions=100)
    elapsed = timeit.default_timer() - start_time
    print('Vector space learned in %f seconds.' % elapsed)
else:
    raise ValueError('Unknown method "%s"' % method)

save = input('Save model on disk? (y/n): ')
if save == 'y':
    pickle.dump(my_semantic_model, open('my_semantic_model.pickle', 'wb'))

on = True
while on:
    a_word = str(input('Type a word: '))
    if a_word in my_corpus.vocabulary:
        if a_word != '_quit':
            print('Most similar words to "%s": %s' % (a_word, ', '.join(my_semantic_model.most_similar_words(a_word))))
        else:
            on = False
