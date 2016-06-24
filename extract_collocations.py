# coding: utf-8
import nlp.transformation as transformation
import timeit
import pickle
import argparse
import numpy as np

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Identify the most significant collocates of a given word')
    p.add_argument('i', metavar='input', type=str, help='Input pickle file')
    p.add_argument('--k', metavar='k', type=str, help='Laplace smoothing constant, default to 0', default=0)
    p.add_argument('--ppmi', metavar='ppmi', type=bool, help='Laplace smoothing constant, default to 0', default=False)
    args = p.parse_args()

    print('Loading pickled corpus...')
    start_time = timeit.default_timer()
    my_corpus = pickle.load(open(args.i, 'rb'))
    print('   Corpus size: %d\n   Vocabulary size: %d' % (my_corpus.size, len(my_corpus.vocabulary)))
    elapsed = timeit.default_timer() - start_time
    print('Corpus loaded in %f seconds.' % elapsed)

    if args.ppmi:
        print('Applying PPMI transformation on the co-occurrence matrix...')
        start_time = timeit.default_timer()
        my_corpus = transformation.ppmi(my_corpus)
        elapsed = timeit.default_timer() - start_time
        print('Done in %f seconds.' % elapsed)

    a_word = ''
    while a_word != '_quit_':
        a_word = input('Type a word: ')
        word_id = my_corpus.vocabulary_map[a_word]
        print(a_word)
        word_vector = my_corpus.X[word_id, :].toarray()
        word_vector = word_vector[0, :]
        sorted_ids = np.argsort(word_vector)
        for i in range(len(sorted_ids)-10, len(sorted_ids)):
            print(my_corpus.vocabulary[sorted_ids[i]])
