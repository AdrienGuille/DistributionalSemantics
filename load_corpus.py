# coding: utf-8
from structure.corpus import Corpus
import timeit
import pickle
import argparse

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Prepare a corpus (vocabulary extraction + co-occurrence matrix computation)')
    p.add_argument('i', metavar='input', type=str, help='Input csv file')
    p.add_argument('o', metavar='output', type=str, help='Output pickle file')
    p.add_argument('--mnf', metavar='max_nb_features', type=int, help='Vocabulary size (default to 50000)', default=50000)
    p.add_argument('--ws', metavar='window_size', type=int, help='Context window size (default to 5)', default=5)
    p.add_argument('--dw', metavar='decreasing_weighting', type=bool, help='Decreasing weighting (True or False, default to False)', default=False)
    args = p.parse_args()

    print('Arguments:\n   Input file: %s\n   Output file: %s\n   Max number of features: %d\n   Window size: %d\n   Decreasing weighting: %s' %
          (args.i, args.o, args.mnf, args.ws, args.dw))
    print('Loading corpus...')
    start_time = timeit.default_timer()
    my_corpus = Corpus(args.i, max_nb_features=args.mnf, window_size=args.ws, decreasing_weighting=args.dw)
    elapsed = timeit.default_timer() - start_time
    print('Corpus loaded in %f seconds.' % elapsed)
    pickle.dump(my_corpus, open(args.o, 'wb'))
