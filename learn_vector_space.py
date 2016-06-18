# coding: utf-8
from nlp.semantic_model import PPMI_SVD, COALS, GloVe
import timeit
import pickle
import argparse

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Prepare a corpus (vocabulary extraction + co-occurrence matrix computation)')
    p.add_argument('i', metavar='input', type=str, help='The input csv file')
    p.add_argument('o', metavar='output', type=str, help='The output pickle file')
    p.add_argument('--m', metavar='method', type=str, help='Vector space learning method (PPMI+SVD, COALS or GloVe, default to GloVe)', default='GloVe')
    p.add_argument('--d', metavar='dimensions', type=int, help='Number of dimensions (default to 100', default=100)
    p.add_argument('--xm', metavar='x_max', type=int, help='GloVe: x_max (default to 100)', default=100)
    p.add_argument('--a', metavar='alpha', type=bool, help='GloVe: alpha (default to 0.75)', default=0.75)
    args = p.parse_args()

    print('Arguments:\n   Input file: %s\n   Output file: %s\n   Method: %d\n   Dimensions: %d' % (args.i, args.o, args.m, args.d))
    my_corpus = pickle.load(args.i)
    my_semantic_model = None
    if args.m == 'PPMI+SVD':
        print('Learning vector space with PPMI+SVD...')
        start_time = timeit.default_timer()
        my_semantic_model = PPMI_SVD(my_corpus)
        my_semantic_model.learn_vector_space(dimensions=args.d)
        elapsed = timeit.default_timer() - start_time
        print('Vector space learned in %f seconds.' % elapsed)
    elif args.m == 'COALS':
        print('Learning vector space with COALS...')
        start_time = timeit.default_timer()
        my_semantic_model = COALS(my_corpus)
        my_semantic_model.learn_vector_space(dimensions=args.d)
        elapsed = timeit.default_timer() - start_time
        print('Vector space learned in %f seconds.' % elapsed)
    elif args.m == 'GloVe':
        print('Learning vector space with GloVe...')
        start_time = timeit.default_timer()
        my_semantic_model = GloVe(my_corpus, x_max=args.xm, alpha=args.a)
        my_semantic_model.learn_vector_space(dimensions=args.d)
        elapsed = timeit.default_timer() - start_time
        print('Vector space learned in %f seconds.' % elapsed)
    else:
        raise ValueError('Unknown method "%s"' % args.m)
    pickle.dump(my_semantic_model, open(args.o, 'wb'))
