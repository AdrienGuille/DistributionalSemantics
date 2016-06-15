# coding: utf-8
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn import decomposition
from nlp import similarity, transformation

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class SemanticModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, corpus):
        self.corpus = corpus
        self.vector_space = None
        self.dimensions = None

    @abstractmethod
    def learn_vector_space(self, dimension=100):
        pass

    def vector_for_id(self, word_id):
        vector = self.vector_space[word_id, :]
        return vector

    def vector_for_word(self, word):
        if self.vector_space is not None:
            if self.corpus.vocabulary_map.get(word) is not None:
                word_id = self.corpus.vocabulary_map[word]
                return self.vector_for_id(word_id)
            else:
                raise ValueError("'%s' is not part of the vocabulary" % word)
        else:
            raise ValueError('Vector space undefined')

    def most_similar_words(self, word, nb_words=5, similarity_measure='cosine'):
        sim = []
        similarity_function = None
        if self.corpus.vocabulary_map.get(word) is not None:
            if similarity_measure not in ('cosine', 'jaccard'):
                raise ValueError("'similarity_measure' can only be either 'cosine or 'jaccard'")
            elif similarity_measure == 'cosine':
                similarity_function = similarity.cosine
            elif similarity_measure == 'jaccard':
                similarity_function = similarity.jaccard
            word_vector = self.vector_for_word(word)
            for i in range(len(self.corpus.vocabulary)):
                sim.append(similarity_function(word_vector, self.vector_for_id(i)))
            similar_word_ids = np.argsort(np.array(sim)).tolist()[::-1]
            similar_words = []
            for j in range(0, nb_words):
                this_word = self.corpus.vocabulary[similar_word_ids[j]]
                similar_words.append(this_word)
            return similar_words
        else:
            raise ValueError("'%s' is not part of the vocabulary" % word)


class PPMI_SVD(SemanticModel):

    def learn_vector_space(self, dimensions=100, k=0):
        self.dimensions = dimensions
        # apply PPMI transformation on X
        print('   Applying Positive Pointwise Mutual Information transformation on X...')
        self.corpus = transformation.ppmi(self.corpus, k)
        # compute truncated SVD
        print('   Computing truncated Singular Value Decomposition of X...')
        svd = decomposition.TruncatedSVD(n_components=dimensions)
        self.vector_space = svd.fit_transform(self.corpus.X)


class COALS(SemanticModel):

    def learn_vector_space(self, dimensions=100):
        self.dimensions = dimensions
        # apply PPSC transformation on X
        print('   Applying Positive Pairwise Squared Correlation transformation on X...')
        self.corpus = transformation.ppsc(self.corpus)
        # compute truncated SVD
        print('   Computing truncated Singular Value Decomposition of X...')
        svd = decomposition.TruncatedSVD(n_components=dimensions)
        self.vector_space = svd.fit_transform(self.corpus.X)
