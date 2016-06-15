# coding: utf-8
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn import decomposition
from nlp import similarity, transformation
from random import shuffle

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
        self.vector_space = svd.fit_transform(self.corpus.X.to_csr())


class COALS(SemanticModel):

    def learn_vector_space(self, dimensions=100):
        self.dimensions = dimensions
        # apply PPSC transformation on X
        print('   Applying Positive Pairwise Squared Correlation transformation on X...')
        self.corpus = transformation.ppsc(self.corpus)
        # compute truncated SVD
        print('   Computing truncated Singular Value Decomposition of X...')
        svd = decomposition.TruncatedSVD(n_components=dimensions)
        self.vector_space = svd.fit_transform(self.corpus.X.to_csr())


class GloVe(SemanticModel):

    def __init__(self, corpus, x_max=100, alpha=0.75, gamma=0.05):
        super(GloVe, self).__init__(corpus)
        self.x_max = x_max
        self.alpha = alpha
        self.gamma = gamma
        self.W_main = None
        self.W_context = None
        self.b_main = None
        self.b_context = None

    def weighting_function(self, x):
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        else:
            return 1

    def next_iteration(self):
        # initialize the global loss after for this iteration
        global_cost = 0
        # randomly shuffle entries
        entries = list(self.corpus.X.items())
        shuffle(entries)
        # loop through all the entries, compute gradients and update word vectors and biases
        for entry in entries:
            i = entry[0][0]
            j = entry[0][1]
            x_ij = entry[1]
            w_i = self.W_main[i, :]
            w_j = self.W_context[j, :]

            # calculate the weight related to x_ij
            f_x_ij = self.weighting_function(x_ij)
            # evaluate the local loss
            inner_local_loss = np.dot(w_i, w_j) + self.b_main[i] + self.b_context[j] - np.log(x_ij)
            local_loss = f_x_ij * inner_local_loss
            # update the global loss
            global_cost += local_loss

            # update gradients for the W matrices
            grad_w_i = f_x_ij * inner_local_loss * w_j
            grad_w_j = f_x_ij * inner_local_loss * w_i
            # update gradients for the biases
            grad_b_i = f_x_ij * inner_local_loss
            grad_b_j = f_x_ij * inner_local_loss

            # update word vectors and biases in the opposite direction of gradients
            self.W_main[i, :] -= self.gamma * grad_w_i
            self.W_context[j, :] -= self.gamma * grad_w_j
            self.b_main[i] -= self.gamma * grad_b_i
            self.b_context[j] -= self.gamma * grad_b_j
        print(global_cost)

    def learn_vector_space(self, dimensions=100, iterations=25):
        # randomly initialize (main & context) word vectors and (main & context) word biases
        bound = 1
        self.W_main = np.random.uniform(-bound, bound, [len(self.corpus.vocabulary), dimensions])
        self.W_context = np.random.uniform(-bound, bound, [len(self.corpus.vocabulary), dimensions])
        self.b_main = np.random.uniform(-bound, bound, [len(self.corpus.vocabulary), 1])
        self.b_context = np.random.uniform(-bound, bound, [len(self.corpus.vocabulary), 1])

        # perform stochastic gradient descent
        for i in range(iterations):
            print(i)
            self.next_iteration()
        self.vector_space = self.W_main
        self.W_main = None
        self.W_context = None
        self.b_main = None
        self.b_context = None
