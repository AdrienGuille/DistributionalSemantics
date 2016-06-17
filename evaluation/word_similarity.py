# coding: utf-8

# standard
import csv

# math
from scipy import stats

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class WordSimilarity:

    def __init__(self, evaluation_data):
        if evaluation_data not in ('wordsim353', 'rg', 'mc'):
            raise ValueError("'similarity_measure' can only be either 'wordsim353', 'rg' or 'mc'")
        input_file = open('evaluation_data/'+evaluation_data+'.tsv', 'r')
        csv_reader = csv.reader(input_file, delimiter='\t')
        self.ground_truth = []
        for row in csv_reader:
            word_1 = row[0]
            word_2 = row[1]
            similarity = row[2]
            self.ground_truth.append((word_1, word_2, similarity))
        print(self.ground_truth)

    def evaluate(self, semantic_model, similarity_measure='cosine', binary=False, correlation_measure='spearman'):
        correlation_function = None
        if correlation_measure not in ('pearson', 'spearman'):
            raise ValueError("'similarity_measure' can only be either 'pearson or 'spearman'")
        elif similarity_measure == 'pearson':
            correlation_function = stats.pearsonr
        elif similarity_measure == 'spearman':
            correlation_function = stats.spearmanr
        truth = []
        estimation = []
        for word_1, word_2, similarity in self.ground_truth:
            word_1_vector = semantic_model.get_vector(word_1)
            word_2_vector = semantic_model.get_vector(word_2)
            estimated_similarity = semantic_model.measure_similarity(word_1_vector,
                                                                     word_2_vector,
                                                                     similarity_measure=similarity_measure,
                                                                     binary=binary)
            truth.append(similarity)
            estimation.append(estimated_similarity)
        return correlation_function(truth, estimation)
