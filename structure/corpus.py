# coding: utf-8
import csv
import string

from scipy.sparse import dok_matrix
import numpy as np

__authors__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def tokenize(text):
    # split the documents into tokens based on whitespaces
    raw_tokens = text.split()
    # trim punctuation and convert to lower case
    return [token.strip(string.punctuation).lower() for token in raw_tokens]


class Corpus:
    def __init__(self, source_file_path, max_nb_features=10000, window_size=5, decreasing_weighting=False):
        input_file = open(source_file_path)
        csv_reader = csv.reader(input_file, delimiter='\t')
        header = next(csv_reader)
        text_column_index = header.index('text')

        # first pass to identify features, i.e. the vocabulary
        print('   Identifying features (i.e. the vocabulary)...')
        self.size = 0
        word_frequency = {}
        for line in csv_reader:
            self.size += 1
            words = tokenize(line[text_column_index])
            # update word frequency
            for word in words:
                if len(word) > 0:
                    frequency = 0
                    if word_frequency.get(word) is not None:
                        frequency = word_frequency[word]
                    frequency += 1
                    word_frequency[word] = frequency
        # sort words w.r.t frequency
        vocabulary = list(word_frequency.items())
        vocabulary.sort(key=lambda x: x[1], reverse=True)
        self.vocabulary = []
        self.vocabulary_map = {}
        # construct the structures
        for i in range(min(max_nb_features, len(vocabulary))):
            feature = vocabulary[i][0]
            self.vocabulary.append(feature)
            self.vocabulary_map[feature] = i
        print('      Vocabulary size: %d' % len(self.vocabulary))

        # second pass to compute the co-occurrence matrix

        print('   Computing X (i.e. the co-occurrence frequency matrix)...')
        self.X = dok_matrix((len(self.vocabulary), len(self.vocabulary)), dtype=np.short)
        # go back to the beginning of the csv file
        input_file.seek(0)
        csv_reader = csv.reader(input_file, delimiter='\t')
        for line in csv_reader:
            words = tokenize(line[text_column_index])
            nb_words = len(words)
            for i in range(nb_words):
                # check if the current word is part of the vocabulary
                if self.vocabulary_map.get(words[i]) is not None:
                    row_index = self.vocabulary_map[words[i]]
                    # extract surrounding words w.r.t window size
                    start = i - window_size
                    if start < 0:
                        start = 0
                    end = i + window_size
                    if end >= nb_words:
                        end = nb_words - 1
                    # scan left context
                    context_left = words[start:i]
                    for j in range(0, len(context_left)):
                        if self.vocabulary_map.get(context_left[j]) is not None:
                            column_index = self.vocabulary_map[context_left[j]]
                            # update co-occurrence count
                            count = 0
                            weight = 1
                            if decreasing_weighting:
                                weight = len(context_left) - j
                            if (row_index, column_index) in self.X:
                                count = self.X[row_index, column_index]
                            self.X[row_index, column_index] = count + 1 / weight
                    # scan right context
                    context_right = words[i + 1:end + 1]
                    for j in range(0, len(context_right)):
                        if self.vocabulary_map.get(context_left[j]) is not None:
                            column_index = self.vocabulary_map[context_left[j]]
                            # update co-occurrence count
                            count = 0
                            weight = 1
                            if decreasing_weighting:
                                weight = j
                            if (row_index, column_index) in self.X:
                                count = self.X[row_index, column_index]
                            self.X[row_index, column_index] = count + 1 / weight
        print('      Number of non-zero entries: %d' % self.X.getnnz())
        self.X = self.X.tocsr()
