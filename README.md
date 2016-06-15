# Distributional Semantics

This is a WIP Python 3 library of functions for learning vector space representations of words.

## Functions

As of now, this library offers functions for:

### Processing corpora
- Vocabulary
- Word-word co-occurrence matrix (with or without decreasing weighting)
    
### Learning vector space representations of words
- PPMI+SVD
- COALS
- GloVe (with classical stochastic gradient descent instead of AdaGrad)
    
### Measuring semantic similarity between words
- Cosine
- Generalized Jaccard

## Requirements

    NumPy
    SciPy
    scikit-learn
