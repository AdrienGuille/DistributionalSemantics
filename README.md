# Distributional Semantics

This is a WIP Python 3 library of functions for learning vector space representations of words.

## Functions

As of now, this library offers functions for:

### Processing corpora
- Feature selection
- Co-occurrence matrix (with or without decreasing weighting)
    
### Learning vector space representations of words
- PPMI+SVD
- COALS
- GloVe (with regular stochastic gradient descent instead of AdaGrad)
    
### Measuring semantic similarity between words
- Cosine
- Generalized Jaccard

### Evaluating semantic models on a word similarity task
- Ground truth
    - WordSim-353
    - MC
    - RG
- Correlation
    - Pearson
    - Spearman

## Requirements

    NumPy
    SciPy
    scikit-learn