# Multi-Class Perceptron Algorithm

## Overview

This project, part of a university assignment, involves programming the multi-class perceptron algorithm from scratch. The algorithm was then applied to determine whether a 6-class dataset is perfectly linearly separable in a space defined by: the original (12D) feature space; linear PCA; and Kernel PCA. 

Both PCA forms' source code and implementation involved partial use of third-party libraries (NumPy and Pandas). However, for hyperparameter tuning, a grid-search function was programmed from scratch and was used to produce a heatmap. 

## Dataset

The provided dataset included 12 numerical features, each corresponding to an encoding of variables related to human speech. Each record had a corresponding categorical variable for six different accents (US, UK, GE, FR, ES and IT). 

The goal was to investigate whether it was possible to categorise these six accents into distinct regions separated by a linear classifier.

Through implementing the multi-class perceptron on different feature spaces (original 12D, linear PCA and kernel PCA), it was found that perfect linear separability was possible in the feature space defined kernel PCA. This was possible for specific pairs of hyperparameters found via a grid search. These consisted of gamma values used as part of the Radial Basis Kernel (RBF) function used to compute Kernel PCA and the top number of Kernel Principal Components.
