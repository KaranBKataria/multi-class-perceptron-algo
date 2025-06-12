# Multi-Class Perceptron Algorithm

## Overview

This project, part of a university assignment, involves programming the multi-class perceptron algorithm from scratch. The algorithm was then applied to determine whether a 6-class dataset is perfectly linearly separable in a space defined by: the original ($\mathrm{R}^{12}$) input space; linear PCA; and (Radial Basis Function) Kernel PCA. 

Both PCA forms' source code and implementation involved partial use of third-party libraries (NumPy and Pandas). However, for hyperparameter tuning, a grid-search function was programmed from scratch and was used to produce a heatmap. 

## Dataset

The provided dataset included 12 numerical input dimensions, each corresponding to an encoding of variables related to human speech. Each record had a corresponding categorical variable for six accents (US, UK, GE, FR, ES and IT). 

The goal was to investigate whether it was possible to separate these six accents into distinct regions separated by a linear classifier.

Implementing the multi-class perceptron on different spaces (12-dimensional input space, linear principal subspace, and kernel principal subspace) revealed that perfect linear separability was possible in the feature space-defined kernel PCA. This was possible for specific pairs of RBF hyperparameters found via a grid search. These consisted of gamma values used as part of the Radial Basis Kernel (RBF) function used to compute Kernel PCA and the top number of Kernel Principal Components.

## Kernelising PCA

If analysis suggests the data in the original input space is not linearly separable via a linear discriminant function (hyperplane), or in the linear PCA space, this suggests that the data may be lying in a lower-dimensional, non-linear manifold. This motivates the use of kernelised PCA, one of the simplest non-linear dimensionality reduction / manifold learning approaches to project data onto a non-linear, arbitrary curve in the original input space.

Kernelising PCA requires the selection of a valid kernel $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ (under Mercer's conditions); the simplest and most common kernel is the Radial Basis Function (RBF) kernel:

$$
k(\mathbf{x}, \mathbf{x}') = \mathrm{exp}(-\frac{\lVert \mathbf{x} - \mathbf{x}' \rVert{2}^{2}}{2 \ell^{2}})
$$
