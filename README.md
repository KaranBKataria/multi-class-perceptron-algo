# Multi-Class Perceptron Algorithm

## Overview

This university project involves programming the multi-class perceptron algorithm from scratch. The algorithm was then applied to determine whether a 6-class dataset is perfectly linearly separable in a space defined by: the original ($\mathrm{R}^{12}$) input space; linear principal component analysis (PCA); and (Radial Basis Function) Kernel PCA. Since the perceptron algorithm produces a (linear) binary classifier, a One-vs-Rest (OvR) approach is taken to obtain linear discriminants (hyperplanes) for each of the 6 classes, with evaluation metrics on the validation/test sets determined using macro-averaging. 

Both PCA forms' source code and implementation involved partial use of third-party libraries (NumPy and Pandas). However, for hyperparameter tuning, a grid-search function was programmed from scratch and was used to produce a heatmap.

## Dataset

The provided dataset included 12 numerical inputs, each corresponding to an encoding of variables related to human speech. Each record had a corresponding categorical variable for six accents (US, UK, GE, FR, ES and IT). 

The goal was to investigate whether it was possible to separate these six accents into distinct regions separated by linear discriminants.

Implementing the multi-class perceptron on different spaces (a 12-dimensional input space, a linear principal subspace, and a kernel principal subspace) revealed that perfect linear separability was possible in the feature space-defined kernel PCA. This was possible for specific pairs of Radial Basis Function (RBF) hyperparameters found via a grid search. These consisted of $\gamma$ values used as part of the RBF used to compute Kernel PCA and the top number of Kernel Principal Components.

## Kernelising PCA

If analysis suggests the data in the original input space is not linearly separable via a linear discriminant function (hyperplane), or in the linear PCA space, this suggests that the data may be lying in a lower-dimensional, non-linear manifold. This motivates the use of kernelised PCA, one of the simplest non-linear dimensionality reduction / manifold learning techniques, which projects the data onto a non-linear, arbitrary curve in the original input space.

Kernelising PCA requires the selection of a valid kernel $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ (under Mercer's conditions); the simplest and most common kernel is the Radial Basis Function (RBF) kernel:

$$
k(\mathbf{x}, \mathbf{x}') = \mathrm{exp}\Bigg(-\frac{\lVert \mathbf{x} - \mathbf{x}' \rVert_{2}^{2}}{2 \ell^{2}} \Bigg) = \mathrm{exp}\Bigg(- \gamma \lVert \mathbf{x} - \mathbf{x}' \rVert_{2}^{2} \Bigg)
$$

for $\mathbf{x}, \mathbf{x}' \in \mathcal{X} \subseteq \mathbb{R}^{n}$, $\ell \in \mathbb{R}$ which is the lengthscale parameter and $\lVert \cdot \rVert_{2}$ is the Euclidean norm.
