
def PCA(X: pd.DataFrame) -> pd.DataFrame:

    """
    This function calculates the principal components (PCs) of the original
    matrix of predictor variables using Singular Value Decomposition (SVD).

    The output is the change of basis containing the PCs of the original predictor
    matrix as columns. 
    """

    # Normalisation of the original matrix (mean = 0)
    X_centered = X.apply(lambda x: x-x.mean())

    # Decomposing the normalised dataset via SVD 
    U, S, VT = np.linalg.svd(X_centered)

    # Returning the change of basis matrix with PCs as columns
    return np.matmul(X_centered, VT.T)



if __name__ == "__main__":
    pass