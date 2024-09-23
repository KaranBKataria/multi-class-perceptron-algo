
def kernel_pca(X: pd.DataFrame, g: float) -> np.array:

    """
    This function computes kernel PCA on the original dataframe depending on the
    input parameter g.

    The output is a change of basis data matrix produced through kernel PCA using
    the RBF kernel function.
    """
    
    K_normalized = normalized_kernel_matrix(X, g)

    # Performing SVD on the normalized kernel matrix K_normalized
    U, S, VT = np.linalg.svd(K_normalized)

    # Determining the change of basis matrix Z and converting into a matrix
    kernel_pca_df = np.matmul(K_normalized, VT.T)

    return kernel_pca_df


if __name__ == "__main__":
    pass