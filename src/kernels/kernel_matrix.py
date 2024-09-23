
def kernel_matrix(X: pd.DataFrame, g: float) -> np.array:

    """
    This function computes the Kernel matrix K for a given gamma value g which
    serves a hyperparameter. Each element (i, j) of matrix K is the output of the
    RBF Kernel function.
    """

    # We intialise K to be an empty n x n matrix to populate
    K = np.empty((X.shape[0], X.shape[0]))

    for i in list(range(X.shape[0])):
        for j in list(range(X.shape[0])):

            # We use the RBF kernel function to impute values into K element-wise
            K[i][j] =  np.exp((-g)*(np.dot(np.array(X)[i, :] - np.array(X)[j, :], np.array(X)[i, :] - np.array(X)[j, :])))
    
    return K



def normalized_kernel_matrix(X: pd.DataFrame, g: float) -> np.array:

    """
    This function computes the normalised kernel matrix
    """

    # We call the kernel_matrix function to produce our kernel matrix K
    K = kernel_matrix(X, g)

    # We intialise the n x n matrix with all values of 1/n 
    A = np.full((X.shape[0], X.shape[0]), (1)/(X.shape[0]))

    # We normalize the kernel matrix K to produce the normalized kernel matrix
    K_norm = K - A@K - K@A + A@K@A

    return K_norm


if __name__ == "__main__":
    pass