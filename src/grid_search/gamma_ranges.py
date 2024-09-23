
def gamma_ranges(X: pd.DataFrame) -> list[float]:

    """
    This function computes the min and max values for gamma to be used as the
    limits to the hyperparameter tuning process.
    """

    # Normalising the dataset X 
    X_centered = X.apply(lambda x: x-x.mean())

    variance_list = []
    min_max_list = []

    for i in list(range(X.shape[1])):

        # For each column in X, determine the variance and add to the list variance_list
        variance_list.append(np.var(np.array(X_centered)[:, i]))
    
    for j in variance_list:  
        if j == np.max(variance_list):
            min_max_list.append(j)
        elif j == np.min(variance_list):
            min_max_list.append(j)
        else:
            continue
    
    g_min = (1)/(2*(np.max(min_max_list)+0.05))
    
    g_max = (1)/(2*(np.min(min_max_list)-0.05))
    
    return [g_min, g_max]



def gamma_increments(X: pd.DataFrame) -> list[float]:

    """
    This function outputs the range of gamma values we can perform grid-search on.

    Note that the increments of 0.01 are as per the specifications of the coursework.
    """

    gamma_vals = gamma_ranges(X)

    # We intialise the gamma summation to be equal to the minimum value
    sum_gamma = np.min(gamma_vals)

    list_gamma = [sum_gamma]

    while sum_gamma < np.max(gamma_vals)-0.010:

        # As long we are within the range between min and max gamma, add on increments of 0.010 
        sum_gamma += 0.010
        list_gamma.append(sum_gamma)

    # Append the maximum value into the gamma list list_gamma
    list_gamma.append(np.max(gamma_vals))

    return list_gamma



def grid_matrix2(X: pd.DataFrame, Y: pd.DataFrame, n: int, m: int) -> np.array:

    """
    This function aims to populate an intially empty matrix (numpy array) element-wise 
    with values corresponding to the number of errors the multi-class perceptron 
    makes for a given gamma values for PCs m to n.
    
    This intially empty matrix once populated is returned to form the grid matrix, 
    which will be used to produce the matrix plot for the grid-search.
    """

    # We add 1 onto n to make sure the max PC is included in functions such as
    # np.arange which would otherwise include the PC n-1 and not n 
    n += 1
    
    matrix_grid_search = np.empty((n-m, 8))

    # If the empty array/matrix has any null values we replace them with 0 
    matrix_grid_search_updated = np.nan_to_num(matrix_grid_search)
    
    # the list of gammas for a dataset X to be tested on 
    gamma_list = gamma_increments(X)
    
    X = X.to_numpy()
    
    for gamma in gamma_list:
        kpca = kernal_pca(X, gamma)

        # Where g_index is the index of the gamma value in gamma_list
        g_index = gamma_list.index(gamma)

        for k in list(np.arange(m, n, 1)):
            X_training = kpca[:, :k+1]
            W_combined = combined_weights(X_training, Y)
            W_combined_optimal = multi_class_perceptron(X_training, Y, W_combined)

            # Makes i,j be the indices of the k and gamma currently in the loop 
            i, j = list(np.arange(m, n, 1)).index(k), g_index

            # We populate the array element-wise with values corresponding to the 
            # number of errors made for a given PC i and gamma value j
            matrix_grid_search_updated[i][j] = sanity_check(X_training, W_combined_optimal)["r4"]
    
    return matrix_grid_search_updated



if __name__ == "__main__":
    pass