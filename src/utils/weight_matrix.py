
def combined_weights(X: pd.DataFrame, Y: pd.DataFrame) -> np.array:

    """
    This function creates a global weight matrix where reach row corresponds to
    the weight vector for a given class returned from the binary perceptron
    algorithm. As the number of accents is deterministic, the weight matrix
    will have dimensions 6 x 13 (12 predictors and an additional column of 1's to
    add bias). 
    """

    # Note accents is a global variable (list)    
    # Stacking the weight vectors as rows to produce a weight matrix
    W_combined = np.stack([Perceptron2(X, Y, i) for i in accents], axis=0) 
    
    return W_combined



if __name__ == "__main__":
    pass