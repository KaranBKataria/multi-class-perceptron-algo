
def multi_class_perceptron(X: pd.DataFrame, Y: pd.DataFrame, W_combined: np.array) -> np.array:

    """
    This function defines the standard MULTI-class Perceptron algorithm. It
    fits a linear classifier through updating the weight matrix provided as input
    (the combination of the invidiual weight vectors for each accent class).

    The maximum number of iterations in fitting the linear classifier is fixed
    as a integer literal of 10, as per the specifications of the coursework to
    prevent infinite recursion (assuming perfect linear classification is not
    possible).

    The output is a refined weightm matrix for the linear classifier.
    """

    errors = 1
    steps = 0

    # Continue the loop until the number of iterations completed = 10 or num. of errors = 0
    while (errors != 0 and steps < 10):
        errors = 0

        r = list(range(X.shape[0]))
        random.shuffle(r)   # Randomizing the feature vector we evaluate at each step

        for i in r:
            x = X[i, :]
            y = Y[i]

            y_pred = np.argmax((np.matmul(W_combined, x)))

            e = y - y_pred

            if e != 0:

                errors += 1

                # Updating the row which corresponds to the weight vector for the incorrect prediction by reducing it by x 
                W_combined[y_pred, :] = W_combined[y_pred, :] - x

                # Updating the row which corresponds to the weight vector for the ground truth by increasing it by x 
                W_combined[y, :] = W_combined[y, :] + x

        # Once all i's in r are completed, increase iteration count by 1
        steps += 1
    
    return W_combined


if __name__ == "__main__":
    pass