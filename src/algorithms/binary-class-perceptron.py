
def Perceptron2(X: pd.DataFrame, Y: pd.DataFrame, accent: str) -> np.array:

    """
    This function defines the standard BINARY class Perceptron algorithm. It
    creates a linear classifier for each class where the target class is given the
    label +1 and all others are binned into the -1 class.

    The maximum number of iterations in fitting the linear classifier is fixed
    as a integer literal of 10, as per the specifications of the coursework to
    prevent infinite recursion (assuming perfect linear classification is not
    possible).

    The output is a vector of weights for the linear classifier.
    """

    NEW_Y = Y.apply(lambda x: 1 if x == accent else -1)

    w = np.zeros(X.shape[1])

    errors = 1
    steps = 0

    # Continue the loop until the number of iterations completed = 10 or num. of errors = 0
    while (errors != 0 and steps < 10):
        errors = 0

        r = list(range(X.shape[0]))
        random.shuffle(r)  # Randomizing the feature vector we evaluate at each step 

        for i in r:
            x = X[i, :]
            y = NEW_Y[i]

            # Using the sign function to ensure we get a prediction of {-1,1}
            e = y - np.sign(np.matmul(w, x))

            # If an error occurs i.e e != 0, count an error into the total number of errors found and updated the weights 
            if e != 0:
                errors += 1 

                # If an error occurs, update the weight accordingly
                w = w + (y*x)

        steps += 1  # Once an iteration is completed, increase the count by 1 until 10 is reached

    return w


    if __name__ == "__main__":
        pass