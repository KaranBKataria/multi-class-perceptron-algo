# The function below is for the standard binary class Perceptron algorithm 
# The function below calculates a linear classifier for each class (requiring a total of 6 linear classifiers for all 6 accents) by letting a given accent have a label of +1 and every other class the label of -1.

def Perceptron2(X: pd.DataFrame, Y: pd.DataFrame, accent: str) -> np.array:
    NEW_Y = Y.apply(lambda x: 1 if x == accent else -1)
    w = np.zeros(X.shape[1])
    errors = 1
    steps = 0
    while (errors != 0 and steps < 10):    # Continue the loop until the number of iterations completed = 10 or num. of errors = 0 
        errors = 0
        r = list(range(X.shape[0]))
        random.shuffle(r)  # Randomizing the feature vector we evaluate at each step 
        for i in r:
            x = X[i,:]
            y = NEW_Y[i]
            e = y - np.sign(np.matmul(w,x))   # Using the sign function to ensure we get a prediction of {-1,1}
            if e != 0:   # If an error occurs i.e e != 0, count an error into the total number of errors found and updated the weights 
                errors += 1   # If an error occurs, include the error count by 1 
                w = w + (y * x)  # If an error occurs, update the weight accordingly 
        steps += 1  # Once an iteration is completed, increase the count by 1 until 10 is reached 
    return w

# The function returns a weight vector 'w' which has the refined weights (after 10 iterations)