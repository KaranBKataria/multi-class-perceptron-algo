# Below is the function for the Multi-Class Perceptron Algorithm. 
# The function/algorithm is limited to 10 iterations. 1 run through of all 329 data points counts as 1 iteration. 
# The function will return a new, refined weight matrix 'W_combined'.
# The new, refined weight matrix is computed through updating the weights per iteration if errors occur and is returned by the function 

def multi_class_perceptron(X, Y, W_combined):

    """
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

            y_pred = np.argmax((np.matmul(W_combined,x)))

            e = y - y_pred

            if e != 0:

                errors += 1

                W_combined[y_pred,:] = W_combined[y_pred,:] - x   # Updating the row which corresponds to the weight vector for the incorrect prediction by reducing it by x 
                
                W_combined[y,:] = W_combined[y,:] + x    # Updating the row which corresponds to the weight vector for the ground truth by increasing it by x 
        
        steps += 1   # Once all i's in r are completed, increase iteration count by 1
    
    return W_combined


if __name__ == "__main__":
    pass