
# Creating a sanity check function to output the number of errors and the accuracy of the linear classifier 
# The function evalutes the performance of the multi-class perceptron algorithm
# sanity_check takes in as input the original dataset X (which acts as both are training and validation set)
  # and the refined weight matrix W which is given by the multi_class_perceptron function 

def sanity_check(X: pd.DataFrame, W: pd.DataFrame):
    errSum = 0   # Intialise the number of errors found to be 0 
    r = list(range(X.shape[0]))
    random.shuffle(r)
    for i in r:
        x = X[i,:] 
        y = labels[i]
        e = y - np.argmax((np.matmul(W, x)))   # compute the loss using argmax where y is the i'th ground truth
        if e != 0:
            errSum += 1    # Every time an error occurs, inrease the count of errors errSum by 1 
    return({"r3": ((X.shape[0])-errSum)/X.shape[0], "r4": errSum, "r10": 'accuracy is ' + str(((X.shape[0])-errSum)/X.shape[0])})