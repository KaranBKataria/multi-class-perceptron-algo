
def sanity_check(X: pd.DataFrame, W: pd.DataFrame):

  """
  This function acts as a sanity check and computes the number of errors and
  accuracy of the lineary classifiers.
  """

  errSum = 0

  r = list(range(X.shape[0]))
  random.shuffle(r)

  for i in r:
    x = X[i, :] 
    y = labels[i]

    # compute the loss using argmax where y is the i'th ground truth
    e = y - np.argmax((np.matmul(W, x)))

    # Every time an error occurs, increment the count of errors errSum by 1 
    if e != 0:
        errSum += 1

  return \
    {
      "r3": ((X.shape[0])-errSum)/X.shape[0],
      "r4": errSum,
      "r10": f"accuracy is {str(((X.shape[0])-errSum)/X.shape[0])}"
    }