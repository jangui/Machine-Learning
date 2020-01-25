import numpy as np

def MSE(theta, X, y):
    """
    Returns mean squared error.
    theta is the feature vector (numpy array, shape: features x 1)
    X is the input data (numpy array, shape: num of data points x features)
    Y is the true output (numpy array, shape: num of data points x 1)
    """
    m = X.shape[0]
    yhat = X.dot(theta)
    mse = (1/ (2*m)) * np.sum((yhat - y)**2)
    return mse

