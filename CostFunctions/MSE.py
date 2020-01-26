import numpy as np

class MSE:
    def calc_cost(self, X, y, features):
        """
        Returns mean squared error.
        features is the feature vector (numpy array, shape: features x 1)
        X is the input data (numpy array, shape: num of data points x features)
        Y is the true output (numpy array, shape: num of data points x 1)
        """
        m = X.shape[0]
        yhat = X.dot(features)
        mse = (1/ (2*m)) * np.sum((yhat - y)**2)
        return mse

    def calc_gradients(self, X, y, features):
        """
        Calculates gradients used for gradients decent using the partial
        derivative, in terms of features, of the mean squared error.
        """
        yhat = X.dot(features)
        m = X.shape[0]
        gradients = X.T.dot(yhat - y)
        return (1/m) * gradients

