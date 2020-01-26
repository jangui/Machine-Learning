import numpy as np

class NormalEquation:
        """
        Closed form solution for linear Regression
        No need to normalize before using
        Note: slow when number of features is high (above 10,000ish)
        """
    def __init__(self, X, y, theta, cost, alpha, itr, plot=False):
        self.X = X
        self.y = y
        self.theta = theta

    def optimize(self):
        inv = np.linalg.inv(np.dot(self.X.T, self.X))
        self.theta = np.dot(np.dot(inv, self.X.T), self.y)
        return self.theta
