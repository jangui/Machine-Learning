import numpy as np
import matplotlib.pyplot as plt
from Error import Error

class LinearRegression:
    def __init__(self):
        self.normalized = False
        self.features = None

    def insert_ones_column(self, X):
        if not (np.all(X[:,0] == np.ones((1, X.shape[0])))):
            #if first column is not all ones, append a column of ones
            return np.hstack(( np.ones(( X.shape[0],1 )), X))
        return X

    def normalize(self, X):
        return (X - np.mean(X)) / np.std(X)

    def compile(self, cost, optimizer):
        self.cost = cost
        self.optimizer = optimizer
        self.compiled = True

    def fit(self, X, y, features=None, normalize=True):
        if not self.compiled:
            raise Error("Model must be compiled before fitted")

        if normalize:
            X = self.normalize(X)
            self.normalized = True
        X = self.insert_ones_column(X)

        if type(features) == type(None):
            features = np.ones((X.shape[1], 1))

        self.optimizer.optimize(X, y, features, self.cost)
        self.features = features

    def predict(self, X):
        X = self.insert_ones_column(X)
        return X.dot(self.features)

    def calc_cost(self, X, y, features=None):
        if type(features) == type(None):
            features = self.features

        if self.normalized:
            X = self.normalize(X)
        X = self.insert_ones_column(X)
        return self.cost.calc_cost(X, y, features)

    def score(self, X, y):
        #TODO fix this
        X = self.insert_ones_column(X)
        mse = MSE(X, y, self.theta)
        tss = np.sum(X.dot(self.theta)**2)
        return (tss - mse) / tss

    def plot(self, X, y, features=None):
        if not X.shape[0] > 2:
            print("Plotting only availble for 2 dimensional data")
            return

        if type(features) == type(None):
            features = self.features

        plt.scatter(X[:,1], y)
        yhat = X.dot(features)
        plt.plot(X[:,1].reshape(X.shape[0],1), yhat, c='r')
        plt.legend(['Regression Line','Data Points'], loc='upper left')
        plt.show()



