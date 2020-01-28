import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, X, y, theta=None, normalize=True):
        self.X = X
        self.y = y
        self.theta = theta
        self.normalized = normalize

        #reshape y if shape is z, to z,1
        try:
            y.shape[1]
        except:
            self.y = y.reshape(y.shape[0], 1)

        #reshape x if shape is z, to z,1
        try:
            X.shape[1]
        except:
            self.X = X.reshape(X.shape[0], 1)

        if self.normalized:
            self.normalize()
        self.insert_ones_column()

        if type(self.theta) == type(None):
            self.theta = np.zeros((self.X.shape[1], 1))

    def insert_ones_column(self):
        if not (np.all(self.X[:,0] == np.ones((1, self.X.shape[0])))):
            #if first column is not all ones, append a column of ones
            self.X = np.hstack(( np.ones(( self.X.shape[0],1 )), self.X))

    def normalize(self):
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)

    def normal_equation(self):
        """
        Closed form solution for linear Regression
        No need to normalize before using
        Note: slow when number of features is high (above 10,000ish)
        """
        inv = np.linalg.inv(np.dot(self.X.T, self.X))
        self.theta = np.dot(np.dot(inv, self.X.T), self.y)
        return self.theta

    def calc_cost(self):
        """
        Returns mean squared error.
        features is the feature vector (numpy array, shape: features x 1)
        X is the input data (numpy array, shape: num of data points x features)
        Y is the true output (numpy array, shape: num of data points x 1)
        """
        m = self.X.shape[0]
        yhat = self.X.dot(self.theta)
        mse = (1/ (2*m)) * np.sum((yhat - self.y)**2)
        return mse

    def step_gradient(self, lr):
        yhat = self.X.dot(self.theta)
        m = self.X.shape[0]
        gradients = self.X.T.dot(yhat - self.y)
        self.theta = self.theta - ( lr * (1/m) * gradients)

    def gradient_descent(self, lr, itr, plot=False):
        """
        Gradient Descent using mean squared error as cost function
        """
        if plot:
            cost_hist = np.empty((itr, 1))
            for i in range(itr):
                self.step_gradient(lr)
                cost_hist[i,0] = self.calc_cost()
            plt.plot([i for i in range(1,itr+1)], cost_hist)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title("Cost over Iterations")
            plt.show()

        else:
            for i in range(itr):
                self.step_gradient(lr)

        return self.theta

    def SGD(self):
        #TODO
        pass

    def predict(self, X=None):
        if type(X) == type(None):
            return self.X.dot(self.theta)
        X = self.insert_ones_column(X)
        return X.dot(self.theta)

    def score(self, X, y):
        #TODO fix this
        X = self.insert_ones_column(X)
        mse = MSE(X, y, self.theta)
        tss = np.sum(X.dot(self.theta)**2)
        return (tss - mse) / tss

    def plot(self, X=None, y=None, theta=None):
        if type(X) == type(None):
            X = self.X
        if type(y) == type(None):
            y = self.y
        if type(theta) == type(None):
            theta = self.theta

        if X.shape[1] > 2:
            print("Plotting only availble for 2 dimensional data")
            return

        plt.scatter(X[:,1], y)
        yhat = X.dot(theta)
        plt.plot(X[:,1].reshape(X.shape[0],1), yhat, c='r')
        plt.legend(['Regression Line','Data Points'], loc='upper left')
        plt.show()
