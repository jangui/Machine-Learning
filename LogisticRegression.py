import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, X, y, theta=None, normalize=True):
        self.y = y

        if normalize:
            X = self.normalize(X)
        self.X = self.insert_ones_column(X)

        if not theta:
            self.theta = np.ones((self.X.shape[1], 1))

    def insert_ones_column(self, X):
        if not (np.all(X[:,0] == np.ones((1, X.shape[0])))):
            #if first column is not all ones, append a column of ones
            return np.hstack(( np.ones(( X.shape[0],1 )), X))
        return X

    def normalize(self, X):
        return (X - np.mean(X)) / np.std(X)

    def fit(self):
        pass

    def predict(self, X):
        X = self.insert_ones_column(X)
        return X.dot(self.theta)

    def compute_cost(self, X, y):
        """
        Returns the cost (mean squared error) of predicted values vs true values.
        """
        if self.normalized:
            X = self.normalize(X)
        X = self.insert_ones_column(X)
        return MSE(self.theta, X, y)

    def confusion_matrix(self):
        #add a self.accuracy?
        pass


    def plot(self):
        if not self.X.shape[0] > 2:
            print("Plotting only availble for 2 dimensional data")
            return
        plt.scatter(self.X[:,1], self.y)
        yhat = self.X.dot(self.theta)
        plt.plot(self.X[:,1].reshape(self.X.shape[0],1), yhat, c='r')
        plt.legend(['Regression Line','Data Points'], loc='upper left')
        plt.show()

    def normalEq(self):
        """
        Closed form solution
        No need to normalize, slow when number of features is high (above 10,000ish)
        """
        inv = np.linalg.inv(np.dot(self.X.T, self.X))
        self.theta = np.dot(np.dot(inv, self.X.T), self.y)

    def step_gradient(self, alpha):
        yhat = self.X.dot(self.theta)
        m = self.X.shape[0]
        gradients = self.X.T.dot(yhat - self.y)
        self.theta = self.theta - (alpha * (1/m) * gradients)

    def gradient_descent(self, alpha, itr, plot=False):
        if plot:
            cost_hist = np.empty((itr, 1))
            for i in range(itr):
                self.step_gradient(alpha)
                cost_hist[i,0] = MSE(self.theta, self.X, self.y)
                print(cost_hist[i,0])

            plt.plot([i for i in range(1,itr+1)], cost_hist)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title("Cost over Iterations")
            plt.show()

        else:
            for i in range(itr):
                self.step_gradient(alpha)


    def SGD(self, alpha, itr, plot=False):
        """
        Stochastic Gradient Descent optimizing based of MSE for cost
        Stops once cost within threshold defined by self.accuracy
        """
        pass
