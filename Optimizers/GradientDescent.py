import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, lr, itr, plot=False):
        self.lr = lr
        self.plot = plot
        self.itr = itr

    def optimize(self, X, y, features, cost):
        if self.plot:
            cost_hist = np.empty((self.itr, 1))
            for i in range(self.itr):
                features -= self.lr * cost.calc_gradients(X, y, features)
                cost_hist[i,0] = cost.calc_cost(X, y, features)
            self.plot_cost(cost_hist)

        else:
            for i in range(self.itr):
                features -= self.lr * cost.calc_gradients(X, y, features)

        return features

    def plot_cost(self, cost_hist):
        plt.plot([i for i in range(1,self.itr+1)], cost_hist)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost over Iterations")
        plt.show()
