#!/usr/bin/env python3
import numpy as np
from LinearRegression import LinearRegression
from Optimizers.GradientDescent import GradientDescent
from CostFunctions.MSE import MSE


data = np.array(np.genfromtxt('linreg2.txt', delimiter=','))
X = data[:,0:-1]
#X = X.reshape(X.shape[0], 1)
y = data[:,-1]
y = y.reshape(y.shape[0], 1)

lr = LinearRegression()
cost = MSE()
opt = GradientDescent(0.01, 1500, True)
lr.compile(cost, opt)
lr.fit(X, y)
print(lr.calc_cost(X,y))
lr.fit(X, y, lr.features)
print(lr.calc_cost(X,y))

#print(lr.score(X, y))

#print(lr.compute_cost(X,y))
"""
lr.gradient_descent(alpha=0.01, itr=1500, plot=True)
#print(lr.score(X, y))
print(lr.compute_cost(X, y))

#print(lr.predict(X))

#lr.plot()
"""
