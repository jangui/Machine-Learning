#!/usr/bin/env python3
import numpy as np
from LinearRegression import LinearRegression

data = np.array(np.genfromtxt('ex1data2.txt', delimiter=','))
X = data[:,0:-1]
#X = X.reshape(X.shape[0], 1)
y = data[:,-1]
y = y.reshape(y.shape[0], 1)

lr = LinearRegression(X, y, normalize=False)
#lr.fit()

#print(lr.score(X, y))

#print(lr.compute_cost(X,y))

lr.gradient_descent(alpha=0.01, itr=1500, plot=True)
#print(lr.score(X, y))
print(lr.compute_cost(X, y))

#print(lr.predict(X))

#lr.plot()
