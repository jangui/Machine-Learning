import numpy as np
from LogisticRegression import LogisticRegression

data = np.array(np.genfromtxt('ex1data2.txt', delimiter=','))
X = data[:,0:-1]
#X = X.reshape(X.shape[0], 1)
y = data[:,-1]
y = y.reshape(y.shape[0], 1)

lr = LogisticRegression(X, y, normalize=True)
