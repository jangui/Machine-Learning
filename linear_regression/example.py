#!/usr/bin/env python3
from linear_regression import LinearRegression
import numpy as np
import os

########## single variable linear regression ##########
print("\nSingle Variable Linear Regression")
file_name = os.path.join('data', 'data1.txt')
data = np.array(np.genfromtxt(file_name, delimiter=','))

X = data[:,0]
y = data[:,1]


### Normal Equation ###
lr = LinearRegression(X, y, normalize=False)

print("\nUsing Normal Equation")
print(f"Starting Cost: {lr.calc_cost()}")
lr.normal_equation()

print(f"Final Cost: {lr.calc_cost()}")
print(f"Optimal parameters:\n{lr.theta}")
lr.plot()

######################


### Gradient Descent ###
lr = LinearRegression(X, y, normalize=False)
learning_rate = 0.01
iterations = 1500

print("\nUsing Gradient Descent")
print(f"Learning Rate: {learning_rate} Iterations: {iterations}")
print(f"Starting Cost: {lr.calc_cost()}")
lr.gradient_descent(learning_rate, iterations, plot=True)

print(f"Final Cost : {lr.calc_cost()}")
print(f"Optimal parameters:\n{lr.theta}")
lr.plot()

#######################

########## multivariate linear regression ##########
print("\nMultivariate Linear Regression")
file_name = os.path.join('data', 'data2.txt')
data = np.array(np.genfromtxt(file_name, delimiter=','))

X = data[:,0:2]
y = data[:,2]


### Normal Equation ###
lr = LinearRegression(X, y, normalize=True)
#Note: Normalization not needed when using normal equation
#Only normalizing to compare results vs gradient descent

print("\nUsing Normal Equation")
print(f"Starting Cost: {lr.calc_cost()}")
lr.normal_equation()

print(f"Final Cost: {lr.calc_cost()}")
print(f"Optimal parameters:\n{lr.theta}")
lr.plot()

######################

### Gradient Descent ###
lr = LinearRegression(X, y, normalize=True)
learning_rate = 0.01
iterations = 1500

print("\nUsing Gradient Descent")
print(f"Learning Rate: {learning_rate} Iterations: {iterations}")
print(f"Starting Cost: {lr.calc_cost()}")
lr.gradient_descent(learning_rate, iterations, plot=True)

print(f"Final Cost: {lr.calc_cost()}")
print(f"Optimal parameters:\n{lr.theta}")
lr.plot()

########################
