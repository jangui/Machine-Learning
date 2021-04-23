import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main():
    print("Adversiting Data")
    advertising_data = pd.read_csv("./data/advertising.csv")
    linear_regression(advertising_data, ['TV'], 'Sales')
    print()

    print("Example 2D Data")
    advertising_data = pd.read_csv("./data/ex_data.csv")
    linear_regression(advertising_data, ['x1', 'x2'], 'y')
    print()

def linear_regression(df: pd.DataFrame, x_data: list[str], y_data: str, verbose=True, plot=True):
    lin_reg = LinearRegression(normalize = True)
    X = df[x_data]
    y = df[[y_data]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lin_reg.fit(X, y)

    coeficients = lin_reg.coef_
    intercept = lin_reg.intercept_
    y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = lin_reg.score(X, y)

    if verbose:
        print(f"Coeficients: {coeficients}")
        print(f"Intercept: {intercept}")
        print(f"R^2: {r_squared}")
        print(f"MSE: {mse}")

    if plot:
        plot_data(X, y, lin_reg)

    return coeficients, intercept

def plot_data(X, y, lin_reg):
    dimensions = len(X.columns) + 1
    if dimensions > 3:
        return

    if dimensions == 2:
        plt.xlabel(X.columns[0])
        plt.ylabel(y.columns[0])
        plt.title(f"{X.columns[0]} vs. {y.columns[0]}")

        # plot data
        plt.scatter(X, y)

        # plot regression line
        y_pred = X.dot(lin_reg.coef_[0]) + lin_reg.intercept_
        plt.plot(X, y_pred, color='r')

    elif dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_column = X.columns[0]
        y_column = X.columns[1]
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_zlabel(y.columns[0])

        # plot data
        ax.scatter(X[[x_column]], X[[y_column]], y)

        # plot regression plane
        coefs = lin_reg.coef_[0]
        normal = [coefs[0], coefs[1], -1]
        xx, yy = np.meshgrid(X[[x_column]], X[[y_column]])
        z = (-normal[0] * xx - normal[1] * yy - lin_reg.intercept_) * 1. /normal[2]
        y_pred = X.dot(lin_reg.coef_[0]) + lin_reg.intercept_
        ax.plot_surface(xx, yy, z, color='r')

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
