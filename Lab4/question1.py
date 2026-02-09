#!/usr/bin/env python3

""" Implement gradient descent from scratch  """

import numpy as np

def loadData():
    x = np.array([10, 20, 30, 40, 50])
    y = np.array([234.2, 479.4, 645.3, 700.2, 989.1])
    return x, y

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)

    for i in range(iterations):
        # hypothesis hθ = Xθ
        y_pred = np.dot(X, theta)

        # gradient = (1/m) Xᵀ(hθ − y)
        gradient = (1/m) * np.dot(X.T, (y_pred - y))

        # update theta
        theta = theta - alpha * gradient

        # cost function
        cost = (1/(2*m)) * np.sum((y_pred - y) ** 2)

        if i % 100 == 0:
            print(f"Iteration {i} | Cost: {cost:.4f}")

    return theta

def r2Score(y_true, y_pred):
    a = np.sum((y_true - y_pred) ** 2)
    b = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (a / b)

def main():
    x, y = loadData()

    # add bias term (column of ones)
    X = np.c_[np.ones(len(x)), x]   # shape (m, 2)

    # initialize theta (θ0, θ1)
    theta = np.zeros(2)

    alpha = 0.001
    iterations = 2000

    theta_final = gradient_descent(X, y, theta, alpha, iterations)

    # predictions
    y_pred = np.dot(X, theta_final)

    # R² score
    r2 = r2Score(y, y_pred)

    print("\nFinal theta:", theta_final)
    print("R² score:", r2)

if __name__ == "__main__":
    main()
