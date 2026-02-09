#!/usr/bin/python3

""" Use your implementation and train ML models for both californiahousing
 and simulated datasets and compare your results with the scikit-learn models."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


""" to load data """
def loadData():
    data = fetch_california_housing(as_frame=True)
    return data.frame


""" Giving x and y values """
def x_y_form(data):
    x = data.drop("MedHouseVal", axis=1).values   #droping the target value which is y
    y = data["MedHouseVal"].values                #only  the target value #axis 1 is for column
    y = y.reshape(-1, 1)                                  #Reshaping y so it acts like a matrix (column vector)
    return x, y


""" Splitting the data into training and test sets """
def splitData(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=250)
    print("X_train shape:", x_train.shape),
    print("X_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return x_train, x_test, y_train, y_test


""" Normalizing the Data """
def normalize(x):
    m = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    # avoid division by zero
    sigma[sigma == 0] = 1

    x_norm = (x - m) / sigma
    return x_norm, m, sigma  


"""Gradient Descent """
def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)

    for i in range(iterations):
        # hypothesis hθ = Xθ
        y_pred = np.dot(x, theta)

        # gradient = (1/m) Xᵀ(hθ − y)
        gradient = (1/m) * np.dot(x.T, (y_pred - y))

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


""" Main function """
def main():
        data = loadData()
        x, y = x_y_form(data)

        # ADD THIS: Split the data
        x_train, x_test, y_train, y_test = splitData(x, y)

        # Normalize TRAINING data
        x_train_norm, mean, std = normalize(x_train)
        x_test_norm = (x_test - mean) / std

        # Add bias term
        X_train = np.c_[np.ones(len(x_train_norm)), x_train_norm]
        X_test = np.c_[np.ones(len(x_test_norm)), x_test_norm]

        theta = np.zeros((X_train.shape[1], 1))

        alpha = 0.01
        iterations = 2000

        # Train on training set
        theta_final = gradient_descent(X_train, y_train, theta, alpha, iterations)

        # Evaluate on test set
        y_test_pred = np.dot(X_test, theta_final)
        r2 = r2Score(y_test, y_test_pred)

        print("\nFinal theta:", theta_final)
        print("Test R² score:", r2)

if __name__ == "__main__":
    main()
