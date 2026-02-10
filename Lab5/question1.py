#!/usr/bin/env python3

""" Implement Stochastic Gradient Descent algorithm from scratch """

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

""" to load data """
def loadData():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data


""" Giving x and y values """
def x_y_form(data):
    x = data.drop("disease_score_fluct", axis=1).values   #droping the target value which is y
    y = data["disease_score_fluct"].values                #only  the target value #axis 1 is for column
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


""" adding bias """
def addBias(x):
    m = x.shape[0]
    bias = np.ones((m, 1))
    return np.hstack((bias, x))


""" Normalizing the Data """
def normalize(x):
    m = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    # avoid division by zero
    sigma[sigma == 0] = 1

    x_norm = (x - m) / sigma
    return x_norm, m, sigma


""" Stochastic Gradient Descent """
def stochastic_gradient_descent(x, y, alpha=0.01, iterations=100):
    x = addBias(x)               #adding bias
    m, n = x.shape               #shaping the matrix

    theta = np.zeros(n)                #creating a theta as vector initialised with zeroes
    y = y.flatten()                    #convert y into 1D array

    for iteration in range(iterations):
        for i in range(m):
            y_pred = np.dot(x[i], theta)          #hypothesis function h(θ) = theta * x
            error = y_pred - y[i]                 #calculating the error, error = h(θ) - y
            theta = theta - alpha * error * x[i]         #updating theta

    return theta.reshape(-1, 1)               #will convert 1D array into 2D column vector


""" r2 score """
def r2Score(y_true, y_pred):
    a = np.sum((y_true - y_pred) ** 2)
    b = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (a / b)


""" main """
def main():
    data = loadData()       #load the dataset
    x, y = x_y_form(data)      #separate features(x) and target (y)
    x_train, x_test, y_train, y_test = splitData(x, y)         #split into training and testing data

    x_train_norm, mean, std = normalize(x_train)       #normalize training features
    x_test_norm = (x_test - mean) / std          #Normalize test features using training mean/std

    alpha = 0.01              #set stochastic gradient hyperparameters
    iterations = 50

    theta_final = stochastic_gradient_descent(x_train_norm, y_train, alpha, iterations)          #Train using stochastic gradient descent

    x_test_bias = addBias(x_test_norm)           #add bias to test set
    y_test_pred = np.dot(x_test_bias, theta_final)          #predict test outputs

    mse = np.mean((y_test - y_test_pred) ** 2)              #calculate mean squared error
    r2 = r2Score(y_test, y_test_pred)            #calculate R2 score

    print("\nFinal theta:")
    print(theta_final.flatten())
    print("\nTest MSE:", mse)
    print("Test R2 score:", r2)


if __name__ == "__main__":
    main()