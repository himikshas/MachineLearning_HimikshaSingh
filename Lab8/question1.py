#!/usr/bin/env python3

""" Implement L2-norm and L1-norm from scratch """

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


""" to load data """
def loadData():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data


""" Giving x and y values """
def x_y_form(data):
    x = data.drop("disease_score_fluct", axis=1).values   # dropping the target value which is y
    y = data["disease_score_fluct"].values                 # only the target value
    y = y.reshape(-1, 1)                                   # reshaping y into a column vector
    return x, y


""" Splitting the data into test and train data """
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


""" Adding bias """
def addBias(x):
    m = x.shape[0]
    bias = np.ones((m, 1))
    return np.hstack((bias, x))


""" Normalizing the data """
def normalize(x):
    m = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma == 0] = 1    #to avoid division by zero
    x_norm = (x - m) / sigma
    return x_norm, m, sigma


""" Standardizing the data """
def standardization(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    # Prevent division by zero
    std[std == 0] = 1

    # Standardize both train and test using train statistics only
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    return x_train_std, x_test_std


""" L2 norm (Ridge regularization term) """
def L2Norm(theta, x, y, lam=0.1):
    n = len(x)
    h = np.dot(x, theta)           # hypothesis: shape (n, 1)
    error = h - y                  # residuals
    cost = (1 / (2 * n)) * np.sum(error ** 2)
    l2_norm = lam * np.sum(theta ** 2)   # L2 penalty (excludes bias term by convention)
    return cost + l2_norm


""" L1 norm (Lasso regularization term) """
def L1Norm(theta, x, y, lam=0.1):
    n = len(x)
    h = np.dot(x, theta)           # hypothesis: shape (n, 1)
    error = h - y                  # residuals
    cost = (1 / (2 * n)) * np.sum(error ** 2)
    l1_norm = lam * np.sum(np.abs(theta))  # L1 penalty
    return cost + l1_norm


""" main """
def main():
    # 1. Load and prepare data
    data = loadData()
    x, y = x_y_form(data)

    # 2. Split into train/test sets
    x_train, x_test, y_train, y_test = split_data(x, y)

    # 3. Standardize features (fit on train, apply to both)
    x_train_std, x_test_std = standardization(x_train, x_test)

    # 4. Add bias column (column of 1s) to both sets
    x_train_b = addBias(x_train_std)
    x_test_b  = addBias(x_test_std)

    # 5. Initialize theta (weights) to zeros
    num_features = x_train_b.shape[1]
    theta = np.zeros((num_features, 1))

    # 6. Compute and display regularized costs
    l2_cost = L2Norm(theta, x_train_b, y_train, lam=0.1)
    l1_cost = L1Norm(theta, x_train_b, y_train, lam=0.1)

    print(f"Initial L2 (Ridge) cost: {l2_cost:.4f}")
    print(f"Initial L1 (Lasso) cost: {l1_cost:.4f}")
    print(f"Training samples : {x_train_b.shape[0]}")
    print(f"Test samples     : {x_test_b.shape[0]}")
    print(f"Number of features (incl. bias): {num_features}")


if __name__ == "__main__":
    main()