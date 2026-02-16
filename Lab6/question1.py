#!/usr/bin/env python3

""" K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods """

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression

""" to load data """
def loadData():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data


""" Giving x and y values """
def x_y_form(data):
    x = data.drop(["disease_score", "disease_score_fluct"], axis=1).values   #droping the target value which is y
    y = data["disease_score_fluct"].values                #only  the target value #axis 1 is for column
    y = y.reshape(-1, 1)                                  #Reshaping y so it acts like a matrix (column vector)
    return x, y


""" Standardizing the data """
def standardization(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    #Prevent division by zero
    std[std == 0] = 1

    #standardizing both train and test data
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    return x_train_std, x_test_std


""" Splitting the data into training and test sets """
def splitData(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=250)
    print("X_train shape:", x_train.shape),
    print("X_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return x_train, x_test, y_train, y_test


""" Adding bias features """
def addBias(x):
    bias = np.mean(x, axis=0)
    return np.c_[np.ones((x.shape[0], 1)), x]      #add column of ones for x (linear regression needs bias term )


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


""" k-fold cross validation from scratch """
def kfold_scratch(x, y, k=10):
    n = len(x)
    fold_size = n // k         #saving all the folds in fold_size
    indices = np.arange(n)
    np.random.shuffle(indices)

    mse_list = []

    print(f"\nK-Fold Cross Validation From Scratch (K={k})")
    print("-" * 50)

    for fold in range(k):
        # get test indices for current fold
        start = fold * fold_size
        end = start + fold_size if fold < k - 1 else n

        test_index = indices[start:end]    #from start to end
        train_index = np.concatenate([indices[:start], indices[end:]])         #remaining samples

        # split data
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]

        # standardize
        x_train_std, x_test_std = standardization(x_train, x_test)

        # add bias
        x_train_bias = addBias(x_train_std)
        x_test_bias = addBias(x_test_std)

        # initialize theta
        theta = np.zeros((x_train_bias.shape[1], 1))

        # train
        print(f"\nFold {fold + 1}:")
        theta = gradient_descent(x_train_bias, y_train, theta, alpha=0.01, iterations=500)

        # predict
        y_pred = np.dot(x_test_bias, theta)

        # calculate MSE
        mse = np.mean((y_pred - y_test) ** 2)
        mse_list.append(mse)
        print(f"Fold {fold + 1} MSE: {mse:.4f}")

    print("\n" + "=" * 50)
    print(f"Average MSE: {np.mean(mse_list):.4f}")
    print(f"Std MSE: {np.std(mse_list):.4f}")
    print("=" * 50)

    return mse_list


""" K-fold cross validation using sklearn """
def kfold_sklearn(x, y, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mse_list = []

    print(f"\nK-Fold Cross Validation Using Sklearn (K={k})")
    print("-" * 50)

    fold = 1
    for train_idx, test_idx in kf.split(x):
        # split data
        x_train = x[train_idx]
        y_train = y[train_idx].ravel()
        x_test = x[test_idx]
        y_test = y[test_idx].ravel()

        # standardize
        x_train_std, x_test_std = standardization(x_train, x_test)

        # train model
        model = LinearRegression()
        model.fit(x_train_std, y_train)

        # predict
        y_pred = model.predict(x_test_std)

        # calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
        print(f"Fold {fold} MSE: {mse:.4f}")

        fold += 1

    print("\n" + "=" * 50)
    print(f"Average MSE: {np.mean(mse_list):.4f}")
    print(f"Std MSE: {np.std(mse_list):.4f}")
    print("=" * 50)

    return mse_list


""" Main """
def main():
    # load data
    data = loadData()
    x, y = x_y_form(data)

    print(f"Data loaded: x shape = {x.shape}, y shape = {y.shape}")

    # k-fold from scratch
    mse_scratch = kfold_scratch(x, y, k=10)

    # k-fold sklearn
    mse_sklearn = kfold_sklearn(x, y, k=10)

    print(f"\n\nComparison:")
    print(f"Scratch Avg MSE: {np.mean(mse_scratch):.4f}")
    print(f"Sklearn Avg MSE: {np.mean(mse_sklearn):.4f}")


if __name__ == "__main__":
    main()



