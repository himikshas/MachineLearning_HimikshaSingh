#!/usr/bin/env python3

""" Implement L2-norm and L1-norm from scratch """

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd



""" Load dataset """
def loadData():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data


""" Separate X and Y """
def x_y_form(data):

    # Features
    x = data.drop(["disease_score_fluct"], axis=1).values

    # Target
    y = data["disease_score_fluct"].values

    # Convert to column vector
    y = y.reshape(-1, 1)

    return x, y


""" Train Test Split """
def split_data(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42
    )

    return x_train, x_test, y_train, y_test


""" Add Bias Column """
def addBias(x):

    m = x.shape[0]  #number of samples(rows)
                    #x.shape gives (rows, columns)

    bias = np.ones((m, 1))       #create a column of 1s

    return np.hstack((bias, x))   #attach bias column to feature matrix
    #done only to make h theta(x) = theta 0 * x 0 + theta 1 * x 1 + ...

""" Standardization """
def standardization(x_train, x_test):

    mean = np.mean(x_train, axis=0)     #gives column wise mean (vertically)

    std = np.std(x_train, axis=0)      #calculate standard deviation (sigma) column wise(vertically)

    std[std == 0] = 1                #avoid division by zero

    x_train_std = (x_train - mean) / std      #x_std = (x - mean) / sigma
    x_test_std = (x_test - mean) / std        #test data must use training mean and standard deviation(sigma) otherwise would leake data


    return x_train_std, x_test_std


""" L2 Norm (Ridge Cost Function) """
def L2Norm(theta, x, y, lam=0.1):

    n = len(x)         #gives number of training samples

    h = np.dot(x, theta)    #hypothesis function ( h theta(x) = x * theta )

    error = h - y         #compute error ( error = hypothesis value(predicted value) - actual values )

    mse_cost = (1 / (2 * n)) * np.sum(error ** 2)   #compute mean square error cost

    # Exclude bias term as its not regularised in l2 penalty
    l2_penalty = (lam / (2 * n)) * np.sum(theta[1:] ** 2) #theta[1:] = all other weights

    return mse_cost + l2_penalty



""" L1 Norm (Lasso Cost Function) """
def L1Norm(theta, x, y, lam=0.1):

    n = len(x)

    h = np.dot(x, theta)

    error = h - y

    mse_cost = (1 / (2 * n)) * np.sum(error ** 2)

    # Exclude bias
    l1_penalty = (lam / (2 * n)) * np.sum(np.abs(theta[1:]))  #everything is same as l2 other than squaring the thetas

    return mse_cost + l1_penalty


""" Main Function """
def main():

    # Load Data
    data = loadData()

    # Get X and Y
    x, y = x_y_form(data)

    # Train Test Split
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Standardization
    x_train_std, x_test_std = standardization(x_train, x_test)

    # Add Bias (for both training and test set)
    x_train_b = addBias(x_train_std)
    x_test_b = addBias(x_test_std)

    # Initialize Theta
    num_features = x_train_b.shape[1]     #gets number of columns (features) with bias added

    theta = np.zeros((num_features, 1))    #all theta starts from 0

    # Compute Costs
    l2_cost = L2Norm(theta, x_train_b, y_train, lam=0.1)

    l1_cost = L1Norm(theta, x_train_b, y_train, lam=0.1)

    # Output
    print("\n----- Model Information -----\n")

    print(f"Training samples : {x_train_b.shape[0]}")   #get training size
    print(f"Test samples     : {x_test_b.shape[0]}")    #get test samples
    print(f"Number of features (incl bias): {num_features}")

    print("\n----- Regularization Cost -----\n")

    print(f"Initial L2 (Ridge) Cost : {l2_cost:.4f}")
    print(f"Initial L1 (Lasso) Cost : {l1_cost:.4f}")


if __name__ == "__main__":
    main()