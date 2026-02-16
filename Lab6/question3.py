#!/usr/bin/python3


""" Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch."""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



""" to load data """
def loadData():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data


""" Giving x and y values """
def x_y_form(data):
    x = data.drop(["disease_score", "disease_score_fluct"], axis=1).values  # dropping the target value which is y
    y = data["disease_score_fluct"].values  # only  the target value #axis 1 is for column
    y = y.reshape(-1, 1)  # Reshaping y so it acts like a matrix (column vector)
    return x, y


""" Normalizing the Data - Scale between 0 and 1 """
def normalization(x_train, x_test):
    # get min and max from training data
    x_min = np.min(x_train, axis=0)
    x_max = np.max(x_train, axis=0)

    # avoid division by zero
    range_value = x_max - x_min
    range_value[range_value == 0] = 1

    # normalize: x_new = (x - min) / (max - min)
    x_train_norm = (x_train - x_min) / range_value
    x_test_norm = (x_test - x_min) / range_value

    return x_train_norm, x_test_norm


""" Standardizing the data """
def standardization(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    # Prevent division by zero
    std[std == 0] = 1

    # standardizing both train and test data
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    return x_train_std, x_test_std


""" main """
def main():
    data = loadData()
    x, y = x_y_form(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # normalization
    x_train_norm, x_test_norm = normalization(x_train, x_test)
    print("Normalized data (0 to 1):")
    print(x_train_norm)
    print("\n" + "="*50 + "\n")

    # standardization
    x_train_std, x_test_std = standardization(x_train, x_test)
    print("Standardized data (mean=0, std=1):")
    print(x_train_std)


if __name__ == "__main__":
    main()