#!/usr/bin/env python3

""" Implement Adaboost classifier without using scikit-learn. Use the Iris dataset. """

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


""" Loading the dataset """
def loadData():
    dataset = load_iris()

    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    data["target"] = dataset.target

    return data


""" EDA of the dataset """
def eda(data):

    print("Shape of Dataset :", data.shape)

    print("\nData description:")
    print(data.describe())

    print("\nInformation about Dataset :")
    print(data.info())

    print("\nFirst five rows:")
    print(data.head())

    print("\nLast five rows:")
    print(data.tail())

    print("\nMissing values:")
    print(data.isnull().sum())

    return data


""" Forming x and y"""
def x_y_form(data):

    x = data.drop(["target"], axis=1)
    y = data["target"]

    return x, y


""" Splitting data into train and test sets """
def split_data(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42 )

    return x_train, x_test, y_train, y_test
