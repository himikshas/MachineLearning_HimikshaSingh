#!/usr/bin/env python3

""" Implement Random Forest algorithm for regression and
    classification using scikit-learn. Use diabetes and iris datasets. """

import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


"""                          ###### RANDOM FOREST REGRESSION ######                  """


""" Loading the dataset """
def loadData():
    data = load_diabetes()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return df


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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 45 )

    return x_train, x_test, y_train, y_test



""" Standardizing the dataset"""
def preProcessing(x_train, x_test):

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled



""" Training the model """
def trainModel(x_train_scaled, y_train):

    base_model = DecisionTreeRegressor()

    model = BaggingRegressor(
        estimator = base_model,
        n_estimators = 100,
        random_state = 45
    )
    model.fit(x_train_scaled, y_train)

    return model



""" Evaluation of the trained model"""
def evaluateModel(model, x_test_scaled, y_test):

    y_pred = model.predict(x_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean squared error of the dataset: ", mse)
    print("r2 score of the dataset is: ", r2)



""" Main function """
def main():

    #1. Load the dataset
    data = loadData()

    #2. Do eda on the dataset
    data = eda(data)

    #3. Form x and y
    x, y = x_y_form(data)

    #4. Train and test split of dataset
    x_train, x_test, y_train, y_test = split_data(x, y)

    #5. Data Pre-processing
    x_train_scaled, x_test_scaled = preProcessing(x_train, x_test)

    #6. Train the model
    model = trainModel(x_train_scaled, y_train)

    #7. Evaluating or testing the model
    evaluateModel(model, x_test_scaled, y_test)



""" Run the program """
if __name__ == "__main__":
    main()