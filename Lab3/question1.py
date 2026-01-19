#!/usr/bin/python3

""" Implement a linear regression model using scikit-learn for the simulated dataset -
simulated_data_multiple_linear_regression_for_ML.csv  - to predict the “disease_score” from multiple
clinical parameters. """

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
data.head()

#Load data
def load_data():
    X = data.drop("disease_score", axis=1)
    y = data["disease_score"]
    return X, y

def main():
    [X, y] = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=250)
    print("X_train shape:", X_train.shape),
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Standardize the data
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)

    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train_scale, y_train)

    # Test the model
    y_predict = model.predict(X_test_scale)
    print("The prediction after training the model is:", y_predict)
    mse = mean_squared_error(y_test, y_predict)
    print("The mean squared error is:", mse)
    r2 = r2_score(y_test, y_predict)
    print("The r2 score is:", r2)
    print("Done")
    print(X.shape)
    print(y.shape)


if __name__ == '__main__':
    main()