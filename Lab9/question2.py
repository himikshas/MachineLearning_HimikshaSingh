#!/usr/bin/python3

""" Implement a regression decision tree algorithm using
    scikit-learn for the simulated dataset.
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



""" Load Dataset """
def loadData():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    return data


""" Form X and Y """
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
        x, y, test_size=0.2, random_state=42 )

    return x_train, x_test, y_train, y_test


""" Train Model """
def train_model(x_train, y_train):

    model = DecisionTreeRegressor(
        max_depth=4,   #limits depth of tree
        random_state=42
    )
    model.fit(x_train, y_train)
    return model


""" Evaluate Model """
def evaluate_model(model, x_test, y_test):

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)


""" Main Function """
def main():

    # Load dataset
    data = loadData()

    # Form X and Y
    x, y = x_y_form(data)

    # Split data
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Train regression tree
    model = train_model(x_train, y_train)

    # Evaluate model
    evaluate_model(model, x_test, y_test)


# Run Program
if __name__ == "__main__":
    main()