#!/usr/bin/env python3

""" Implement logistic regression using scikit-learn for the breast cancer dataset -
 https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data """

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


""" Loading the data """
def load():
    cancer_data = pd.read_csv('data.csv')
    return cancer_data


""" Getting x and y values """
def x_y_form(data):
    # Drop useless columns (Unnamed: 32 is full of NaN)
    data = data.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")

    x = data.drop("diagnosis", axis=1).values
    y = data["diagnosis"].values

    return x, y


""" Splitting the data into test and train data """
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


""" Normalizing the data """
def normalize(x):
    m = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    sigma[sigma == 0] = 1

    x_norm = (x - m) / sigma
    return x_norm, m, sigma


""" Training the model """
def train_model(x_train, y_train):
    model = LogisticRegression(max_iter=10000)         #creates logistic regression model object from sklearn
    model.fit(x_train, y_train)             #trains model, finds best parameters for logistic regression and learns coefficient
    return model


""" Evaluating the model """
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)                 #.predict computes probabilities internally
    accuracy = accuracy_score(y_test, y_pred)        #Accuracy = Number of correct predictions/Total predictions
    conf_matrix = confusion_matrix(y_test, y_pred)            #just contingency table
    return accuracy, conf_matrix


""" The main function """
def main():
    data = load()
    x, y = x_y_form(data)

    x_train, x_test, y_train, y_test = split_data(x, y)

    # Normalize using ONLY training mean/std
    x_train_norm, mean, std = normalize(x_train)
    x_test_norm = (x_test - mean) / std

    # Train + test on normalized data
    model = train_model(x_train_norm, y_train)
    accuracy, conf_matrix = evaluate_model(model, x_test_norm, y_test)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    main()
