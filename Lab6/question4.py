#!/usr/bin/python3

""" Use validation set to do feature and model selection."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


"""Load data from CSV"""
def loadData():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data


"""Extract features (X) and target (y)"""
def x_y_form(data):
    x = data.drop(["disease_score", "disease_score_fluct"], axis=1).values
    y = data["disease_score_fluct"].values
    return x, y


"""Split data into train and test sets"""
def splitData(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=250)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    return x_train, x_test, y_train, y_test


""" Split into train and validation sets """
def train_val_split(x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=250)
    print(f"Train: {x_train.shape}, Val: {x_val.shape}")
    return x_train, x_val, y_train, y_val


"""Standardize features using train statistics"""
def standardization(x_train, x_test):
    
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0] = 1  # Prevent division by zero

    x_train_std = (x_train - mean) / std    #formula of standardization  (for train set)
    x_test_std = (x_test - mean) / std      #standardzation for test set

    return x_train_std, x_test_std


"""Compare multiple models and select the best one"""
def modelSelection(x_train, y_train):

    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=250)
    }

    print("\n=== Model Selection Results ===")

    best_score = -np.inf
    # -inf = negative infinity, means any r2 score even in negative values is greater than it
    # so the first model would be best and then update later

    best_model_name = None        #best model placeholder starts with "None"

    # Test each model
    for name, model in models.items():
        scores = cross_val_score(model, x_train, y_train, cv=10, scoring='r2')
        #cv means cross validation folds
        #The model trains on 9 folds and is tested on the remaining 1 fold

        mean_score = scores.mean()
        std_score = scores.std()

        print(f"{name}: {mean_score:.3f} (+/- {std_score:.3f})")

        # Track best model
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with R² = {best_score:.3f}")
    return best_model_name


""" Main """
def main():
    #Load data
    data = loadData()
    
    #Forming x and y
    x, y = x_y_form(data)

    #splitting data
    x_train, x_test, y_train, y_test = splitData(x, y)

    #Splitting validation set
    x_train, x_val, y_train, y_val = train_val_split(x_train, y_train)

    #Standardize(val set and train set)
    x_train_std, x_val_std = standardization(x_train, x_val)

    #Select best model
    best_model_name = modelSelection(x_train_std, y_train)



if __name__ == "__main__":
    main()

