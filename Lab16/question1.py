#!/usr/bin/env python3

""" Write a Python program to aggregate  predictions from multiple
trees to output a final prediction for a regression problem. """


import numpy as np
import pandas as pd
from ISLP import load_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


""" Step 1: Load Dataset """
def load_dataset():
    boston = load_data("Boston")
    df = pd.DataFrame(boston)

    X = df.drop("medv", axis=1)
    y = df["medv"]

    return X, y


""" Step 2: Train-Test Split """
def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



""" Step 3: Bootstrap Sampling """
def bootstrap_sample(X, y):
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X.iloc[indices], y.iloc[indices]


""" Step 4: Train Ensemble of Trees """
def train_ensemble(X_train, y_train, n_trees=20, max_depth=6):
    trees = []

    for i in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X_train, y_train)

        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=i)
        tree.fit(X_sample, y_sample)

        trees.append(tree)

    return trees


""" Step 5: Predict (Aggregation) """
def predict_ensemble(trees, X):
    all_preds = []

    for tree in trees:
        preds = tree.predict(X)
        all_preds.append(preds)

    all_preds = np.array(all_preds)

    # Averaging
    return np.mean(all_preds, axis=0)


""" Step 6: Evaluate Model """
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("Mean Squared Error:", mse)
    print("R² Score:", r2)


""" Step 7: Main Function """
def main():
    # Load data
    X, y = load_dataset()

    #Train test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train ensemble
    trees = train_ensemble(X_train, y_train, n_trees=20)

    # Predict
    y_pred = predict_ensemble(trees, X_test)

    # Evaluate
    evaluate(y_test, y_pred)


if __name__ == "__main__":
    main()