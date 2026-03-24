#!/usr/bin/env python3

""" Implement decision tree classifier without using scikit-learn using the iris dataset.
Fetch the iris dataset from scikit-learn library.
"""

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


""" Load the iris dataset from scikit-learn library."""
def loadData():

    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return df


""" Performing EDA """
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


""" Forming x and y """
def x_y(data):

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return x, y


""" splitting data into train and test """
def split_data(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    return x_train, y_train, x_test, y_test


""" Preprocessing """
def preprocessing(x_train, x_test):

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled


""" STEP - 1 Compute root entropy """
def compute_root_entropy(y_train):

    outcome, total = np.unique(y_train, return_counts=True )
    probability = total / np.sum(total)

    entropy_root = -np.sum(probability * np.log2(probability))

    return entropy_root


""" STEP - 2 Computing Information Gain"""
def information_gain(X_column, y_train, threshold):

    # 1. Parent entropy
    parent_entropy = compute_root_entropy(y_train)

    # 2. Split using loop
    left_y = []
    right_y = []

    for i in range(len(X_column)):
        if X_column[i] <= threshold:
            left_y.append(y_train[i])
        else:
            right_y.append(y_train[i])

    # Convert to numpy arrays
    left_y = np.array(left_y)
    right_y = np.array(right_y)

    # 3. Handle empty split
    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    # 4. Calculate child entropy
    n = len(y_train)
    n_left = len(left_y)
    n_right = len(right_y)

    entropy_left = compute_root_entropy(left_y)
    entropy_right = compute_root_entropy(right_y)

    child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

    # 5. Information Gain
    ig = parent_entropy - child_entropy

    return ig


""" Selecting split value """
def best_split(x_train, y_train):

    best_ig = -1
    best_feature = None
    best_threshold = None

    for feature in range(x_train.shape[1]):

        X_column = x_train[:, feature]

        for value in X_column:   # iterating through training values

            ig = information_gain(X_column, y_train, value)

            if ig > best_ig:
                best_ig = ig
                best_feature = feature
                best_threshold = value

    return best_feature, best_threshold

def leaf_node(y_train):

    labels, counts = np.unique(y_train, return_counts=True)
    return labels[np.argmax(counts)]


def build_tree(x_train, y_train, depth=0, max_depth=5):

    # stopping condition
    if len(np.unique(y_train)) == 1 or depth >= max_depth:
        return leaf_node(y_train)

    feature, threshold = best_split(x_train, y_train)

    if feature is None:
        return leaf_node(y_train)

    left_idx = x_train[:, feature] <= threshold
    right_idx = x_train[:, feature] > threshold

    left_subtree = build_tree(x_train[left_idx], y_train[left_idx], depth+1, max_depth)
    right_subtree = build_tree(x_train[right_idx], y_train[right_idx], depth+1, max_depth)

    return {
        "feature": feature,
        "threshold": threshold,
        "left": left_subtree,
        "right": right_subtree
    }

""" Prediction of sample"""
def predict_sample(x, tree):

    if not isinstance(tree, dict):
        return tree

    if x[tree["feature"]] <= tree["threshold"]:
        return predict_sample(x, tree["left"])
    else:
        return predict_sample(x, tree["right"])


""" Predict """
def predict(X, tree):
    return np.array([predict_sample(x, tree) for x in X])


""" Checking accuracy """
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


""" Model Training """
def modelTraining(x_train, y_train, x_test, y_test):

    tree = build_tree(x_train, y_train, max_depth=5)

    y_pred = predict(x_test, tree)

    acc = accuracy(y_test, y_pred)

    print("\nAccuracy:", acc)


""" Main Function """
if __name__ == "__main__":

    data = loadData()

    eda(data)

    x, y = x_y(data)

    x_train, y_train, x_test, y_test = split_data(x, y)

    x_train, x_test = preprocessing(x_train, x_test)

    modelTraining(x_train, y_train, x_test, y_test)










