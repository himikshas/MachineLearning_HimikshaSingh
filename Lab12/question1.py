#!/usr/bin/env python3

""" Implement a decision regression tree algorithm without using scikit-learn using the diabetes dataset.
Fetch the dataset from scikit-learn library. """

#!/usr/bin/env python3

""" Implement decision tree regressor without using scikit-learn using the diabetes dataset.
Fetch the diabetes dataset from scikit-learn library.
"""

from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


""" Load the diabetes dataset """
def loadData():

    data = load_diabetes()
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


""" STEP - 1 Compute variance (MSE) """
def compute_variance(y):

    if len(y) == 0:
        return 0

    mean = np.mean(y)
    variance = np.mean((y - mean) ** 2)

    return variance


""" STEP - 2 Compute Variance Reduction """
def variance_reduction(X_column, y_train, threshold):

    parent_variance = compute_variance(y_train)

    left_y = []
    right_y = []

    for i in range(len(X_column)):
        if X_column[i] <= threshold:
            left_y.append(y_train[i])
        else:
            right_y.append(y_train[i])

    left_y = np.array(left_y)
    right_y = np.array(right_y)

    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    n = len(y_train)
    n_left = len(left_y)
    n_right = len(right_y)

    var_left = compute_variance(left_y)
    var_right = compute_variance(right_y)

    weighted_variance = (n_left / n) * var_left + (n_right / n) * var_right

    vr = parent_variance - weighted_variance

    return vr


""" Selecting best split """
def best_split(x_train, y_train):

    best_vr = -1
    best_feature = None
    best_threshold = None

    for feature in range(x_train.shape[1]):

        X_column = x_train[:, feature]

        for value in X_column:

            vr = variance_reduction(X_column, y_train, value)

            if vr > best_vr:
                best_vr = vr
                best_feature = feature
                best_threshold = value

    return best_feature, best_threshold


""" Leaf node (mean value) """
def leaf_node(y_train):
    return np.mean(y_train)


""" Build tree """
def build_tree(x_train, y_train, depth=0, max_depth=5):

    if len(y_train) == 0 or depth >= max_depth:
        return leaf_node(y_train)

    feature, threshold = best_split(x_train, y_train)

    if feature is None:
        return leaf_node(y_train)

    left_idx = x_train[:, feature] <= threshold
    right_idx = x_train[:, feature] > threshold

    if len(y_train[left_idx]) == 0 or len(y_train[right_idx]) == 0:
        return leaf_node(y_train)

    left_subtree = build_tree(x_train[left_idx], y_train[left_idx], depth+1, max_depth)
    right_subtree = build_tree(x_train[right_idx], y_train[right_idx], depth+1, max_depth)

    return {
        "feature": feature,
        "threshold": threshold,
        "left": left_subtree,
        "right": right_subtree
    }


""" Prediction of single sample """
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


""" Evaluation (MSE) """
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


""" Model Training """
def modelTraining(x_train, y_train, x_test, y_test):

    tree = build_tree(x_train, y_train, max_depth=5)

    y_pred = predict(x_test, tree)

    error = mse(y_test, y_pred)

    print("\nMSE:", error)


""" Main Function """
if __name__ == "__main__":

    data = loadData()

    eda(data)

    x, y = x_y(data)

    x_train, y_train, x_test, y_test = split_data(x, y)

    x_train, x_test = preprocessing(x_train, x_test)

    modelTraining(x_train, y_train, x_test, y_test)