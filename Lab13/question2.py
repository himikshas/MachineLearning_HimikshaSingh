#!/usr/bin/env python3

""" Implement Bagging Regressor without using scikit-learn """

from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


""" Load dataset """
def loadData():

    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return df


""" EDA """
def eda(data):
    print("Shape:", data.shape)
    print("\nDescribe:\n", data.describe())
    print("\nInfo:")
    print(data.info())
    print("\nMissing values:\n", data.isnull().sum())

    return data


""" x and y """
def x_y(data):
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return x, y


""" Split """
def split_data(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    return x_train, y_train, x_test, y_test


""" Preprocessing """
def preprocessing(x_train, x_test):

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test


# ===================== TREE PART =====================

""" STEP: 1 Compute Variance"""
def compute_variance(y):
    if len(y) == 0:
        return 0
    return np.mean((y - np.mean(y)) ** 2)


""" STEP: 2 Reduce Variance"""
def variance_reduction(X_column, y, threshold):

    parent_var = compute_variance(y)

    left_y, right_y = [], []

    for i in range(len(X_column)):
        if X_column[i] <= threshold:
            left_y.append(y[i])
        else:
            right_y.append(y[i])

    left_y, right_y = np.array(left_y), np.array(right_y)

    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    n = len(y)
    n_l, n_r = len(left_y), len(right_y)

    var_l = compute_variance(left_y)
    var_r = compute_variance(right_y)

    weighted = (n_l/n)*var_l + (n_r/n)*var_r

    return parent_var - weighted


""" Best split value """
def best_split(x, y):

    best_vr = -1
    best_feature = None
    best_threshold = None

    for feature in range(x.shape[1]):
        X_column = x[:, feature]

        for value in X_column:
            vr = variance_reduction(X_column, y, value)

            if vr > best_vr:
                best_vr = vr
                best_feature = feature
                best_threshold = value

    return best_feature, best_threshold


""" leaf node"""
def leaf_node(y):
    return np.mean(y)


""" build tree """
def build_tree(x, y, depth=0, max_depth=5):

    if len(y) == 0 or depth >= max_depth:
        return leaf_node(y)

    feature, threshold = best_split(x, y)

    if feature is None:
        return leaf_node(y)

    left_idx = x[:, feature] <= threshold
    right_idx = x[:, feature] > threshold

    if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
        return leaf_node(y)

    left = build_tree(x[left_idx], y[left_idx], depth+1, max_depth)
    right = build_tree(x[right_idx], y[right_idx], depth+1, max_depth)

    return {
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right
    }

""" Predict sample """
def predict_sample(x, tree):

    if not isinstance(tree, dict):
        return tree

    if x[tree["feature"]] <= tree["threshold"]:
        return predict_sample(x, tree["left"])
    else:
        return predict_sample(x, tree["right"])

""" Predict tree """
def predict_tree(X, tree):
    return np.array([predict_sample(x, tree) for x in X])


# ===================== BAGGING =====================

""" Bootstrap sampling """
def bootstrap_sample(x, y):

    n_samples = x.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)

    return x[indices], y[indices]


""" Train multiple trees """
def bagging_train(x_train, y_train, n_estimators=5):

    trees = []

    for _ in range(n_estimators):

        x_sample, y_sample = bootstrap_sample(x_train, y_train)

        tree = build_tree(x_sample, y_sample, max_depth=5)

        trees.append(tree)

    return trees


""" Predict using bagging """
def bagging_predict(X, trees):

    predictions = []

    for tree in trees:
        predictions.append(predict_tree(X, tree))

    predictions = np.array(predictions)

    return np.mean(predictions, axis=0)


""" MSE """
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


""" Training """
def modelTraining(x_train, y_train, x_test, y_test):

    trees = bagging_train(x_train, y_train, n_estimators=10)

    y_pred = bagging_predict(x_test, trees)

    error = mse(y_test, y_pred)

    print("\nBagging MSE:", error)


""" Main """
def main():
    data = loadData()

    eda(data)

    x, y = x_y(data)

    x_train, y_train, x_test, y_test = split_data(x, y)

    x_train, x_test = preprocessing(x_train, x_test)

    modelTraining(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()