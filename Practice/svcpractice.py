#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from ISLP import load_data


def loadData():

    data = load_data("OJ")

    return data
def eda(data):

    data.shape
    print("\nTarget distribution\n", data['Purchase'].value_counts())

    plt.figure()
    sns.countplot(x="Purchase", data=data)
    plt.title("Target Distribution")
    plt.show()

def preprocessing(data):

    data = data.copy()

    y = data['Purchase']

    X = data.drop('Purchase', axis=1)

    X = pd.get_dummies(X, drop_first=True)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y

def splitData(X, y):

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=800,
        random_state=42,
        stratify=y
    )

    return x_train, x_test, y_train, y_test


def scale_data(x_train, x_test):

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)

    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled

def trainLinearSVM(x_train_scaled, x_test, y_train, y_test):

    model = SVC(kernel='linear')

    model.fit(x_train_scaled, y_train)

    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test)

    trainAccuracy = accuracy_score(y_train, y_train_pred)
    testAccuracy = accuracy_score(y_test, y_test_pred)

    return trainAccuracy, testAccuracy

def best_parameters(x_train_scaled, y_train):

    param_grid = { 'C' :[0.01, 1, 10]}

    model = SVC(kernel='linear')

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid.fit(x_train_scaled, y_train)

    print("\nGrid Search Results:\n")
    print("Best C :", grid.best_params_['C'])
    print("Best Accuracy :", grid.best_score_)

    return grid.best_estimator_

def evaluate_model(model, x_train_scaled, x_test_scaled, y_train, y_test):

    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    trainAccuracy = accuracy_score(y_train, y_train_pred)



def trainRBFSVM(x_train_scaled, x_test_scaled, y_train, y_test):

    model = SVC(kernel='rbf')

    model.fit(x_train_scaled, y_train)

    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    trainAccuracy = accuracy_score(y_train, y_train_pred)
    testAccuracy = accuracy_score(y_test, y_test_pred)

    return trainAccuracy, testAccuracy

def main():

    data = loadData()

    eda(data)

    x, y = preprocessing(data)

    x_train, x_test, y_train, y_test = splitData(x, y)

    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    lin_train_acc, lin_test_acc = trainLinearSVM(x_train_scaled, x_test_scaled, y_train,
                                                 y_test)

    print("\n--- Linear SVM (C = 0.01) ---")
    print("Train Accuracy:", lin_train_acc)
    print("Test Accuracy:", lin_test_acc)


    rbf_train_acc, rbf_test_acc = trainRBFSVM(x_train_scaled, x_test_scaled, y_train,
                                              y_test)

    print("\n--- RBF SVM (C = 0.01) ---")
    print("Train Accuracy:", rbf_train_acc)
    print("Test Accuracy:", rbf_test_acc)

    if rbf_test_acc > lin_test_acc:
        print("\n RBF kernel performs better than linear kernel.")
    else:
        print("\n Linear kernel performs better than RBF kernel.")

if __name__ == "__main__":
    main()
