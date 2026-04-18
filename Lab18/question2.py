#!/usr/bin/env python3

""" Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features. Leave out 10% of each class and
    test prediction performance on these observations. """

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

""" Load the data """
def load_iris_binary():
    iris = datasets.load_iris()
    X = iris.data[:, :2]   # first two features
    y = iris.target

    # keep only class 1 and 2
    mask = y != 0
    X = X[mask]
    y = y[mask]

    return X, y

""" Train - Test split """
def split_data(X, y, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


""" Train SVM Model """
def train_svm(X_train, y_train, kernel='linear', C=1.0):
    model = svm.SVC(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    return model


""" Evaluate the model """
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    return acc

""" Plot Decision Boundary """
def plot_decision_boundary(model, X, y, title="SVM Decision Boundary"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title(title)
    plt.show()


""" Plot Support Vector """
def plot_with_support_vectors(model, X, y):
    plt.figure()

    # plot data
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

    # highlight support vectors
    plt.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=150,
        facecolors='none',
        edgecolors='k',
        linewidths=1.5
    )

    plt.title("Support Vectors Highlighted")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.show()

""" Main """
def run_svm_pipeline(kernel='rbf'):
    # Load
    X, y = load_iris_binary()

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    model = train_svm(X_train, y_train, kernel=kernel)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Plot boundary
    plot_decision_boundary(model, X_train, y_train,
                           title=f"SVM ({kernel}) Decision Boundary")

    # Plot support vectors
    plot_with_support_vectors(model, X_train, y_train)
    
    
if __name__ == "__main__":
    run_svm_pipeline(kernel='rbf')