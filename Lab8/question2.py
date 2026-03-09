#!/usr/bin/env python3

""" Build a classification model for wisconsin dataset
 using Ridge and Lasso classifier using scikit-learn
"""


""" Import Libraries """
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score



""" Function to load dataset """
def load_dataset():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y



""" Function for preprocessing """
def preprocess_data(X, y):

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test



""" Ridge Classifier function """
def ridge_classifier(X_train, X_test, y_train, y_test):

    model = RidgeClassifier(alpha=0.1)    #alpha is lambda here

    # Train model
    model.fit(X_train, y_train)    #model learns coefficients(weights)
                                  #y = theta1*x1 + theta2*x2 + ... + theta n * x n + l2 penalty

    # Predict
    y_pred = model.predict(X_test)      #testing the model

    # Evaluation
    print("\nRidge Classifier Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))



""" Lasso Classifier function """
def lasso_classifier(X_train, X_test, y_train, y_test):

    model = LogisticRegression(
        penalty='l1',      #some features which are not required their theta becomes close to 0
        solver='liblinear',        #algorithm used for optimization
        max_iter=1000
    )

    # Training model
    model.fit(X_train, y_train)

    # Predict (testing model)
    y_pred = model.predict(X_test)

    # Evaluation
    print("\nLasso (L1) Logistic Regression Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))



""" Main function """
def main():

    # Load dataset
    X, y = load_dataset()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train Ridge classifier
    ridge_classifier(X_train, X_test, y_train, y_test)

    # Train Lasso classifier
    lasso_classifier(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()


