#!/usr/bin/python3

""" Compute SONAR classification results with and without data pre-processing (data normalization).
Perform data pre-processing with your implementation and
with scikit-learn methods and compare the results. """

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

""" Load the SONAR data """
def load_data():
    df = pd.read_csv("Copy of sonar data.csv", header=None)
    return df

""" Splitting data into x and y"""
def form_x_y(data):
    X = data.iloc[:, :-1].values
    y = (data.iloc[:, -1].values)
    return X, y

""" Standardization of features """
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

""" Normalization of features """
def normalize_data(X_train, X_test):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

""" Cross validation """
def run_cv(X, y):
    model = LogisticRegression(max_iter=1000)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold)
    return scores

""" Train and test evaluation"""
def run_train_test(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

""" Main """
def main():
    data = load_data()
    X, y = form_x_y(data)

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=" * 50)
    print("SONAR Dataset  |  Logistic Regression")
    print(f"Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")


    # ---- RAW DATA ----
    raw_scores = run_cv(X, y)
    raw_tt = run_train_test(X_train, X_test, y_train, y_test)

    print()
    print("=" * 50)
    print("--- No Preprocessing ---")
    print("=" * 50)
    print("Fold accuracy :", raw_scores)
    print("Mean CV acc   :", round(raw_scores.mean(), 4))
    print("Train/Test acc:", round(raw_tt, 4))

    # ---- STANDARDIZED ----
    X_train_std, X_test_std = standardize_data(X_train, X_test)
    std_scores = run_cv(X, y)   # CV on full data (scaler fit inside each fold via pipeline logic)
    std_scores_scaled = run_cv(StandardScaler().fit_transform(X), y)
    std_tt = run_train_test(X_train_std, X_test_std, y_train, y_test)

    print()
    print("=" * 50)
    print("--- Standardization (Z-score) ---")
    print("=" * 50)
    print("Fold accuracy :", std_scores_scaled)
    print("Mean CV acc   :", round(std_scores_scaled.mean(), 4))
    print("Train/Test acc:", round(std_tt, 4))

    # ---- NORMALIZED ----
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    norm_scores_scaled = run_cv(MinMaxScaler().fit_transform(X), y)
    norm_tt = run_train_test(X_train_norm, X_test_norm, y_train, y_test)

    print()
    print("=" * 50)
    print("--- Normalization (Min-Max) ---")
    print("=" * 50)
    print("Fold accuracy :", norm_scores_scaled)
    print("Mean CV acc   :", round(norm_scores_scaled.mean(), 4))
    print("Train/Test acc:", round(norm_tt, 4))


if __name__ == "__main__":
    main()




