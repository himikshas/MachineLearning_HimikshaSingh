#!/usr/bin/env python3

# Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from ISLP import load_data


# ---------------------------------------------------
# Function to load dataset
# ---------------------------------------------------
def loadData():
    df = load_data("OJ")
    return df


# ---------------------------------------------------
# Function for preprocessing
# ---------------------------------------------------
def preprocess_data(df):

    df = df.copy()

    # Target
    y = df['Purchase']

    # Features
    X = df.drop('Purchase', axis=1)

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Encode target (CH=0, MM=1)
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y


# ---------------------------------------------------
# Train-test split
# ---------------------------------------------------
def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=800,
        random_state=42,
        stratify=y        #When split data into train and test, we want both sets to have same class distribution as the original dataset.
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------
# Scaling
# ---------------------------------------------------
def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# ---------------------------------------------------
# (b) Linear SVM (C = 0.01)
# ---------------------------------------------------
def train_linear_svm(X_train_scaled, X_test_scaled, y_train, y_test):

    model = SVC(kernel='linear', C=0.01)

    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    return train_acc, test_acc, train_f1, test_f1


# ---------------------------------------------------
# (c) Grid Search for best C
# ---------------------------------------------------
def grid_search_svm(X_train_scaled, y_train):

    param_grid = {'C': [0.01, 1, 10]}     #hyperparameters want to try

    model = SVC(kernel='linear')

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,                           #5-fold cross validation
        scoring='accuracy'              #choose best based on accuracy
    )

    grid.fit(X_train_scaled, y_train)           #runs all combinations of parameters and cross-validation folds

    print("\n--- Grid Search Results ---")
    print("Best C:", grid.best_params_['C'])
    print("Best CV Accuracy:", grid.best_score_)      #best cross validation accuracy obtained during grid search

    return grid.best_estimator_


# ---------------------------------------------------
# (d) Evaluate best model
# ---------------------------------------------------
def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):

    #evaluates the best model from GridSearchCV

    #Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    #Accuracy calculations
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    #error rate calculation
    train_error = 1 - train_acc
    test_error = 1 - test_acc

    #print values
    print("\n--- Best Model Performance ---")
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Train Error:", train_error)
    print("Test Error:", test_error)

    #return values
    return train_acc, test_acc


# ---------------------------------------------------
# (e) RBF Kernel SVM
# ---------------------------------------------------
def train_rbf_svm(X_train_scaled, X_test_scaled, y_train, y_test):

    model = SVC(kernel='rbf')

    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    return train_acc, test_acc, train_f1, test_f1


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():

    # Step 1: Load data
    data = loadData()

    # Step 2: Preprocess
    X, y = preprocess_data(data)

    # Step 3: Split (Part a)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Scale
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Step 5: Linear SVM (Part b)
    lin_train_acc, lin_test_acc, lin_train_f1, lin_test_f1 = train_linear_svm(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    print("\n--- Linear SVM (C = 0.01) ---")
    print("Train Accuracy:", lin_train_acc)
    print("Test Accuracy:", lin_test_acc)
    print("Train F1 score:", lin_train_f1)
    print("Test F1 score:", lin_test_f1)

    # Step 6: Grid Search (Part c)
    best_model = grid_search_svm(X_train_scaled, y_train)

    # Step 7: Evaluate Best Model (Part d)
    best_train_acc, best_test_acc = evaluate_model(
        best_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    # Step 8: RBF SVM (Part e)
    rbf_train_acc, rbf_test_acc, rbf_train_f1, rbf_test_f1 = train_rbf_svm(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    print("\n--- RBF Kernel SVM ---")
    print("Train Accuracy:", rbf_train_acc)
    print("Test Accuracy:", rbf_test_acc)
    print("Train F1 score:", rbf_train_f1)
    print("Test F1 score:", rbf_test_f1)

    # Final comparison
    print("\n--- Final Conclusion ---")
    if rbf_test_acc > best_test_acc:
        print("RBF kernel performs better.")
    else:
        print("Linear SVM with tuned C performs better.")


# Run
main()