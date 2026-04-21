#!/usr/bin/env python3

# Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ISLP import load_data

# ---------------------------------------------------
# Function to load dataset
# ---------------------------------------------------
def loadData():
    """
    Loads the OJ dataset from ISLP module
    """

    df = load_data("OJ")  # directly load dataset

    return df


# ---------------------------------------------------
# Function for preprocessing
# ---------------------------------------------------
def preprocess_data(df):
    """
    - Separates features and target
    - Encodes categorical variables
    - Converts target (CH/MM) → numeric (0/1)
    """
    df = df.copy()          #to avoid modifying original dataset

    # Target variable
    y = df['Purchase']

    # Feature matrix
    X = df.drop('Purchase', axis=1)

    # Convert categorical features into dummy variables(dummy is one hot encoding)
    X = pd.get_dummies(X, drop_first=True)            #converts categorical variables into numerical

    # Encode target variable (CH=0, MM=1)
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y


# ---------------------------------------------------
# Function to split data
# ---------------------------------------------------
def split_data(X, y):
    """
    Splits data into:
    - 800 training samples
    - Remaining test samples
    Uses stratification to maintain class balance
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=800,
        random_state=42,
        stratify=y              #so that train and test data should remain in same class proportions
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------
# Function to scale data
# ---------------------------------------------------
def scale_data(X_train, X_test):
    """
    Standardizes features:
    - Mean = 0
    - Std = 1
    Important for SVM (distance-based model)
    """
    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply same transformation on test data
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# ---------------------------------------------------
# (b) Linear SVM Model
# ---------------------------------------------------
def train_linear_svm(X_train, X_test, y_train, y_test):
    """
    Trains a linear SVM with C = 0.01
    Calculates train and test accuracy
    """

    # Create model with linear kernel
    model = SVC(kernel='linear', C=0.01)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy calculation
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    return train_acc, test_acc


# ---------------------------------------------------
# (c) RBF Kernel SVM Model
# ---------------------------------------------------
def train_rbf_svm(X_train, X_test, y_train, y_test):
    """
    Trains SVM with RBF kernel
    Uses default gamma
    """

    # Create RBF model
    model = SVC(kernel='rbf')

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy calculation
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    return train_acc, test_acc


# ---------------------------------------------------
# Main function (runs everything)
# ---------------------------------------------------
def main():
    # Step 1: Load dataset
    df = loadData()

    # Step 2: Preprocess data
    X, y = preprocess_data(df)

    # Step 3: Split data (Part a)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Step 5: Train Linear SVM (Part b)
    lin_train_acc, lin_test_acc = train_linear_svm(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    print("\n--- Linear SVM (C = 0.01) ---")
    print("Train Accuracy:", lin_train_acc)
    print("Test Accuracy:", lin_test_acc)

    # Step 6: Train RBF SVM (Part c)
    rbf_train_acc, rbf_test_acc = train_rbf_svm(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    print("\n--- RBF Kernel SVM ---")
    print("Train Accuracy:", rbf_train_acc)
    print("Test Accuracy:", rbf_test_acc)

    # Step 7: Compare models
    if rbf_test_acc > lin_test_acc:
        print("\nConclusion: RBF kernel performs better.")
    else:
        print("\nConclusion: Linear kernel performs better.")


# Run program
main()