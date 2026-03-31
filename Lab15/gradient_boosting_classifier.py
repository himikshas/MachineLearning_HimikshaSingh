#!/usr/bin/env python3


"""  #################### GRADIENT BOOSTING REGRESSION ########################## """

""" 
    Dataset: Weekly Dataset (from ISLP)
    PIPELINE :
    STEP 1 Load Weekly dataset
    STEP 2 Split into train/tes
    STEP 3 Train Gradient Boosting Regressor
    STEP 4 Evaluate using MSE & R²
"""


import pandas as pd
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


""" Step 1: Load Dataset """
def load_dataset():
    weekly = load_data("Weekly")
    df = pd.DataFrame(weekly)

    # Convert Direction to numeric
    df["Direction"] = df["Direction"].map({"Up": 1, "Down": 0})

    X = df.drop("Direction", axis=1)
    y = df["Direction"]

    return X, y


""" Step 2: Train-Test Split """
def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


""" Step 3: Train Model """
def train_model(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


""" Step 4: Evaluate Model """
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, cm, f1


""" Main Function """
def main():
    # Load data
    X, y = load_dataset()

    # Split data
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy, cm, f1 = evaluate_model(model, X_test, y_test)

    print("Classification Results:")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("F1 Score:\n", f1)


if __name__ == "__main__":
    main()