#!/usr/bin/env python3

""" Implement Gradient Boost Regression and Classification using scikit-learn.
Use the Boston housing dataset from the ISLP package for the regression problem and weekly dataset from the ISLP
package and use Direction as the target variable for the classification.
"""

"""  #################### GRADIENT BOOSTING REGRESSION ########################## """

""" Dataset: Boston Housing (from ISLP)
    PIPELINE :
    STEP 1 Load Weekly dataset
    STEP 2 Split into train/tes
    STEP 3 Train Gradient Boosting Regressor
    STEP 4 Evaluate using MSE & R²
"""

import pandas as pd
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


""" Step 1: Load Dataset """
def load_dataset():
    boston = load_data("Boston")
    df = pd.DataFrame(boston)

    X = df.drop("medv", axis=1)
    y = df["medv"]

    return X, y


""" Step 2: Train-Test Split """
def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


""" Step 3: Train Model """
def train_model(X_train, y_train):
    model = GradientBoostingRegressor(
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

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2



""" Main Function """
def main():
    # Load data
    X, y = load_dataset()

    # Split data
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    mse, r2 = evaluate_model(model, X_test, y_test)

    print("Regression Results:")
    print("MSE:", mse)
    print("R2 Score:", r2)


if __name__ == "__main__":
    main()

