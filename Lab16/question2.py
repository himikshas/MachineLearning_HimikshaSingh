#!/usr/bin/env python3

""" Implement XGBoost classifier and regressor using scikit-learn """


""" #################### XGBOOST (CLASSIFIER + REGRESSOR) ########################## """

"""
PIPELINE:
STEP 1: Import Libraries
STEP 2: Classification using XGBoost
STEP 3: Regression using XGBoost
"""

# STEP 1: Import Libraries

# Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Regression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# XGBoost
from xgboost import XGBClassifier, XGBRegressor


# STEP 2: XGBOOST CLASSIFIER


print("\n===== XGBOOST CLASSIFIER =====")

# Load Dataset
data_clf = load_iris()
X_clf = data_clf.data
y_clf = data_clf.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Train Model
clf_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric='mlogloss'
)

clf_model.fit(X_train, y_train)

# Prediction
y_pred_clf = clf_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred_clf))
print("F1 Score:", f1_score(y_test, y_pred_clf, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_clf))


# STEP 3: XGBOOST REGRESSOR

print("\n===== XGBOOST REGRESSOR =====")

# Load Dataset
data_reg = fetch_california_housing()
X_reg = data_reg.data
y_reg = data_reg.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train Model
reg_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

reg_model.fit(X_train, y_train)

# Prediction
y_pred_reg = reg_model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_reg))
print("R² Score:", r2_score(y_test, y_pred_reg))