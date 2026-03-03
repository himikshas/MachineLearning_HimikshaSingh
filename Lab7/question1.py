#!/usr/bin/env python3

""" Perform 10-fold cross validation for SONAR dataset in scikit-learn using logistic regression.
SONAR dataset is a binary classification problem with target variables as Metal or Rock.
 i.e. signals are from metal or rock. """

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Copy of sonar data.csv", header=None)

#divide the data into x and y
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode labels (M=1, R=0)
y = LabelEncoder().fit_transform(y)

# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)


# 10-Fold CV on Training Data
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X_train, y_train, cv=kfold)

print("Cross-validation accuracy (training):", cv_scores.mean())


# Train Final Model
model.fit(X_train, y_train)


# Evaluate on Test Set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("Test accuracy:", test_accuracy)








