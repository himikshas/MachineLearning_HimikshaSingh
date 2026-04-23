#!/usr/bin/env python3

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from ISLP import load_data
from sklearn.linear_model import LogisticRegression

# Load Weekly dataset
weekly = load_data("Weekly")

# Convert Direction to binary (Up = 1, Down = 0)
weekly['Direction'] = (weekly['Direction'] == 'Up').astype(int)

# Select features and target
X = weekly[['Lag1', 'Lag2']].values
y = weekly['Direction'].values

n = len(y)

correct_predictions = []

# LOOCV loop
for i in range(n):

    # Create train and test sets
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i)

    X_test = X[i].reshape(1, -1)
    y_test = y[i]

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)[0]

    # Store result (1 if correct, 0 if wrong)
    correct_predictions.append(int(y_pred == y_test))


# Compute average accuracy
loocv_accuracy = np.mean(correct_predictions)

print("LOOCV Estimated Test Accuracy:", loocv_accuracy)