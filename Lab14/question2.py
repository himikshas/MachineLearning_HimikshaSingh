#!/usr/bin/env python3

""" Implement Adaboost classifier without using scikit-learn. Use the Iris dataset.
 """
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Load and preprocess data (convert to binary classification)
def load_data():
    data = load_iris()
    X = data.data                 # Features
    y = data.target               # Labels (0,1,2)

    # Convert to binary: class 0 → -1, others → +1
    y = np.where(y == 0, -1, 1)

    # Split into train and test sets
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Train a decision stump (weak learner)
def train_stump(X, y, w):
    n_samples, n_features = X.shape
    min_error = float('inf')      # Initialize minimum error
    best_stump = {}               # Store best feature, threshold, polarity

    # Loop over all features
    for feature in range(n_features):
        values = X[:, feature]
        thresholds = np.unique(values)   # Possible split points

        # Try all thresholds and both polarity directions
        for threshold in thresholds:
            for polarity in [1, -1]:

                preds = np.ones(n_samples)   # Initialize predictions

                # Apply decision rule
                if polarity == 1:
                    preds[values < threshold] = -1
                else:
                    preds[values >= threshold] = -1

                # Compute weighted error
                error = np.sum(w[y != preds])

                # Store best stump with minimum error
                if error < min_error:
                    min_error = error
                    best_stump = {
                        "feature": feature,
                        "threshold": threshold,
                        "polarity": polarity
                    }

    return best_stump, min_error


# Predict using a trained stump
def stump_predict(X, stump):
    n_samples = X.shape[0]
    values = X[:, stump["feature"]]
    preds = np.ones(n_samples)

    # Apply learned rule
    if stump["polarity"] == 1:
        preds[values < stump["threshold"]] = -1
    else:
        preds[values >= stump["threshold"]] = -1

    return preds


# Train AdaBoost model
def adaboost_train(X, y, n_estimators=20):
    n_samples = X.shape[0]

    # Initialize all sample weights equally
    w = np.ones(n_samples) / n_samples

    models = []   # Store all weak learners
    alphas = []   # Store their weights

    # Iterate over number of weak learners
    for _ in range(n_estimators):

        # Train weak learner (decision stump)
        stump, error = train_stump(X, y, w)

        # Avoid division by zero
        error = max(error, 1e-10)

        # Compute model weight (alpha)
        alpha = 0.5 * np.log((1 - error) / error)

        # Get predictions from stump
        preds = stump_predict(X, stump)

        # Update sample weights:
        # Misclassified → weight increases
        # Correct → weight decreases
        w *= np.exp(-alpha * y * preds)

        # Normalize weights (sum = 1)
        w /= np.sum(w)

        # Store model and its importance
        models.append(stump)
        alphas.append(alpha)

    return models, alphas


# Predict using full AdaBoost model
def adaboost_predict(X, models, alphas):
    final_pred = np.zeros(X.shape[0])

    # Weighted sum of all weak learners
    for stump, alpha in zip(models, alphas):
        preds = stump_predict(X, stump)
        final_pred += alpha * preds

    # Final prediction based on sign
    return np.sign(final_pred)


# Compute accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Main function to run everything
def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train AdaBoost model
    models, alphas = adaboost_train(X_train, y_train, n_estimators=20)

    # Predict on test data
    y_pred = adaboost_predict(X_test, models, alphas)

    # Evaluate model
    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)


# Execute
main()