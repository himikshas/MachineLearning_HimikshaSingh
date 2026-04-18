"""
Logistic Regression on Heart Dataset
Includes:
- Threshold variation
- Confusion matrices
- Metrics calculation
- ROC curve and AUC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc


def load_data(file_path):
    """
    Load dataset and preprocess it.
    Handles missing values.
    """
    df = pd.read_csv(file_path)

    # Drop unwanted column
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Convert categorical to numeric
    df = pd.get_dummies(df, drop_first=True)

    # Convert target
    if 'AHD' in df.columns:
        df['AHD'] = df['AHD'].map({'Yes': 1, 'No': 0})
        y = df['AHD']
        X = df.drop(columns=['AHD'])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    # 🔥 HANDLE MISSING VALUES
    X = X.fillna(X.mean())

    return X, y


def split_data(X, y):
    """
    Split dataset into training and testing sets.

    Parameters:
    X : Features
    y : Target

    Returns:
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    """
    Apply standardization to features.

    StandardScaler:
    - Mean = 0
    - Std deviation = 1

    Returns:
    Scaled X_train and X_test
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def train_model(X_train, y_train):
    """
    Train logistic regression model.

    Parameters:
    X_train : Training features
    y_train : Training labels

    Returns:
    Trained model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


def get_probabilities(model, X_test):
    """
    Predict probabilities for test data.

    Returns:
    Probability of class = 1
    """
    return model.predict_proba(X_test)[:, 1]


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics manually.

    Metrics:
    - Accuracy
    - Precision
    - Sensitivity (Recall)
    - Specificity
    - F1-score

    Returns:
    Dictionary of metrics
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1
    }


def evaluate_thresholds(y_true, y_prob, thresholds):
    """
    Evaluate model performance at different thresholds.

    For each threshold:
    - Convert probabilities to class labels
    - Compute confusion matrix
    - Print all metrics
    """
    for t in thresholds:
        print(f"\n===== Threshold: {t} =====")

        y_pred = (y_prob >= t).astype(int)
        metrics = compute_metrics(y_true, y_pred)

        print("Confusion Matrix:")
        print(f"[[{metrics['TN']} {metrics['FP']}]\n [{metrics['FN']} {metrics['TP']}]]")

        print(f"Accuracy: {metrics['Accuracy']:.3f}")
        print(f"Precision: {metrics['Precision']:.3f}")
        print(f"Sensitivity: {metrics['Sensitivity']:.3f}")
        print(f"Specificity: {metrics['Specificity']:.3f}")
        print(f"F1-score: {metrics['F1']:.3f}")


def plot_roc(y_true, y_prob):
    """
    Plot ROC curve and compute AUC.

    ROC Curve:
    - X-axis: False Positive Rate
    - Y-axis: True Positive Rate

    AUC:
    - Measures overall model performance
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    plt.show()

    print(f"AUC Score: {roc_auc:.3f}")


def main():
    """
    Main function to execute pipeline.

    Steps:
    - Load data
    - Split data
    - Scale features
    - Train model
    - Predict probabilities
    - Evaluate thresholds
    - Plot ROC curve
    """
    X, y = load_data("Heart.csv")

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test = scale_data(X_train, X_test)

    model = train_model(X_train, y_train)

    y_prob = get_probabilities(model, X_test)

    thresholds = [0.3, 0.5, 0.7]
    evaluate_thresholds(y_test.values, y_prob, thresholds)

    plot_roc(y_test, y_prob)


if __name__ == "__main__":
    main()