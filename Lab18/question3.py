"""
Twitter Sentiment Analysis using SVM
Dataset: Tweets.csv (Kaggle sentiment140 format)

- Uses TF-IDF
- Trains SVM with multiple kernels
- Compares performance
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

""" ================= LOAD DATA ================= """


def load_data(path):
    df = pd.read_csv(path)

    print("Columns in dataset:", df.columns)

    # Features
    X = df['text']

    # Labels (negative / neutral / positive)
    y = df['airline_sentiment']

    return X, y


""" ================= VECTORIZE ================= """


def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    """ ===== STANDARDIZATION ===== """
    scaler = StandardScaler(with_mean=False)

    X_train_scaled = scaler.fit_transform(X_train_vec)
    X_test_scaled = scaler.transform(X_test_vec)

    return X_train_scaled, X_test_scaled


""" ================= TRAIN ================= """


def train_svm(X_train, y_train, kernel_type):
    model = SVC(kernel=kernel_type)
    model.fit(X_train, y_train)

    return model


""" ================= EVALUATE ================= """


def evaluate(model, X_test, y_test, kernel_name):
    y_pred = model.predict(X_test)

    print("\n===== Kernel:", kernel_name, "=====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


""" ================= MAIN ================= """


def main():
    # Load dataset (MAKE SURE file is in same folder)
    X, y = load_data("Tweets.csv")

    print("Dataset loaded successfully!")
    print("Total samples:", len(X))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vectorization
    X_train_vec, X_test_vec = vectorize_text(X_train, X_test)

    # Try different kernels
    kernels = ['linear', 'rbf', 'poly']

    for k in kernels:
        model = train_svm(X_train_vec, y_train, k)
        evaluate(model, X_test_vec, y_test, k)


if __name__ == "__main__":
    main()