#!/usr/bin/env python3

""" Implement Kmeans clustering algorithm using scikit learn. Use appropriate dataset from ISLP. """


import pandas as pd
import matplotlib.pyplot as plt

from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


""" Load data """
def load_dataset():
    return load_data("OJ")


""" Preprocess data """
def preprocess(data):
    df = data.copy()

    # Drop target (unsupervised)
    df = df.drop('Purchase', axis=1)

    # Convert categorical → numeric
    df = pd.get_dummies(df, drop_first=True)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    return X


""" Apply KMeans """
def run_kmeans(X, k=3):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return model, labels


""" Plot clusters """
def plot_clusters(X, labels):
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("K-Means Clusters (PCA View)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


""" Main """
def main():
    data = load_dataset()
    X = preprocess(data)

    model, labels = run_kmeans(X, k=3)

    print("Cluster Centers:\n", model.cluster_centers_)

    # Show plot
    plot_clusters(X, labels)

if __name__ == "__main__":
    main()

