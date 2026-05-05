#!/usr/bin/env python3

""" Work on NCI data - build classification model after reducing the gene expression features using hierarchical clustering.
    Compare this with the PCA approach
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


""" Load dataset """
def load_data():
    data = pd.read_csv("NCI60.csv")

    # Drop first column (sample names like V1, V14)
    data = data.iloc[:, 1:]

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Ensure numeric
    X = X.apply(pd.to_numeric)

    return X, y


""" Standardization """
def preprocess(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


""" HIERARCHICAL FEATURE REDUCTION """
def hierarchical_fit(X_train, n_clusters=50):
    X_T = X_train.T   #earlier it was (samples, genes) then becomes (genes, samples), we want to cluster genes not samples

    clustering = AgglomerativeClustering(n_clusters=n_clusters)                       #similar clusters, clustered together
    labels = clustering.fit_predict(X_T)

    return labels       #cluster ID


def hierarchical_transform(X, labels, n_clusters=50):
    X_reduced = []

    for i in range(n_clusters):                                     #loop over all clusters
        cluster_features = X[:, labels == i]
        cluster_mean = cluster_features.mean(axis=1)                          #mean of all features of one clusters, make a new feature
        X_reduced.append(cluster_mean)                                        #50 clusters --> 50 features

    return np.array(X_reduced).T                                              #transpose again


""" PCA """
def pca_fit(X_train, n_components=50):
    pca = PCA(n_components=n_components)
    X_train_p = pca.fit_transform(X_train)
    return pca, X_train_p


def pca_transform(pca, X_test):
    return pca.transform(X_test)


""" Model training """
def train_classifier(X_train, y_train):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    return model


""" Evaluate model """
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, zero_division=0))

    return acc


""" Main """
def main():
    #load data
    X, y = load_data()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardization
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)


    # Hierarchical Clustering
    labels = hierarchical_fit(X_train_scaled, 50)

    X_train_h = hierarchical_transform(X_train_scaled, labels, 50)
    X_test_h = hierarchical_transform(X_test_scaled, labels, 50)

    model_h = train_classifier(X_train_h, y_train)

    acc_h = evaluate(model_h, X_test_h, y_test)
    print("\nHierarchical Clustering Results:", acc_h)


    # PCA
    pca, X_train_p = pca_fit(X_train_scaled, 50)
    X_test_p = pca_transform(pca, X_test_scaled)

    model_p = train_classifier(X_train_p, y_train)

    acc_p = evaluate(model_p, X_test_p, y_test)
    print("\nPCA Results:", acc_p)


if __name__ == "__main__":
    main()
