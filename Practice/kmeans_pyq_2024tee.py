#!/usr/bin/env python3

""" A2. Use the NCI160 data for solving this problem. It consists of gene expression levels for 64 cancer
cell lines (not just 60) measured on 6830 genes. Each observation (cell line) is labeled with a cancer
type (e.g., breast, lung, colon, leukemia, etc.). Perform clustering using K-Means algorithm with k=4 """

""" here we are using y labels only for evaluation not during training phase """

# ---------------------------------------------------
# Import required libraries
# ---------------------------------------------------
from ISLP import load_data          # to load NCI60 dataset
import pandas as pd                # for data handling
import matplotlib.pyplot as plt    # for plotting

from sklearn.preprocessing import StandardScaler   # for scaling features
from sklearn.cluster import KMeans                # K-Means algorithm
from sklearn.decomposition import PCA             # for dimensionality reduction


# ---------------------------------------------------
# Function to load dataset
# ---------------------------------------------------
def loadData():
    """
    Loads the NCI60 dataset
    Returns:
        X : gene expression data (features)
        y : cancer type labels
    """
    nci = load_data("NCI60")

    X = nci['data']
    y = nci['labels']

    #Convert 2D to 1D array
    y = y.values.ravel()                  #converts (64,1) to (64,) as we want y to be a single column

    return X, y


# ---------------------------------------------------
# Function to preprocess data
# ---------------------------------------------------
def preprocessData(X):
    """
    Standardizes the data so that each feature has
    mean = 0 and variance = 1

    This is important because K-Means is distance-based
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# ---------------------------------------------------
# Function to train K-Means model
# ---------------------------------------------------
def trainModel(X_scaled, k=4):
    """
    Applies K-Means clustering

    Parameters:
        X_scaled : standardized data
        k        : number of clusters (default = 4)

    Returns:
        model    : trained K-Means model
        clusters : cluster labels for each sample
    """
    model = KMeans(n_clusters=k, random_state=42)          #so randomness stay same

    # fit model and assign cluster labels
    clusters = model.fit_predict(X_scaled)

    return model, clusters


# ---------------------------------------------------
# Function to evaluate clustering
# ---------------------------------------------------
def evaluateModel(clusters, y):
    """
    Compares predicted clusters with actual labels
    using a contingency table
    """
    print("\n--- Cluster vs Actual Labels ---\n")

    # Crosstab shows how clusters align with real cancer types
    print(pd.crosstab(clusters, y))            #here rows(clusters) represent predicted clusters and columns(y) represent actual labels


# ---------------------------------------------------
# Function to visualize clusters using PCA
# ---------------------------------------------------
def plotClusters(X_scaled, clusters):
    """
    Reduces high-dimensional data to 2D using PCA
    and plots clusters as we cant visualize high dimensional data
    """
    pca = PCA(n_components=2)          #reduce data from 6830 to 2 features so we can plot on 2D graph

    # Transform data into 2 principal components
    X_pca = pca.fit_transform(X_scaled)       #fit finds max variance and transform converts data into new 2D space

    # Plot clusters
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)        #PC1 x-axis and PC2 y-axis

    plt.title("K-Means Clustering (k = 4)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.show()


# ---------------------------------------------------
# Main function (execution starts here)
# ---------------------------------------------------
def main():
    # Step 1: Load dataset
    X, y = loadData()

    # Step 2: Preprocess (scaling)
    X_scaled = preprocessData(X)

    # Step 3: Train model
    model, clusters = trainModel(X_scaled, k=4)

    # Step 4: Evaluate results
    evaluateModel(clusters, y)

    # Step 5: Visualize clusters
    plotClusters(X_scaled, clusters)


# ---------------------------------------------------
# Run the program
# ---------------------------------------------------
if __name__ == "__main__":
    main()