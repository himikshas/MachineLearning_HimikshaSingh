#!/usr/bin/env python3

""" 3. In this problem, you will perform K-means clustering manually, with K = 2, on a small example with n = 6
observations and d = 2 features. The observations are as follows.

Write the python code for the following

(a) Plot the observations.
(b) Randomly assign a cluster label to each observation. You can use the np.random.choice() function to do this.
Report the cluster labels for each observation.
(c) Compute the centroid for each cluster.
(d) Assign each observation to the centroid to which it is closest, in terms of Euclidean distance. Report the cluster
labels for each observation.
(e) Repeat (c) and (d) until the answers obtained stop changing.
(f) In your plot from (a), color the observations according to the cluster labels obtained."""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    return np.array([
        [1, 4],
        [1, 3],
        [0, 4],
        [5, 1],
        [6, 2],
        [4, 0]
    ])

# -----------------------------
# Plot Data
# -----------------------------
def plot_data(X):
    plt.scatter(X[:, 0], X[:, 1], color='black')           #X[:,0] --> all X1 values
                                                           #X[:,1] --> all X2 values (plot points in 2D)
    for i, point in enumerate(X):                          #loop through each data point
        plt.text(point[0] + 0.1, point[1] + 0.1, str(i+1))   #write label near each point
    plt.title("Original Data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

# -----------------------------
# Initialize Clusters
# -----------------------------
def initialize_labels(n, K):
    np.random.seed(0)                     #fixes randomness --> same result every run
    return np.random.choice(K, n)                 #randomly assigns each point to cluster 0 or 1

# -----------------------------
# Compute Centroids
# -----------------------------
def compute_centroids(X, labels, K):
    centroids = []
    for k in range(K):                #loop over clusters (0 and 1)
        points = X[labels == k]             #select only those points that belongs to cluster k
        centroids.append(points.mean(axis=0))          #column wise mean (x1 and x2 separately)
    return np.array(centroids)                         #convert list --> NUmPy array

# -----------------------------
# Assign Clusters
# -----------------------------
def assign_clusters(X, centroids):
    new_labels = []
    for point in X:                  #loop through each observation
        distances = [np.linalg.norm(point - c) for c in centroids]           #compute euclidean distance from point to each centroid
        new_labels.append(np.argmin(distances))                              #assign cluster of minimum distance
    return np.array(new_labels)                                #return updated labels

# -----------------------------
# K-Means Algorithm
# -----------------------------
def kmeans(X, K):
    n = X.shape[0]                                 #number of datapoints
    labels = initialize_labels(n, K)               #random cluster assignment

    iteration = 0
    while True:
        print(f"\nIteration {iteration+1}")

        centroids = compute_centroids(X, labels, K)
        print("Centroids:\n", centroids)

        new_labels = assign_clusters(X, centroids)
        print("Labels:", new_labels)

        if np.array_equal(labels, new_labels):                    #check convergence( labels = new_labels)
            print("Converged!")
            break

        labels = new_labels
        iteration += 1

    return labels, centroids

# -----------------------------
# Plot Final Clusters
# -----------------------------
def plot_clusters(X, labels, centroids):
    colors = ['red', 'blue']

    for k in range(len(centroids)):                          #loop through cluster
        cluster_points = X[labels == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[k], label=f"Cluster {k}")

    plt.scatter(centroids[:, 0], centroids[:, 1],
                color='yellow', marker='X', s=200, label='Centroids')

    plt.title("Final Clusters")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    X = load_data()
    plot_data(X)

    K = 2
    labels, centroids = kmeans(X, K)

    plot_clusters(X, labels, centroids)

    print("\nFinal Labels:", labels)

# Run program
if __name__ == "__main__":
    main()