#!/usr/bin/env python3

""" Implement K-Means algorithm ground-up using Python """

"""
K-Means Clustering from Scratch (Function-based)

This script implements K-Means clustering using only NumPy.
It includes:
- Initialization of centroids
- Assignment of data points to clusters
- Updating centroids
- Convergence check
- Example with visualization
"""

import numpy as np
import matplotlib.pyplot as plt


def initialize_centroids(X, k):
    """
    Randomly select k data points from X as initial centroids.

    Parameters:
    X : ndarray of shape (n_samples, n_features)
    k : number of clusters

    Returns:
    centroids : ndarray of shape (k, n_features)
    """
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def compute_distances(X, centroids):
    """
    Compute Euclidean distance between each data point and each centroid.

    Parameters:
    X : ndarray (n_samples, n_features)
    centroids : ndarray (k, n_features)

    Returns:
    distances : ndarray (n_samples, k)
    """
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances


def assign_clusters(X, centroids):
    """
    Assign each data point to the nearest centroid.

    Returns:
    labels : ndarray (n_samples,)
    """
    distances = compute_distances(X, centroids)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k):
    """
    Compute new centroids as mean of assigned points.

    Handles empty clusters by reinitializing randomly.

    Returns:
    new_centroids : ndarray (k, n_features)
    """
    new_centroids = []

    for i in range(k):
        points = X[labels == i]

        if len(points) == 0:
            """
            If no points assigned to cluster,
            reinitialize centroid randomly.
            """
            new_centroids.append(X[np.random.randint(0, X.shape[0])])
        else:
            new_centroids.append(points.mean(axis=0))

    return np.array(new_centroids)


def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    Perform K-Means clustering.

    Parameters:
    X : data points
    k : number of clusters
    max_iters : maximum iterations
    tol : tolerance for convergence

    Returns:
    centroids : final cluster centers
    labels : cluster assignments
    """
    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):
        """
        Step 1: Assign clusters
        Step 2: Update centroids
        Step 3: Check convergence
        """
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        if np.linalg.norm(centroids - new_centroids) < tol:
            """
            Stop if centroids do not change significantly
            """
            break

        centroids = new_centroids

    return centroids, labels


if __name__ == "__main__":
    """
    Example usage of K-Means
    """

    np.random.seed(0)

    """
    Generate synthetic dataset with 3 clusters
    """
    X = np.vstack((
        np.random.randn(100, 2) + [2, 2],
        np.random.randn(100, 2) + [-2, -2],
        np.random.randn(100, 2) + [2, -2]
    ))

    k = 3
    centroids, labels = kmeans(X, k)

    print("Final Centroids:\n", centroids)

    """
    Plot the clustered data
    """
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1],
                color='red', marker='X', s=200, label='Centroids')

    plt.title("K-Means Clustering (From Scratch)")
    plt.legend()
    plt.show()