#!/usr/bin/python3

import numpy as np

# 1. Define the Data Matrix X
X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])

# PART 1: Manual Calculation (Matrix Multiplication)

n_samples = X.shape[0] # This is 5

# Step 1: Compute the mean of each column
# We sum down the rows (axis=0) and divide by N
mean_vector = np.sum(X, axis=0) / n_samples

# Step 2: Center the data (subtract mean from each element)
# This creates the "centered" matrix (often called X_c or B)
X_centered = X - mean_vector

# Step 3: Compute Covariance using Matrix Multiplication
# Formula: (X_centered.T dot X_centered) / (N - 1)
# Note: We use (N-1) for sample covariance (unbiased estimate)
cov_matrix_manual = np.dot(X_centered.T, X_centered) / (n_samples - 1)

print(" Manual Calculation Results:")
print("Mean Vector:\n", mean_vector)
print("\nCentered Matrix (first few rows):\n", X_centered[:2])
print("\nCovariance Matrix:\n", cov_matrix_manual)


# PART 2: Verification with NumPy

# rowvar=False tells numpy that columns are variables, not rows
cov_matrix_numpy = np.cov(X, rowvar=False)

print("\nNumPy Verification: ")
print("NumPy Covariance Matrix:\n", cov_matrix_numpy)

# Check if they match
print("\nDo they match?", np.allclose(cov_matrix_manual, cov_matrix_numpy))