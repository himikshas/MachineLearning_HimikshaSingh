#!/usr/bin/env python3

""" Let x=[x],X2,X3], and y=[y1,y2,y3]. Implement a feature mapping function, (x), called
Transform(x), given by (x) = (X1X1, X1X2, X1X3, X2X1, X2X2, X2X3, X3X1, X3X2, X3X3).

(a) Let x = [1, 2, 3], y = [4, 5, 6]. Use the above "Transform" function to transform these vectors to
a higher dimension and compute the dot product in a higher dimension. Print the value.

(b) Implement a kernel, K(x,y) = (<x,y >)2. <x,y> is a dot product of x and y. Apply this kernel
function and evaluate the output for the same x and y vectors. Show that the result is the same in
both scenarios demonstrating the power of a kernel trick."""

# !/usr/bin/env python3

import numpy as np


# -------------------------------
# Step 1: Feature Mapping Function
# -------------------------------
def transform(x):
    """
    Applies phi(x) mapping:
    (x1x1, x1x2, x1x3, x2x1, x2x2, x2x3, x3x1, x3x2, x3x3)
    """
    phi = []
    for i in range(len(x)):
        for j in range(len(x)):
            phi.append(x[i] * x[j])
    return np.array(phi)


# -------------------------------
# Step 2: Dot Product Function
# -------------------------------
def dot_product(a, b):
    return np.dot(a, b)


# -------------------------------
# Step 3: Kernel Function
# -------------------------------
def polynomial_kernel(x, y):
    """
    K(x,y) = (x.y)^2
    """
    return (np.dot(x, y)) ** 2


# -------------------------------
# Step 4: Pipeline Function
# -------------------------------
def main():
    # Given vectors
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    # Transform to higher dimension
    phi_x = transform(x)
    phi_y = transform(y)

    # Dot product in higher dimension
    high_dim_result = dot_product(phi_x, phi_y)

    # Kernel computation
    kernel_result = polynomial_kernel(x, y)

    # Print results
    print("Phi(x):", phi_x)
    print("Phi(y):", phi_y)
    print("\nDot Product in Higher Dimension:", high_dim_result)
    print("Kernel Result:", kernel_result)

    # Verification
    if high_dim_result == kernel_result:
        print("\nBoth results are SAME → Kernel Trick Verified!")
    else:
        print("\nResults differ!")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    main()