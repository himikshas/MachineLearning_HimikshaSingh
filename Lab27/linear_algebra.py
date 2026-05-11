#!usr/bin/env python3

"""
1. Check whether matrix is positive definite
2. Find eigenvalues of Hessian
3. Determine concavity
4. Find gradient and critical points
"""


import sympy as sp
import numpy as np


""" Check whether matrix is positive definite """
def check_positive_definite():

    A = np.array([
        [9, -15],
        [-15, 21]
    ])

    # Find eigenvalues
    eigenvalues = np.linalg.eigvals(A)

    print("\nMatrix A :")
    print(A)

    print("\nEigenvalues :")
    print(eigenvalues)

    # Check positive definiteness
    if np.all(eigenvalues > 0):

        print("\nMatrix is Positive Definite")

    else:

        print("\nMatrix is NOT Positive Definite")


""" Eigen values of Hessian """
def hessian_eigenvalues():

    x = 3
    y = 1

    Hessian = np.array([
        [12*(x**2), -1],
        [-1, 2]
    ])

    eigenvalues = np.linalg.eigvals(Hessian)

    print("\nHessian Matrix :")
    print(Hessian)

    print("\nEigenvalues of Hessian :")
    print(eigenvalues)


""" Determine concavity """
def determine_concavity():

    points = [(0,0), (3,3), (3,-3)]

    for point in points:

        x, y = point

        # Hessian matrix
        Hessian = np.array([
            [6*x, -1],
            [-1, 12*y]
        ])

        eigenvalues = np.linalg.eigvals(Hessian)

        print("\nPoint :", point)

        print("\nHessian Matrix :")
        print(Hessian)

        print("\nEigenvalues :")
        print(eigenvalues)

        # Determine concavity
        if np.all(eigenvalues > 0):

            print("\nFunction is Convex")

        elif np.all(eigenvalues < 0):

            print("\nFunction is Concave")

        else:

            print("\nNeither Concave nor Convex")


""" Critical Points """
def critical_points():

    x, y = sp.symbols('x y')                        #create symbolic variables

    # Function
    f = 4*x + 2*y - x**2 - 3*y**2

    # Gradient
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)

    print("\nGradient :")
    print("df/dx =", fx)
    print("df/dy =", fy)

    # Solve gradient = 0
    critical = sp.solve([fx, fy], (x, y))

    print("\nCritical Point :")
    print(critical)

    # Hessian matrix
    H = sp.hessian(f, (x, y))

    print("\nHessian Matrix :")
    print(H)

    # Convert Hessian into numpy array
    H_np = np.array(H).astype(np.float64)

    eigenvalues = np.linalg.eigvals(H_np)

    print("\nEigenvalues :")
    print(eigenvalues)

    # Determine nature
    if np.all(eigenvalues > 0):

        print("\nCritical Point is Minimum")

    elif np.all(eigenvalues < 0):

        print("\nCritical Point is Maximum")

    else:

        print("\nCritical Point is Neither")


""" Main """
def main():

    # Question 1
    check_positive_definite()

    # Question 2
    hessian_eigenvalues()

    # Question 3
    determine_concavity()

    # Question 4
    critical_points()


if __name__ == "__main__":
    main()