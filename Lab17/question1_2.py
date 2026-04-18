#!/usr/bin/env python3

""" Let x1 = [3, 6], x2 = [10, 10].  Use the above “Transform” function to transform these vectors to
a higher dimension and  compute the dot product in a higher dimension. Print the value.
"""
"""  Implement a polynomial kernel K(a,b) =  a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2 .
 Apply this kernel function and evaluate the output for the same x1 and x2 values. Notice that the result is the same in both scenarios 
 demonstrating the power of kernel trick.
"""
import numpy as np
import matplotlib.pyplot as plt

def data():
    blue = np.array([
        [1, 13], [1, 18], [2, 9], [3, 6], [6, 3], [9, 2], [13, 1], [18, 1]
    ])

    red = np.array([
        [3, 15], [6, 6], [6, 11], [9, 5], [10, 10], [11, 5], [12, 6], [16, 3]
    ])
    return blue, red

def transform(X):
    x1 = X[:, 0]
    x2 = X[:, 1]

    z1 = x1**2
    z2 = np.sqrt(2) * x1 * x2
    z3 = x2**2

    return np.vstack((z1, z2, z3)).T

def transform_data(blue, red):
    blue_3d = transform(blue)
    red_3d = transform(red)

    return blue_3d, red_3d


def plot_3d(blue_3d, red_3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(blue_3d[:,0], blue_3d[:,1], blue_3d[:,2], label='Blue')
    ax.scatter(red_3d[:,0], red_3d[:,1], red_3d[:,2], label='Red')

    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('z3')

    ax.legend()
    plt.show()


def dot_product(x1, x2):
    phi_x1 = transform(x1)
    phi_x2 = transform(x2)

    dot_product = np.dot(phi_x1, phi_x2.T)
    
    return dot_product[0][0]

def polynomial_kernel(a, b):
    return (a[0]**2 * b[0]**2) + (2 * a[0] * b[0] * a[1] * b[1]) + (a[1]**2 * b[1]**2)


def main():
    # Load data
    blue, red = data()

    # Transform data
    blue_3d, red_3d = transform_data(blue, red)

    # Plot
    plot_3d(blue_3d, red_3d)

    # Compute dot product
    x1 = np.array([[3, 6]])
    x2 = np.array([[10, 10]])

    result = dot_product(x1, x2)
    print("Dot Product in higher dimension:", result)

    # Kernel computation
    a = np.array([3, 6])
    b = np.array([10, 10])

    kernel_result = polynomial_kernel(a, b)
    print("Kernel Value:", kernel_result)
    print("Both the values are same ")

if __name__ == "__main__":
    main()


