#!/usr/bin/env python3

""" Compute the derivative of a sigmoid function and visualize it  """

import numpy as np
import matplotlib.pyplot as plt

""" Sigmoid function """
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


""" Derivative of sigmoid """
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


""" Main function """
def main():
    # Create x values
    x = np.linspace(-10, 10, 100)

    # Calculate sigmoid and derivative
    y_sigmoid = sigmoid(x)
    y_derivative = sigmoid_derivative(x)

    # Plot both
    plt.plot(x, y_derivative, label="Derivative")
    plt.plot(x, y_sigmoid, label="Sigmoid")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title("Sigmoid Function")
    plt.show()

if __name__ == "__main__":
    main()