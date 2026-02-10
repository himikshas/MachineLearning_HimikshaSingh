#!/usr/bin/env python3

""" Implement sigmoid function in python and visualize it """

import numpy as np
import matplotlib.pyplot as plt

""" Sigmoid function """
def sigmoid(z):
    return 1 / (1 + np.exp(-z))          #formula to calculate sigmoid function


""" Main function """
def main():
    # Create x values
    x = np.linspace(-10, 10, 100)

    # Calculate sigmoid
    y = sigmoid(x)

    # Plot
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel('z')
    plt.ylabel('sigmoid(z)')
    plt.title('Sigmoid Function')
    plt.legend(['x', 'y'])
    plt.show()

if __name__ == "__main__":
    main()

