#!/usr/bin/python3

""" Implement Gaussian PDF - mean = 0, sigma = 15 in the range[start=-100, stop=100, num=100] """

import math
import matplotlib.pyplot as plt


def get_gaussian_data():
    # Parameters
    mean = 0  # Mean
    sigma = 15  # Standard deviation

    # Range settings
    start = -100
    stop = 100
    num = 100

    # Calculate step size manually
    # We use (num - 1) because we want to include the last point
    step = (stop - start) / (num - 1)

    x_data = []
    y_data = []

    # Pre-calculate the constant part of the equation to speed it up
    # Formula: (1 / (sigma * sqrt(2*pi))) * e^(-0.5 * ((x-mu)/sigma)^2)
    constant_term = 1 / (sigma * math.sqrt(2 * math.pi))

    for i in range(num):
        # Determine current x
        x = start + (i * step)

        # Calculate the exponent part
        deviation = (x - mean) / sigma
        exponent = -0.5 * (deviation ** 2)

        # Final y value
        y = constant_term * math.exp(exponent)

        x_data.append(x)
        y_data.append(y)

    return x_data, y_data


# Get the data
x, y = get_gaussian_data()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'Mean=0, Sigma=15')
plt.title('Gaussian PDF (Manual Calculation)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()