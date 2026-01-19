#!/usr/bin/python3

""" Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=--10, stop=10, num=100] """

import matplotlib.pyplot as plt

# range setup

start = -10
stop = 10
num = 100
step = (stop - start) / (num - 1)

# data generation

x1 = [start + i * step for i in range(num)]
y = [(2 * x**2) + (3 * x) + 4 for x in x1]

# plot

plt.plot(x1, y)
plt.title("y = 2x1^2 + 3x1 + 4")
plt.grid(True)
plt.show()