#!/usr/bin/python3

""" Implement y = 2x1 + 3 and plot x1, y [start=-100, stop=100, num=100] """

import matplotlib.pyplot as plt

# Setup range parameters

start = -100
stop = 100
num =100

step = (stop - start) / (num - 1)

# Generate x and y using list comprehensions

x1 = [start + (i * step) for i in range(num)]
y = [(2 * x) + 3 for x in x1]

# Plot

plt.plot(x1, y)
plt.title("y = 2x1 + 3")
plt.xlabel("x1")
plt.ylabel("y")
plt.grid(True)
plt.show()