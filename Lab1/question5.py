#!/usr/bin/python3

""" Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].
 Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
 What is the value of x1 at which the function value (y) is zero. What do you infer from this?"""

import matplotlib.pyplot as plt

# 1. Setup Data
start = -10
stop = 10
num = 100
step = (stop - start) / (num - 1)

x1 = [start + i * step for i in range(num)]
y = [x**2 for x in x1]

# 2. Compute Derivatives manually
points = [-5, -3, 0, 3, 5]
# Derivative of x^2 is 2x
derivs = [(p, 2 * p) for p in points]

print("Derivatives:", derivs)

# 3. Plot
plt.plot(x1, y)
# Mark the derivative points
plt.scatter(points, [p**2 for p in points], color='black')
plt.title("y = x1^2")
plt.xlabel("x1")
plt.ylabel("y")
plt.grid(True)
plt.show()