#!/usr/bin/python3

# Vectors
x = [6, 8, 2]
y = [1, 2, 2]

# Calculate dot product manually
result = 0

# Loop through the list indices
for i in range(len(x)):
    # Multiply corresponding values and add to total
    term = x[i] * y[i]
    result = result + term

print("Vector x:", x)
print("Vector y:", y)
print("Dot Product:", result)

"""DOT PRODUCT : It tells you how much two vectors are pointing in the same direction.
Positive Number: They are generally pointing the same way
Zero: They are perfectly perpendicular
Negative Number: They are pointing in opposite directions

pair up the matching numbers from each list, multiply them together, and then add up all those results."""
