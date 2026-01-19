#!/usr/bin/python3

# Define the matrix X (5 samples, 3 features)
X = [
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
]

# Define the vector theta
theta = [2, 3, 3]

# Compute X * theta
result = []

print(f"{'Row Calculation':<30} | {'Result'}")
print("-" * 45)

for row in X:
    # Calculate dot product manually: (r[0]*t[0]) + (r[1]*t[1]) + (r[2]*t[2])
    val = (row[0] * theta[0]) + (row[1] * theta[1]) + (row[2] * theta[2])

    result.append(val)

    # Optional: Print the step to verify
    calc = f"{row[0]}*2 + {row[1]}*3 + {row[2]}*3"
    print(f"{calc:<30} | {val}")

print("\nFinal Result Vector (X_theta):", result)