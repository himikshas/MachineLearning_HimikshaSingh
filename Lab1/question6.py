#!/usr/bin/python3

""" Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are three independent variables.
 Compute the gradient of y at a few points and print the values."""


def calculate_function_and_gradient():
    # Let's pick a few random points to test
    # Format: [x1, x2, x3]
    test_points = [
        [0, 0, 0],  # Origin
        [1, 1, 1],  # Unit step
        [-5, 10, 2],  # Random mix
        [100, 200, 50]  # Large numbers
    ]

    print(f"{'Point (x1,x2,x3)':<20} | {'y value':<10} | {'Gradient [dx1, dx2, dx3]'}")
    print("-" * 68)

    for point in test_points:

        x1 = point[0]
        x2 = point[1]
        x3 = point[2]

        # 1. Compute y
        # y = 2x1 + 3x2 + 3x3 + 4
        y = (2 * x1) + (3 * x2) + (3 * x3) + 4

        # 2. Compute Gradient
        # Since the function is linear, the derivatives are just the coefficients.
        # dy/dx1 = 2
        # dy/dx2 = 3
        # dy/dx3 = 3
        grad = [2, 3, 3]

        # Print row
        print(f"{str(point):<20} | {y:<10} | {grad}")


# Run it
calculate_function_and_gradient()