#!/usr/bin/python3

""" Implement normal equations method from scratch and compare your results on
 a simulated dataset (disease score fluctuation as target) and the admissions dataset """
""" You can compare the results with scikit-learn and your own gradient descent implementation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


""" Loading the data """
def loadData():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data


""" Giving x and y values """
def x_y_form(data):
    x = data.drop("disease_score_fluct", axis=1).values
    y = data["disease_score_fluct"].values
    y = y.reshape(-1, 1)
    return x, y


""" Splitting data into training and test sets """
def splitData(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=250)
    print("X_train shape:", x_train.shape)
    print("X_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    return x_train, x_test, y_train, y_test


""" Hypothesis function """
def hypothesis(x, theta):
    return np.dot(x, theta)


""" Cost function """
def cost(theta, x, y):
    m = len(y)
    y_pred = hypothesis(x, theta)
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost


""" Compute the derivative of the cost function """
def derivative(x, y, theta):
    m = len(x)
    y_pred = hypothesis(x, theta)
    gradient = (1 / m) * np.dot(x.T, (y_pred - y))
    return gradient


"""Updating theta using Normal Equations"""
def update_theta(x, y):
    x_transpose_x = np.linalg.inv(x.T.dot(x))
    x_transpose_y = np.dot(x.T, y)
    theta = np.dot(x_transpose_x, x_transpose_y)
    return theta


""" Computing r2score """
def r2Score(y_true, y_pred):
    a = np.sum((y_true - y_pred) ** 2)
    b = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (a / b)


""" Normalizing the Data """
def normalize(x):
    m = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - m) / sigma
    return x_norm, m, sigma


""" Gradient descent loop """
def gradient_descent(x, y, alpha, iteration):
    # Initialize theta to zeros (not optimal theta!)
    theta = np.zeros((x.shape[1], 1))
    cost_history = []

    for i in range(iteration):
        gradient = derivative(x, y, theta)
        theta = theta - (alpha * gradient)
        current_cost = cost(theta, x, y)
        cost_history.append(current_cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {current_cost:.4f}")

    return theta, cost_history


""" main function """
def main():
    # Load data
    data = loadData()
    x_raw, y = x_y_form(data)

    # Split data
    x_train_raw, x_test_raw, y_train, y_test = splitData(x_raw, y)

    # Normalize features
    x_train_norm, mean, sigma = normalize(x_train_raw)
    x_test_norm = (x_test_raw - mean) / sigma     #formula for normalisation

    # Add bias term (intercept)
    x_train = np.c_[np.ones((x_train_norm.shape[0], 1)), x_train_norm]   # adds a column of 1s to represent bias term θ₀
    x_test = np.c_[np.ones((x_test_norm.shape[0], 1)), x_test_norm]      # same bias term added to test data

    print("\n" + "=" * 60)
    print("METHOD 1 : NORMAL EQUATIONS")
    print("=" * 60)
    theta_normal = update_theta(x_train, y_train)       #computes parameters using closed form solution
    print("Theta from Normal Equations:\n", theta_normal)

    # Predictions and evaluation
    y_pred_train_normal = hypothesis(x_train, theta_normal)      #hypothesis = x * theta
    y_pred_test_normal = hypothesis(x_test, theta_normal)

    train_cost_normal = cost(theta_normal, x_train, y_train)     #cost we need three inputs theta, x, y
    test_cost_normal = cost(theta_normal, x_test, y_test)

    print(f"\nTrain Cost: {train_cost_normal:.4f}")            
    print(f"Test Cost: {test_cost_normal:.4f}")
    print(f"Train R²: {r2Score(y_train, y_pred_train_normal):.4f}")
    print(f"Test R²: {r2Score(y_test, y_pred_test_normal):.4f}")

    print("\n" + "=" * 60)
    print("METHOD 2: GRADIENT DESCENT")
    print("=" * 60)
    alpha = 0.01  # Learning rate
    iteration = 2000

    theta_gd, cost_history = gradient_descent(x_train, y_train, alpha, iteration)
    print("\nTheta from Gradient Descent:\n", theta_gd)

    # Predictions and evaluation
    y_pred_train_gd = hypothesis(x_train, theta_gd)          #hypothesis = x * theta
    y_pred_test_gd = hypothesis(x_test, theta_gd)           

    train_cost_gd = cost(theta_gd, x_train, y_train)            #cost we need three inputs theta, x, y
    test_cost_gd = cost(theta_gd, x_test, y_test)

    print(f"\nTrain Cost: {train_cost_gd:.4f}")
    print(f"Test Cost: {test_cost_gd:.4f}")
    print(f"Train R²: {r2Score(y_train, y_pred_train_gd):.4f}")
    print(f"Test R²: {r2Score(y_test, y_pred_test_gd):.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON WITH SKLEARN")                        #Comparison with scikit library r2 score
    print("=" * 60)
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(x_train_norm, y_train)

    y_pred_sklearn_train = lr.predict(x_train_norm)
    y_pred_sklearn_test = lr.predict(x_test_norm)

    print(f"Train R² (sklearn): {r2_score(y_train, y_pred_sklearn_train):.4f}")
    print(f"Test R² (sklearn): {r2_score(y_test, y_pred_sklearn_test):.4f}")

if __name__ == "__main__":
    main()