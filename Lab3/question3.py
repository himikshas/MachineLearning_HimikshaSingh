#!/usr/bin/python3

"""" Use the above simulated CSV file and implement the following from scratch in Python
Read simulated data csv file
Form x and y (disease_score_fluct)
Write a function to compute hypothesis
Write a function to compute the cost
Write a function to compute the derivative
Write update parameters logic in the main function"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

""" Loading Data """
def loadData():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data

""" Giving x and y values """
def x_y_form(data):
    x = data.drop("disease_score_fluct", axis=1).values   #droping the target value which is y
    y = data["disease_score_fluct"].values                #only  the target value #axis 1 is for column
    y = y.reshape(-1, 1)                                  #Reshaping y so it acts like a matrix (column vector)
    return x, y

""" Normalizing the Data  """
def normalize(x):
# This squeezes the data to be small (around 0)
    m = np.mean(x, axis=0)    #mean of each feature
    sigma = np.std(x, axis=0)   #standard deviation of each feature
    x_norm = (x - m) / sigma   #normalization so that small values won't become irrelevant
    return x_norm

""" Splitting the data into training and test sets """
def splitData(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=250)
    print("X_train shape:", x_train.shape),
    print("X_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return x_train, x_test, y_train, y_test

""" Computing the hypothesis """
def hypothesis(x, θ): #hθ = θx
    return np.dot(x, θ)   #hθ(x) = θ₀x₀ + θ₁x₁ + ..... + θdxd

""" Computing the cost """
def cost(x, y, θ):
    n = len(x)                # to make it into a smaller set of rows
    h = hypothesis(x, θ)
    error = h - y             #to generate error
    squared_error = error ** 2    #cost formula: cost function = 1/2 ∑(hθ(x(i)-y(i))²  OR square of error
    cost = (1 / (2 * n)) * np.sum(squared_error)  #summation of square errors which would be multiply by 1/2*length of input label
    return cost

""" Compute derivative """
def derivative(x, y, θ):
    n = len(x)
    p = hypothesis(x, θ)
    error = p - y
    gradient = (1/n) * np.dot(x.T, error)  #np.dot does matrix multiplication
    return gradient

""" gradient descent loop   """
def gradient_descent(x, y, α, iteration, θ):
    cost_history = []

    for i in range(iteration):
        gradient = derivative(x, y, θ)
        θ = θ - (α * gradient)  #alpha represents learning rate hyperparameter
        current_cost = cost(x, y, θ)
        cost_history.append(current_cost)

        if i % 100 == 0:
            print(f"Iteration {i}:Cost is {current_cost}")
    return θ, cost_history

""" main function"""
def main():
      data = loadData()
      x_raw, y = x_y_form(data)
      x = normalize(x_raw)

      n = len(x)
      x = np.hstack((np.ones((n, 1)), x))
      #np.ones add x0 as vector of 1 for all samples this is for adding a bias anf np.hstack is for horizontal stacking joins array column wise
      x_train, x_test, y_train, y_test = splitData(x, y)

      #Initializing parameters
      θ = np.zeros((x.shape[1], 1))    #shape is now features + 1 because added the ones column
      α = 0.01
      iteration = 1000

      θ_final, history = gradient_descent(x, y, α, iteration, θ)


      # Calculate R2 Score
      y_pred = hypothesis(x, θ_final)
      print("Shape of y:", y.shape)  # Should be (100, 1)
      print("Shape of y_pred:", y_pred.shape)  # Should be (100, 1)
      score = r2_score(y, y_pred)
      print(f"R2 Score: {score}")

      #  Plot the Cost Function
      plt.plot(range(iteration), history)
      plt.xlabel('Iterations')
      plt.ylabel('Cost')
      plt.title('Cost Function Convergence')
      plt.show()

if __name__ == "__main__":
    main()
