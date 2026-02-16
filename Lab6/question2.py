#!/usr/bin/python3


""" Data normalization - scale the values between 0 and 1. Implement code from scratch. """

#Example:
# a = (100.5, 200.3, 300.4, 678, 564)
# 
# for i in a:
#   x_new = ( i - min(a) ) / (max(a) - min(a))
#   print(x_new)


import pandas as pd
import numpy as np



""" to load data """
def loadData():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    return data



""" Giving x and y values """
def x_y_form(data):
    x = data.drop(["disease_score", "disease_score_fluct"], axis=1).values  # droping the target value which is y
    y = data["disease_score_fluct"].values  # only  the target value #axis 1 is for column
    y = y.reshape(-1, 1)  # Reshaping y so it acts like a matrix (column vector)
    return x, y



""" Normalizing the Data - Scale between 0 and 1 """
def normalize(x):
    # get min and max for each feature (column)
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)

    # avoid division by zero
    range_value = x_max - x_min
    range_value[range_value == 0] = 1

    # normalize: x_new = (x - min) / (max - min)
    x_norm = (x - x_min) / range_value

    return x_norm, x_min, x_max



""" main """
def main():
    data = loadData()
    x, y = x_y_form(data)
    x_norm, x_min, x_max = normalize(x)
    print(x_norm)



if __name__ == "__main__":
    main()