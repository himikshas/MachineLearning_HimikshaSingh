#!usr/bin/env python3

"""
Simulate a dataset of 1000 points from a Normal distribution
with mu = 10 and sd = 3

Write a log-likelihood function and optimize it
to estimate mu and sigma
"""


import numpy as np
from scipy.optimize import minimize


""" Simulate a dataset of 1000 points from a Normal distribution """
def simulate_data():

    # Set random seed
    np.random.seed(42)

    # Generate 1000 samples
    data = np.random.normal(
        loc=10,       # mean
        scale=3,      # standard deviation
        size=1000
    )
    return data


""" Negative Log Likelihood Function """
def negative_log_likelihood(params, data):

    # Parameters
    mu = params[0]
    sigma = params[1]

    # Avoid invalid sigma
    if sigma <= 0:                                #standard deviation can never be zero or negative

        return np.inf

    n = len(data)                                 #here n = 1000

    # Log-likelihood formula
    log_likelihood = (
        -n/2 * np.log(2 * np.pi)
        -n * np.log(sigma)
        -np.sum((data - mu)**2) / (2 * sigma**2)
    )

    # Return negative log-likelihood
    return -log_likelihood


""" Optimize parameters """
def estimate_parameters(data):

    # Initial guesses for parameters
    initial_params = [0, 1]                                 #for mu and sigma

    # Optimization with bounds
    result = minimize(
        negative_log_likelihood,                             #this is loss function
        initial_params,
        args=(data,),                                        #passes data into negative_log_likelihood function
        bounds=[(None, None), (1e-6, None)]                  #this restricts parameter values
    )

    return result


""" Main """
def main():

    # Simulate data
    data = simulate_data()

    print("\nFirst 10 Data Points :")
    print(data[:10])

    # Estimate parameters
    result = estimate_parameters(data)

    # Estimated values
    estimated_mu = result.x[0]
    estimated_sigma = result.x[1]

    print("\nEstimated Mean (mu) :")
    print(estimated_mu)

    print("\nEstimated Standard Deviation (sigma) :")
    print(estimated_sigma)


if __name__ == "__main__":
    main()