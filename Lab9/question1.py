#!/usr/bin/env python3

"""" Write a program to partition a dataset (simulated data for regression)
 into two parts, based on a feature (BP) and for a threshold, t = 80. Generate additional two
 partitioned datasets based on different threshold values of t = [78, 82]."""


import pandas as pd


""" Load Dataset """
def load_data():
    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    return data


""" Partition Dataset """
def partition_dataset(data, threshold):

    left = data[data["BP"] <= threshold]     #creates subset where BP <= split value
    right = data[data["BP"] > threshold]    #creates subset where BP > split value

    return left, right



"""Print Partition Details"""
def print_partition(left, right, t):

    print("\nThreshold =", t)

    #shows left subset
    print("Left Partition (BP <= t):", left.shape) #shape returns (rows, columns)
    print(left.head())   #shows first five rows

    #shows right subset
    print("\nRight Partition (BP > t):", right.shape)
    print(right.head())


""" Main Function """
def main():

    #load dataset
    data = load_data()

    #print original dataset (rows, columns)
    print("Original Dataset Shape:", data.shape)

    #given thresholds
    thresholds = [80, 78, 82]

    for t in thresholds:          #loop through three thresholds
        left, right = partition_dataset(data, t)   #calls for the function partition_dataset with args as data and split value(threshold value)
        print_partition(left, right, t)    #shows left and right subsets with threshold with sample rows


if __name__ == "__main__":
    main()