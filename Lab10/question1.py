#!/usr/bin/env python3

""" Implement entropy measure using Python. The function should accept a set of data points and their
class labels and return the entropy value. """


import math
from collections import Counter

def entropy(data, y):


    # Total number of samples
    total = len(y)

    # Count occurrences of each class
    y_counts = Counter(y)

    entropy_value = 0

    # Calculate entropy
    for count in y_counts.values():
        probability = count / total
        entropy_value -= probability * math.log2(probability)

    return entropy_value


# Example usage
if __name__ == "__main__":

    # Sample dataset
    data = [1,2,3,4,5,6,7,8]
    # Corresponding class labels
    y = ['Yes','Yes','No','No','Yes','No','Yes','Yes']
    ent = entropy(data, y)
    print("Entropy:", ent)