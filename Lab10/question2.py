#!/usr/bin/env python3

""" Implement information gain measures. The function should accept data points for parents,
data points for both children and return an information gain value.
"""
#!/usr/bin/env python3

import math
from collections import Counter


""" Calculating Entropy """
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)   #Counter => counts how many times each class label occurs

    ent = 0                   #initialising entropy as 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent


""" Information Gain measures """
def information_gain(parent_labels, left_child_labels, right_child_labels):


    # Parent entropy
    parent_entropy = entropy(parent_labels)

    # Sizes
    n = len(parent_labels)
    n_left = len(left_child_labels)
    n_right = len(right_child_labels)

    # Child entropies
    left_entropy = entropy(left_child_labels)
    right_entropy = entropy(right_child_labels)

    # Weighted entropy of children
    #weighted_child_entropy = (no. of samples / total samples) * entropy_of_child
    weighted_child_entropy = (n_left/n) * left_entropy + (n_right/n) * right_entropy

    # Information Gain
    ig = parent_entropy - weighted_child_entropy

    return ig


#Example usage
if __name__ == "__main__":

    parent = ['Yes','Yes','No','No','Yes','No','Yes','Yes']

    left_child = ['Yes','Yes','Yes','Yes']
    right_child = ['No','No','No','Yes']

    ig = information_gain(parent, left_child, right_child)

    print("Information Gain:", ig)