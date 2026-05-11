#!usr/bin/env python3

"""
Develop prediction model for Iris.csv using joint probability distribution approach
Use only the first two features, SepalLengthCm, SepalWidthCm and the target variable
Add random noise to the features
Discretize the feature values
Build a decision tree model with max_depth = 2, then, compare the accuracy of this model with the joint probability distribution method
"""


import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

""" Load Iris data """
def load_data():

    # Load iris dataset from sklearn
    iris = load_iris()

    # Take only first two features
    X = pd.DataFrame(
        iris.data[:, :2],
        columns=['SepalLengthCm', 'SepalWidthCm']
    )

    # Target variable
    y = pd.Series(iris.target)

    return X, y


""" Adding random noise to the features """
def add_noise(X):

    # Set seed for reproducibility
    np.random.seed(0)

    # Generate Gaussian noise
    noise = np.random.normal(
        loc=0,                                          #loc means mean
        scale=0.1,                                      #scale means std deviation
        size=X.shape                                    #noise matrix should have same size as dataset
    )

    # Add noise to original features
    X_noisy = X + noise

    return X_noisy


""" Discretize """
def discretize(X_train, X_test):

    #create discretizer
    discretizer  = KBinsDiscretizer(                      #creates an object that performs discretization
        n_bins=10,                                         #each feature divided into 5 groups or bins
        encode='ordinal',                                 #ordinal encoding instead of one hot for bins
        strategy='uniform'                                #all bins are of equal width
    )

    X_train_disc = discretizer.fit_transform(X_train)
    X_test_disc = discretizer.transform(X_test)

    return X_train_disc.astype(int), X_test_disc.astype(int)        #KBinsDiscretizer returns float values but we need integers


""" Train Joint Probability """
def train_joint_probability(X_train, y_train):

    #dictionary to store probabilities
    model = {}                                          #initialise as empty dictionary

    classes = np.unique(y_train)                        #np.unique() finds all different class labels

    for c in classes:                                   #loop runs for every class

        model[c] = {}                                   #for every class, create another empty dictionary

        X_c = X_train[y_train == c]                     #select only those rows where class = c

        for sample in X_c:                              #loop through each sample(each row) of a particular class one by one

            key = tuple(sample)                         #convert sample(row) into tuple --> example: [2,3] => (2,3)

            if key not in model[c]:                     #checks if the sample already exists in class dictionary as initially set as 0
                model[c][key] = 0                       #inplace key it would be example: (2,3)         eg: model[0][(2,3)] = 0    , c = 0 here

            model[c][key] += 1                          #increase occurrence count by 1 to show how many times a sample/feature combination occurred in a class

    return model


""" Predict Joint Probabilities"""
def predict_joint_probability(model, X_test):

    predictions = []                                        #created empty prediction list initially

    for row in X_test:                                      #loop through each row/sample in inference phase or X_test

        key = tuple(row)                                    #convert row into tuple as dictionary keys are tuples

        best_class = -1                                     #class with the highest count         #start with -1 as no class found so far
        best_count = -1                                     #highest count found so far

        for c in model:                                     #loop through each class in the model

            count = model[c].get(key, 1)                    #how many times does any particular feature combination appeared in class c
                                                            #.get(key,1) means if the key exists then return the occurrence of the key in a unique class
                                                            #if not present then return default_value which is 1 here (laplace smoothing) for unseen keys in inference stage
            if count > best_count:                          #count and best_count which is initially -1

                best_count = count                          # predict the highest count and the class label as best class
                best_class = c

        predictions.append(best_class)                      #predicted class is added to the prediction list (appends in the list for every unique class label)

    return predictions



""" Decision tree model """
def decision_tree(X_train, X_test, y_train):

    model = DecisionTreeClassifier(max_depth=2)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return predictions



""" main """
def main():

    # Load data
    X, y = load_data()

    print("\nOriginal Dataset")
    print(X[:5])

    # Add noise
    X_noisy = add_noise(X)

    print("\nDataset After Adding Noise")
    print(X_noisy[:5])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy,
        y,
        test_size=0.3,
        random_state=0
    )

    # Discretize data
    X_train, X_test = discretize(X_train, X_test)

    # Joint Probability Method
    jp_model = train_joint_probability(X_train, y_train)

    jp_pred = predict_joint_probability(jp_model, X_test)

    jp_accuracy = accuracy_score(y_test, jp_pred)

    # Decision Tree
    dt_pred = decision_tree(X_train, X_test, y_train)

    dt_accuracy = accuracy_score(y_test, dt_pred)

    # Print accuracies
    print("\nJoint Probability Accuracy : ", jp_accuracy)

    print("\nDecision Tree Accuracy : ", dt_accuracy)

if __name__ == "__main__":
    main()