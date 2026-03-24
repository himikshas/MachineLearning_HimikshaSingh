#!/usr/bin/env python3

""" Implement Adaboost classifier using scikit-learn. Use the Iris dataset."""
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


""" Loading the dataset """
def loadData():
    dataset = load_iris()

    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    data["target"] = dataset.target

    return data


""" EDA of the dataset """
def eda(data):

    print("Shape of Dataset :", data.shape)

    print("\nData description:")
    print(data.describe())

    print("\nInformation about Dataset :")
    print(data.info())

    print("\nFirst five rows:")
    print(data.head())

    print("\nLast five rows:")
    print(data.tail())

    print("\nMissing values:")
    print(data.isnull().sum())

    return data


""" Forming x and y"""
def x_y_form(data):

    x = data.drop(["target"], axis=1)
    y = data["target"]

    return x, y


""" Splitting data into train and test sets """
def split_data(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42 )

    return x_train, x_test, y_train, y_test


""" Standardizing the dataset"""
def pre_processing(x_train, x_test):

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled


""" Training Model """
def trainModel(x_train_scaled, y_train):

    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
        )

    model.fit(x_train_scaled, y_train)

    return model

""" Evaluating trained model """
def evaluateModel(model, x_test_scaled, y_test):

    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy Score:", acc)


""" Main function """
def main():

    #1. Load the dataset
    data = loadData()

    #2. Do eda on the dataset
    data = eda(data)

    #3. Form x and y
    x, y = x_y_form(data)

    #4. Train and test split of dataset
    x_train, x_test, y_train, y_test = split_data(x, y)

    #5. Data Pre-processing
    x_train_scaled, x_test_scaled = pre_processing(x_train, x_test)

    #6. Train the model
    model = trainModel(x_train_scaled, y_train)

    #7. Evaluating or testing the model
    evaluateModel(model, x_test_scaled, y_test)



""" Run the program """
if __name__ == "__main__":
    main()
