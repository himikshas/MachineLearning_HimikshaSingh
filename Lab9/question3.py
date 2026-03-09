#!/usr/bin/python3

""" Implement a classification decision tree algorithm using scikit-learn for the sonar  dataset.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


""" Load Data """
def load_data():
    data = pd.read_csv("Copy of sonar data.csv", header=None)
    return data


""" EDA """
def eda(data):

    print("Dataset Shape:", data.shape)

    print("Info of dataset:")
    print(data.info())

    print("\nFirst 5 Rows:" )
    print(data.head())

    print("\nLast 5 Rows:")
    print (data.tail())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDescription of dataset:")
    print(data.describe())

    return data


""" Form X and Y """
def x_y_form(data):

    x = data.iloc[:, :-1].values   #dropping last column (-1 means last column)
    y = data.iloc[:, -1].values

    return x, y



""" Train Test Split """
def split_data(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42 )

    return x_train, x_test, y_train, y_test



""" Feature Encoding """
def feature_encoding(y_train, y_test):

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)   #
    y_test_encoded = encoder.transform(y_test)

    return y_train_encoded, y_test_encoded


""" Preprocessing """
def preprocessing(x_train, x_test):

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)   #we only standardize features
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled



""" Train Model """
def train_model(x_train_scaled, y_train_encoded):

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    model.fit(x_train_scaled, y_train_encoded)  #learns parameters and relationship between input variables (x_train) and output variables (y_train)

    return model


""" Evaluate Model """
def evaluate_model(model, x_test_scaled, y_test_encoded):

    y_pred = model.predict(x_test_scaled)   #use x_test only to predict y(target) value
    acc = accuracy_score(y_test_encoded, y_pred)     #accuracy score is the comparision between predicted output(y_pred) and actual output value(y_test)

    print("\nAccuracy:", acc)


""" Main Function """
def main():

    # 1 Load Data
    data = load_data()

    # 2 EDA
    data = eda(data)

    # 3 Form X and Y
    x, y = x_y_form(data)

    # 4 Train Test Split
    x_train, x_test, y_train, y_test = split_data(x, y)

    # 5 Feature Encoding
    y_train_encoded, y_test_encoded = feature_encoding(y_train, y_test)

    # 6 Preprocessing (includes standardization)
    x_train_scaled, x_test_scaled = preprocessing(x_train, x_test)

    # 7 Train Model
    model = train_model(x_train_scaled, y_train_encoded)

    # 8 Evaluate Model
    evaluate_model(model, x_test_scaled, y_test_encoded)


# Run program
if __name__ == "__main__":
    main()