import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def loadData():
    data = pd.read_csv("data.csv")
    return data


def eda(data):

    print("Shape of dataset : ", data.shape)

    print("\nDescription of dataset : ")
    print(data.describe())

    print("\nInformation of the dataset : ")
    print(data.info())

    print("\nMissing values : ")
    print(data.isnull().sum())


def x_y_form(data):

    # data = data.drop(["id", "unnamed = 0"], axis = 1, errors="ignore")
    data = data.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")

    x = data.drop("diagnosis", axis = 1).values
    y = data["diagnosis"].values

    return x, y


def splitData(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    return x_train, x_test, y_train, y_test

def featureEncoding(y_train, y_test):

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    return y_train_encoded, y_test_encoded


def preProcessing(x_train, x_test):

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled

def trainModel(x_train_scaled, y_train):

    model = LogisticRegression(
    max_iter = 1000,
    )

    model.fit(x_train_scaled, y_train)

    return model


def evaluateModel(model, x_test_scaled, y_test):

    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print ("accuracy : ", acc)

def main():

    data = loadData()

    data =eda(data)

    x, y = x_y_form(data)

    x_train, x_test, y_train, y_test = splitData(x, y)

    y_train_encoded, y_test_encoded = featureEncoding(y_train, y_test)

    x_train_scaled, x_test_scaled = preProcessing(x_train, x_test)

    model = trainModel(x_train_scaled, y_train)

    evaluateModel(model, x_test_scaled, y_test_encoded)

if __name__ == "__main__":
    main()
