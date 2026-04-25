import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


""" Load dataset """
def loadData():

    x, y  = make_hastie_10_2()

    #Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y)

    return x, y


""" Test and Train Data """
def trainTestSplit(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=2000, random_state=42, stratify=y)

    return x_train, x_test, y_train, y_test


""" Standard Scaling """
def standScaling(x_train, x_test):

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled


""" Gradient Boosting Classifier training """
def gradientBoost_train(x_train_scaled, y_train):

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(x_train_scaled, y_train)

    return model


""" Gradient Boosting Classifier evaluate """
def gradientBoost_test(model, x_test_scaled, y_test):

     y_pred = model.predict(x_test_scaled)
     accuracy = accuracy_score(y_test, y_pred)
     f1 = f1_score(y_test, y_pred)

     return accuracy, f1


""" Bagging Classifier training """
def baggingClassifier_train(x_train_scaled, y_train):

    base_model = DecisionTreeClassifier()

    model = BaggingClassifier(
        estimator=base_model,
        n_estimators=100,
        random_state=42
    )

    model.fit(x_train_scaled, y_train)
    return model


""" Bagging Classifier Evaluate"""
def baggingClassifier_evaluate(model, x_test_scaled, y_test):

    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, f1


""" SVC training """
def svm_train(x_train_scaled, y_train):

    model = SVC(kernel='rbf', random_state=42)
    model.fit(x_train_scaled, y_train)

    return model


# """ for best parameter """
# def grid_search(model, x_train_scaled, y_train):
#
#     param_grid = { 'C' : [0.01, 0.1, 1, 10]}
#
#     grid = GridSearchCV(
#         model,
#         param_grid,
#         cv=5,
#         scoring='accuracy'
#     )
#
#     grid.fit(x_train_scaled, y_train)
#
#     print("\n Grid Search Results : \n")
#     print("Best C:", grid.best_params_['C'])
#     print("Best Score:", grid.best_score_)
#
#     return grid.best_estimator_


""" SVM evaluate """
def svm_evaluate(model, x_test_scaled, y_test):

    y_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, f1


""" Main """
def main():

    x, y = loadData()

    x_train, x_test, y_train, y_test = trainTestSplit(x, y)

    x_train_scaled, x_test_scaled = standScaling(x_train, x_test)

    model_boost = gradientBoost_train(x_train_scaled, y_train)

    grd_boost_accuracy, grd_boost_f1 = gradientBoost_test(model_boost, x_test_scaled, y_test)
    print("\nGradient Boosting Classifier Results : ")
    print("\nAccuracy of Gradient Boosting Classifier:", grd_boost_accuracy )
    print("F1 score of Gradient Boosting Classifier:", grd_boost_f1 )

    model_bagging = baggingClassifier_train(x_train_scaled, y_train)

    bagging_accuracy, bagging_f1 = baggingClassifier_evaluate(model_bagging, x_test_scaled, y_test)
    print("\n Bagging Classifier Results :")
    print("\nAccuracy of Bagging Classifier:", bagging_accuracy )
    print("F1 score of Bagging Classifier:", bagging_f1 )

    model_svm= svm_train(x_train_scaled, y_train)

    # best_parameter = grid_search(model_svm, x_train_scaled, y_train)


    svm_accuracy, svm_f1 = svm_evaluate(model_svm, x_test_scaled, y_test)
    print("\nSVM Results :")
    print("\nAccuracy of SVM Classifier:", svm_accuracy )
    print("F1 score of SVM Classifier:", svm_f1 )


    #Comparing all three models
    if grd_boost_accuracy > bagging_accuracy:
        print("\nGradient Boosting Classifier is better than Bagging Classifier and svm ")
    elif svm_accuracy > bagging_accuracy:
        print("\nSVM is better than Bagging Classifier and gradient boosting ")
    else:
        print("\nBagging Classifier is better than Gradient Boosting Classifier and svm ")


    #How accurate results as compared to svm
    print("\nAccording to support vector machine the results are pretty accurate cause the data is linear and it giving more accuracy than svm ")


    #Comparing boosting and bagging with reasons
    print("\nGradient Boosting is better than Bagging cause here we are sequentially training the weak learners where as"
          " in bagging approach they train parallel so here we compute training by residual values and then"
          " compute new weights after each iteration.")


if __name__ == "__main__":
    main()




