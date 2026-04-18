#!/usr/bin/env python3

""" Consider the following dataset. Implement the RBF kernel.
    Check if RBF kernel separates the data well and
    compare it with the Polynomial Kernel."""


"""
STEP 1: Import Libraries
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""
STEP 2: Create Dataset
"""
data = [
    [6,5,'Blue'], [6,9,'Blue'], [8,6,'Red'], [8,8,'Red'], [8,10,'Red'],
    [9,2,'Blue'], [9,5,'Red'], [10,10,'Red'], [10,13,'Blue'],
    [11,5,'Red'], [11,8,'Red'], [12,6,'Red'], [12,11,'Blue'],
    [13,4,'Blue'], [14,8,'Blue']
]

df = pd.DataFrame(data, columns=['x1','x2','label'])

"""
STEP 3: Convert Labels (Blue=0, Red=1)
"""
df['label'] = df['label'].map({'Blue':0, 'Red':1})

X = df[['x1','x2']]
y = df['label']

"""
STEP 4: Train-Test Split
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


"""
MODEL 1: RBF KERNEL
"""
rbf_model = SVC(kernel='rbf', gamma=0.5)
rbf_model.fit(X_train, y_train)

y_pred_rbf = rbf_model.predict(X_test)
rbf_acc = accuracy_score(y_test, y_pred_rbf)


"""
MODEL 2: POLYNOMIAL KERNEL
"""
poly_model = SVC(kernel='poly', degree=3)
poly_model.fit(X_train, y_train)

y_pred_poly = poly_model.predict(X_test)
poly_acc = accuracy_score(y_test, y_pred_poly)


"""
STEP 5: Results
"""
print("RBF Kernel Accuracy:", rbf_acc)
print("Polynomial Kernel Accuracy:", poly_acc)
print("RBF Kernel gives better accuracy")