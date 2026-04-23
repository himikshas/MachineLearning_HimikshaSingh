#!/usr/bin/env python3

""" 1.
but non-linear separation between the two classes. Plot the data. Show that in this setting, a support vector
machine with a polynomial kernel (with degree greater than 1) or a radial kernel will outperform a support vector
classifier on the training data. Which technique performs best on the test data? Make plots and report training and
test accuracy in order to back up your assertions.

Hints:
a.
b.

Generate a simulated two-class data set with 100 observations and two features in which there is a visible

(15 Marks)

Use sklearn.datasets make_moons to generate 100 training and 100 test samples. Use noise = 0.3.
Use the below plot_clf() function to draw decision boundaries.
def plot_clf (model, df, grid_range, show_contours=False, show_support_vectors=False) :
# Decision boundary plot

# Get grid of values in given range
x1 = grid_range
x2 = grid_range
xx1, xx2 = np.meshgrid(x1, x2, sparse=False)
Xgrid = np.stack((xx1.flatten(), xx2.flatten() )).T

# Get decision boundary values for plot grid
decision_boundary = model.predict (Xgrid)
decision_boundary_grid = decision_boundary.reshape (len(x2), len(x1) )

# Get decision function values for plot grid
decision_function = model.decision_function (Xgrid)
decision_function_grid = decision_function.reshape (len(x2), len(x1))

fig = plt.figure(figsize=(10, 10))
if show_contours:
plt.contourf(x1, x2, decision_function_grid);
plt.contour (x1, x2, decision_boundary_grid) ;

sns.scatterplot (x='x1', y='x2', hue='y', data=df)
if show_support_vectors:
sns.scatterplot (x=model. support_vectors_[:, 0], y=model. support_vectors_[ :, 1],
color='red', marker='+', s=500)
plt.show()"""

#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ==============================
# 1. Generate Data
# ==============================
def generate_data():
    X_train, y_train = make_moons(n_samples=100, noise=0.3, random_state=1)
    X_test, y_test = make_moons(n_samples=100, noise=0.3, random_state=2)

    train_df = pd.DataFrame({'x1': X_train[:,0], 'x2': X_train[:,1], 'y': y_train})
    test_df = pd.DataFrame({'x1': X_test[:,0], 'x2': X_test[:,1], 'y': y_test})

    return X_train, X_test, y_train, y_test, train_df, test_df


# ==============================
# 2. Plot Data
# ==============================
def plot_data(df, title):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='x1', y='x2', hue='y', data=df)
    plt.title(title)
    plt.show()


# ==============================
# 3. Plot Decision Boundary
# ==============================
def plot_clf(model, df, grid_range, title):
    x1 = grid_range
    x2 = grid_range
    xx1, xx2 = np.meshgrid(x1, x2)

    Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T

    decision_boundary = model.predict(Xgrid)
    decision_boundary_grid = decision_boundary.reshape(len(x2), len(x1))

    decision_function = model.decision_function(Xgrid)
    decision_function_grid = decision_function.reshape(len(x2), len(x1))

    plt.figure(figsize=(7,7))

    plt.contourf(x1, x2, decision_function_grid, alpha=0.3)
    plt.contour(x1, x2, decision_boundary_grid)

    sns.scatterplot(x='x1', y='x2', hue='y', data=df)
    plt.title(title)

    plt.show()


# ==============================
# 4. Train Models
# ==============================
def train_models(X_train, y_train):
    models = {
        "Linear SVM": SVC(kernel='linear'),
        "Polynomial SVM (deg=3)": SVC(kernel='poly', degree=3),
        "RBF SVM": SVC(kernel='rbf')
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


# ==============================
# 5. Evaluate Models
# ==============================
def evaluate_models(models, X_train, X_test, y_train, y_test):
    print("\n===== MODEL PERFORMANCE =====\n")

    for name, model in models.items():
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        print(f"{name}")
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print("-"*40)


# ==============================
# 6. Main Pipeline
# ==============================
def main():

    # Step 1: Generate Data
    X_train, X_test, y_train, y_test, train_df, test_df = generate_data()

    # Step 2: Plot Data
    plot_data(train_df, "Training Data (Non-linear)")

    # Step 3: Train Models
    models = train_models(X_train, y_train)

    # Step 4: Plot Decision Boundaries
    grid = np.linspace(-2, 3, 100)

    for name, model in models.items():
        plot_clf(model, train_df, grid, title=name)

    # Step 5: Evaluate
    evaluate_models(models, X_train, X_test, y_train, y_test)


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    main()