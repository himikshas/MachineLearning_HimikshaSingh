import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
   return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def main():

    x = np.linspace(-10, 10, 100)
    y_sigmoid = sigmoid(x)
    y_sigmoid_derivative = sigmoid_derivative(x)

    plt.plot(x, y_sigmoid_derivative)
    plt.grid(True)
    plt.title("Sigmoid Derivative Function")
    plt.xlabel("sigmoid(z)")
    plt.ylabel("z")
    plt.legend(['x', 'y'])
    plt.show()


if __name__ == "__main__":
    main()
