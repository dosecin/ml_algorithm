import os
import numpy as np
from utils import data
from perceptron import Perceptron


def reduce_data(X, y):
    w = np.where((y == 0) | (y == 1))
    X = X[w]
    y = y[w]
    y[np.where(y == 0)] = -1
    return X, y


def load_data():
    train_X, train_y, test_X, test_y = data.read_mnist_train(os.getcwd())
    return reduce_data(train_X, train_y), reduce_data(test_X, test_y)


if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = load_data()

    train_y[np.where(train_y != 1)] = -1
    test_y[np.where(test_y != 1)] = -1

    p = Perceptron()
    p.fit_dual(train_X, train_y)
    print(p.score(train_X, train_y), p.score(test_X, test_y))
