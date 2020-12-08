import os
import numpy as np
from utils import data
from perceptron import Perceptron


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = data.read_mnist_train(os.getcwd())
    train_y[np.where(train_y != 1)] = -1
    test_y[np.where(test_y != 1)] = -1

    p = Perceptron()
    p.fit(train_X, train_y)
    print(p.score(test_X, test_y))
