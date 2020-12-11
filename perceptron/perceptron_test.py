import os
import numpy as np
from utils import read_mnist_01
from perceptron import Perceptron
import time


if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = read_mnist_01()

    train_y[np.where(train_y != 1)] = -1
    test_y[np.where(test_y != 1)] = -1

    s = time.time()
    p = Perceptron()
    p.fit_dual(train_X, train_y)
    print('耗时：%f' % (time.time()-s))
    print(p.score(train_X, train_y), p.score(test_X, test_y))
