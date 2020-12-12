import numpy as np
from naive_bayes import NaiveBayes
import utils

if __name__ == '__main__':
    # S, M, L = 0, 1, 2
    # X = np.array([[1, S], [1, M], [1, M], [1, S], [1, S], [2, S], [2, M],
    #               [2, M], [2, L], [2, L], [3, L], [3, M], [3, M], [3, L], [3, L]])
    # y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    (train_X, train_y), (test_X, test_y) = utils.read_mnist_01()

    p = NaiveBayes()
    p.fit(train_X, train_y)
    print(p.score(train_X, train_y), p.score(test_X, test_y))
