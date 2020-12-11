import numpy as np
from neighbors import KNearestNeightbors
from sklearn.neighbors import KNeighborsClassifier
import utils

if __name__ == '__main__':
    # dataIn = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    # dataLable = np.array([0, 0, 1, 0, 1, 1])

    # p = KNearestNeightbors(1)
    # p.fit(dataIn, dataLable)
    # print(p.score(np.array([[3, 4.5]]), np.array([0])))

    (train_X, train_y), (test_X, test_y) = utils.read_mnist_01()

    p = KNearestNeightbors(5)
    p.fit(train_X, train_y)
    print(p.score(test_X, test_y))

    # p = KNeighborsClassifier()
    # p.fit(train_X, train_y)
    # # 0.9996052112120016 0.9990543735224586
    # print(p.score(train_X, train_y), p.score(test_X, test_y))
