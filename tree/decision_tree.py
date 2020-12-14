import numpy as np
import math
from collections import Counter


def entropy(data: np.ndarray):
    n_sample = data.shape[0]
    c = Counter(data)
    h = 0
    for num in c.values():
        p = num/n_sample
        h -= p*math.log2(p)
    return h


def information_gain(x: np.ndarray, y: np.ndarray):
    n_sample = x.shape[0]
    hy = entropy(y)
    c = Counter(x)
    hyx = 0
    for (xi, xn) in c.items():
        yd = y[np.where(x == xi)]
        hyx += entropy(yd)*xn/n_sample
    return hy-hyx


def information_gain_ratio(x: np.ndarray, y: np.ndarray):
    return information_gain(x, y)/entropy(x)


class DecisionNode:
    def __init__(self):
        self.dimension = None
        self.leaf = None

    def set_leaf(self, y: np.ndarray):
        c = Counter(y)
        max_num = 0
        for (v, num) in c.items():
            if num > max_num:
                max_num = num
                self.leaf = v

    def fit(self, X: np.ndarray, y: np.ndarray, features: list):
        if len(features) == 1:
            self.set_leaf(y)
            return
        if len(Counter(y)) == 1:
            self.set_leaf(y)
            return
        if X.shape[0] <= 1:
            self.set_leaf(y)
            return
        max_p = 0
        for fi in features:
            p = information_gain_ratio(X[:, fi], y)
            if max_p is None:
                max_p = p
                self.dimension = fi
            elif p > max_p:
                max_p = p
                self.dimension = fi
        features.remove(self.dimension)
        fx = X[:, self.dimension]
        c = Counter(fx)
        self.child = {}
        for xi in c.keys():
            node = DecisionNode()
            self.child[xi] = node
            cond = np.where(fx == xi)
            node.fit(X[cond], y[cond], features)

    def predict(self, x: np.ndarray):
        if self.leaf is not None:
            return self.leaf
        v: np.ndarray = x[:, self.dimension]
        c = self.child[v.item()]
        return c.predict(x)


class DecisionTree:
    def __init__(self, strategy):
        self.strategy = strategy

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = DecisionNode()
        self.tree.fit(X, y, [i for i in range(X.shape[1])])

    def predict(self, X: np.ndarray):
        y_pred = []
        for (i, xi) in enumerate(X):
            y_pred.append(self.tree.predict(xi))
        return np.array(y_pred)
