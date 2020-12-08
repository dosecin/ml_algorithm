import numpy as np
import random


class Perceptron:
    def __init__(self, n_iter: int = 50, eta: float = 0.5):
        self.n_iter = n_iter
        self.eta = eta

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_sample, n_feature = X.shape
        self.w: np.ndarray = np.zeros(n_feature)
        self.b: float = 0

        for _ in range(self.n_iter):
            n_error = 0
            ridx = np.random.permutation(n_sample)
            for idx in ridx:
                xi, yi = X[idx], y[idx]
                if yi*(np.dot(xi, self.w)+self.b) <= 0:
                    self.w += self.eta*yi*xi
                    self.b += self.eta*yi
                    n_error += 1
            if n_error == 0:
                break

    def fit_dual(self, X: np.ndarray, y: np.ndarray):
        n_sample, n_feature = X.shape
        alpha: np.ndarray = np.zeros(n_sample)
        self.b: float = 0
        g = np.dot(X, X.T)*y
        for _ in range(self.n_iter):
            n_error = 0
            ridx = np.random.permutation(n_sample)
            for idx in ridx:
                yi = y[idx]
                if yi*(np.dot(alpha, g[idx])+self.b) <= 0:
                    alpha[idx] += self.eta
                    self.b += self.eta*yi
                    n_error += 1
            if n_error == 0:
                break
        self.w = np.dot(alpha*y, X)

    def predict(self, X: np.ndarray):
        return np.sign(np.dot(X, self.w)+self.b)

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        n_sample = X.shape[0]
        return np.sum(y_pred == y)/n_sample
