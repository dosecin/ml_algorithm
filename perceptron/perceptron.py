import numpy as np
import random


class Perceptron:
    def __init__(self, n_iter: int = 1000, eta: float = 0.5):
        self.n_iter = n_iter
        self.eta = eta
        self.epsilon = 0.999

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w: np.ndarray = np.zeros(X.shape[1])
        self.b: float = 0

        n_sample, n_feature = X.shape
        n_modity = 0
        for _ in range(self.n_iter):
            ridx = np.random.randint(0, n_sample, 1)
            for idx in ridx:
                xi, yi = X[idx], y[idx]
                if yi*(np.dot(xi, self.w)+self.b) <= 0:
                    self.w += self.eta*yi*xi
                    self.b += self.eta*yi
                    n_modity += 1
                    if n_modity % 1000 == 0:
                        if self.score(X, y) >= self.epsilon:
                            break

    def predict(self, X: np.ndarray):
        return np.sign(np.dot(X, self.w)+self.b)

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        n_sample = X.shape[0]
        return np.sum(y_pred == y)/n_sample
