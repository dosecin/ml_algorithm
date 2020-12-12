import numpy as np
from collections import Counter


class NaiveBayes:
    def __init__(self, l=1):
        self.lamb = l

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_sample, n_feature = X.shape

        yc = Counter(y)
        self.py = {d: (n+self.lamb)/(n_sample+len(yc)*self.lamb)
                   for (d, n) in yc.items()}

        self.pxy = []
        self.pxy0 = {}
        for i in range(n_feature):
            xi = X[:, i]
            c = Counter(xi)
            pxy = {}
            for (yi, yn) in yc.items():
                for xil in c.keys():
                    N = yn + self.lamb*len(c)
                    m = np.where((y == yi) & (xi == xil))[0].size+self.lamb
                    pxy[(xil, yi)] = m/N
                self.pxy0[(i, yi)] = self.lamb/(yn+len(c))
            self.pxy.append(pxy)

    def predict(self, X: np.ndarray):
        n_sample, n_feature = X.shape
        y = np.zeros(n_sample)
        for (ni, xi) in enumerate(X):
            py, p = None, 0
            for (yi, pi) in self.py.items():
                cp = pi
                for (i, xil) in enumerate(xi):
                    if (xil, yi) in self.pxy[i]:
                        cp *= self.pxy[i][(xil, yi)]
                    else:
                        cp *= self.pxy0[(i, yi)]
                if cp > p:
                    p = cp
                    py = yi
            y[ni] = py
        return y

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        n_sample = X.shape[0]
        return np.sum(y_pred == y)/n_sample
