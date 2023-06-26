import pandas as pd
import numpy as np

class LinerReg:
    def __init__(self, lr = 0.00001):
        self.lr = lr
    def transform_(self, x):
        return np.concatenate((np.ones((len(x), 1)), x), axis = 1)

    def loss_func(self, x, y,w):
        return sum((y - np.dot(x, w)) ** 2) / x.shape[0]

    def fit(self, x, y):
        dist = np.inf
        eps = 1e-5
        X = self.transform_(x)
        w = np.zeros(X.shape[1])
        while dist > eps:
            loss = self.loss_func(X, y, w)
            w = w + self.lr * 2 * np.dot(X.T, np.dot(X, w) - y) / X.shape[0]
            dist = np.abs(loss - self.loss_func(X, y, w))
        self.w = w
        return w
    def predict(self, x):
        return np.dot(self.transform_(x), self.w)
    