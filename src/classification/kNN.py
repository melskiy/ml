import math
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = np.argsort([self.distance(x, x_train) for x_train in self.X_train])
        k_index = distances[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_index]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

