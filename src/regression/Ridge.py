import numpy as np

class Ridges:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.hstack((np.ones((n_samples, 1)), X))
        A = np.eye(n_features + 1)
        A[0, 0] = 0
        w = np.linalg.inv(X.T @ X + self.alpha * A) @ X.T @ y
        self.intercept_ = w[0]
        self.coef_ = w[1:]
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ np.hstack(([self.intercept_], self.coef_))