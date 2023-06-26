import numpy as np
from math import sqrt, pi,exp


class mySVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.001, n_iters=5000,kernel = 'linear',n = 1):
        self.lr = learning_rate 
        self.lambda_param = lambda_param 
        self.n_iters = n_iters 
        self.weights = None 
        self.bias = None 
        self.kernel = kernel
        self.n = n
    def kernels(self,k,X):
        if k == 'linear':
            return np.dot(X, self.weights)
        if k == 'poly':
            return np.dot(X, self.weights)**self.n
        if k == 'rbf':
            return np.exp(-0.01*(np.sqrt(np.dot((X -self.weights),(X -self.weights))))**2)      
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1) 
        self.weights = np.zeros(n_features)
        self.bias = 0 
        for _ in range(self.n_iters): 
            for i in range(n_samples): 
                M = self.kernels(self.kernel,X[i])
                if y_[i] * (M + self.bias) >= 1: 
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights) 
                else: 
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - y_[i] * X[i]) 
                    self.bias -= self.lr * (-y_[i]) 
        print(self.weights)
        
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias 
        predictions = np.sign(linear_output) 
        predictions = np.where(predictions <= 0, 0, 1) 
        return predictions
    



u = mySVM(kernel = 'linear')
X = np.array([
                [1,2],
              [2,6,],
              [3,6],
              [4,7],
              [7,5]])







y = np.array([1,0,1,0,0])
i = np.array([[1,1],[1,6],[2,6],[4,7],[3,5]])
g = u.fit(X,y)
y_pred = u.predict(i)

print(y_pred)


