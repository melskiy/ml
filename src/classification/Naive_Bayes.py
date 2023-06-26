import numpy as np
from scipy.stats import norm
import pandas as pd
from math import sqrt, pi


class NaiveBayes:

    def magic(self, x):
        p = []
        for c in np.unique(self.y_train):
            k =  self.probs.loc[c]
            for j in range(len(x)):
                k*= np.prod(
                            (1/(sqrt(2*pi)*np.array(self.stds.loc[c])[j])*np.exp(-(x-np.array(
                                self.means.loc[c])[j])**2 / (2 * np.array(self.stds.loc[c])[j]**2)))
                        )
                
            
            p.append([c,k])
        p.sort(key=lambda x: x[1])

        return p[-1][0]

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.X_train.insert(1, 'y', np.array(self.y_train))
        self.means = self.X_train.groupby(['y']).apply(np.mean)
        self.means = self.means.drop('y',axis = 1)
        self.stds = self.X_train.groupby(['y']).apply(np.std)
        self.stds = self.stds.drop('y',axis=1)
        self.probs = self.X_train.groupby(['y']).apply(
            lambda x: len(x)) / (self.X_train.shape[0])
        return self

    def predict(self, X):
        y_pred = []
        for x in np.array(X):
            y_pred.append(self.magic(x))
        return np.array(y_pred)


u = NaiveBayes()
X = [[1,2],[2,6],[3,6],[4,7],[6,5]]
X = pd.DataFrame(data=X)
y = [1,0,1,0,0]
y = pd.DataFrame(data=y)
i = pd.DataFrame(data=[[1,1],[1,6],[2,6],[4,7],[3,5]])
g = u.fit(X,y).predict(i)
print(g)