import numpy as np

class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def r2_score(self):
        sst = np.sum((self.y_true - np.mean(self.y_true))**2)
        r2 = 1 - (self.mse() / sst)*self.y_true.shape[0]
        return r2
    
    def mae(self):
        return np.sum(np.abs(self.y_true - self.y_pred))/self.y_true.shape[0]
    
    def mse(self):
        return np.sum((self.y_true - self.y_pred)**2)/self.y_true.shape[0]
    
    def rmse(self):
        return np.sqrt(self.mse())
    
    def mape(self):
        return np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100
    
    def call(self):
        print(f'R^2:{self.r2_score()}',f'MAE:{self.mae()}',f'MSE:{self.mse()}',f'RMSE:{self.rmse()}',f'MAPE:{self.mape()}',sep = '\n')


