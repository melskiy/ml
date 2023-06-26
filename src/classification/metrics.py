import numpy as np
class Metriks:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
    def confusion_matrix(self):
        matrix = np.array([[0,0],[0,0]])
        for i in range(len(self.y_true)):
            matrix[self.y_true[i]][self.y_pred[i]] += 1
        return matrix
    def accuracy(self):
        matrix = self.confusion_matrix()
        return ((matrix[0][0] + matrix[1][1])/(matrix.sum()))
    def precision(self):
        matrix = self.confusion_matrix()
        return ((matrix[1][1])/(matrix[1][1]+matrix[0][1]))
    def recall(self):
        matrix = self.confusion_matrix()
        return((matrix[1][1] / matrix[1].sum()))
    def F1(self):
        return(2/(1/self.precision()+1/self.recall()))
    def allmetrics(self):
        print(f'accuracy {self.accuracy()}',f'precision {self.precision()}', f'recall {self.recall()}',f'F1 {self.F1()}',sep = '\n')