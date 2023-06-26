import random
from src.DecisinTrees.card import CART
import numpy as np
class Bagging:
    
    def __init__(self, n_estimators=10,model = "Classification"):
        self.n_estimators = n_estimators  # количество базовых моделей
        self.model = model #Classification or Regression
        
    def fit(self, X, y):
       
        self.models = []  # список базовых моделей
        # обучение базовых моделей
        for i in range(self.n_estimators):
            # выбор случайных данных с заменой
            sample_X, sample_y = self._random_sample(X, y)
            # создание модели и обучение ее на выбранной подвыборке
            model = CART(max_depth=3,model = self.model)
            model.fit(sample_X, sample_y)
            self.models.append(model)
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        if self.model == "Classification":
            return [self._most_frequent_class(p) for p in zip(*predictions)]
        else:
            return np.sum(predictions) / len(predictions)
    
    def _random_sample(self, X, y):
        n_samples = X.shape[0]
        sample_indices = [random.randint(0, n_samples-1) for _ in range(n_samples)]
        return X[sample_indices], y[sample_indices]
    
    def _most_frequent_class(self, predictions):
        return max(set(predictions), key=predictions.count)


   