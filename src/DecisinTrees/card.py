import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, mse=None, value=None,info_gain=None):
        self.feature_index = feature_index  # индекс признака
        self.threshold = threshold  # пороговое значение
        self.left = left  # левый узел дерева
        self.right = right  # правый узел дерева
        self.mse = mse  # Mean Squared Error
        self.info_gain = info_gain # Information Gain
        self.value = value  # прогноз для листа (None если не лист)

class CART:
    def __init__(self, max_depth=None,model = 'Classification'):
        self.max_depth = max_depth
        self.model = model

    def fit(self, X, y):
        self.n_features_ = X.shape[1] 
        self.n_classes_ = len(np.unique(y))  
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        # Обход дерева и предсказание значения
        return [self._predict(inputs) for inputs in X]
    def gini_split(self, X, y):
        m = y.size
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        # Энтропия текущего узла
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr
    
    def mse_split(self, X, y):
        m = y.size
        mse_parent = np.var(y)
        best_mse = mse_parent
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds = np.sort(X[:, idx])
            left_values = y[np.argsort(X[:, idx])]

            # количество примеров в левом поддереве и правом поддереве
            left_cnt = 0
            right_cnt = m

            # сумма и квадратная сумма значений в левом поддереве и правом поддереве
            left_sum = 0
            right_sum = np.sum(y)

            left_sum_sq = 0
            right_sum_sq = np.sum(y ** 2)

            for i in range(1, m):               
                c = left_values[i - 1]
                left_cnt += 1
                right_cnt -= 1
                left_sum += c
                right_sum -= c
                left_sum_sq += c ** 2
                right_sum_sq -= c ** 2
                if i < self.max_depth and thresholds[i] != thresholds[i - 1]:
                    left_mse = left_sum_sq - (left_sum ** 2) / left_cnt + \
                               right_sum_sq - (right_sum ** 2) / right_cnt
                    if left_mse < best_mse:
                        best_idx = idx
                        best_thr = (thresholds[i] + thresholds[i - 1]) / 2
                        best_mse = left_mse

        return best_idx, best_thr
    
    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        
        if self.model == 'Classification':
            return  self.gini_split(X, y)
        else:
            return self.mse_split(X,y)
                

    def _grow_tree(self, X, y, depth=0):
        # Предсказание в узле
        if(self.model == "Classification"):
            num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
            predicted_value = np.argmax(num_samples_per_class)
        else:
            predicted_value = np.mean(y)

        # Создание нового листа, если достигнута максимальная глубина или нет разделений в узле
        if depth == self.max_depth or len(y) == 1:
            return Node(value=predicted_value)

        # поиск наилучшего разделения
        idx, thr = self._best_split(X, y)

        # Создаем левое и правое поддерево
        if idx is not None:
            left = np.where(X[:, idx] <= thr)
            right = np.where(X[:, idx] > thr)
            if self.model == 'Classification':
                return Node(feature_index=idx, threshold=thr,
                        left=self._grow_tree(X[left], y[left], depth + 1),
                        right=self._grow_tree(X[right], y[right], depth + 1),
                        )
            else:
                return Node(feature_index=idx, threshold=thr,
                        left=self._grow_tree(X[left], y[left], depth + 1),
                        right=self._grow_tree(X[right], y[right], depth + 1),
                        )

        else:
            return Node(value=predicted_value)

    def _predict(self, inputs):
        # Пройти по дереву до листа
        node = self.tree_
        while node.value is None:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    