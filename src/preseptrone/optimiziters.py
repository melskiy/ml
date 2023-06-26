import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, y=0.9, a=0.999, e=1e-8,num_layers=[],weights=[],grads =[]):
        self.learning_rate = learning_rate
        self.y = y
        self.a = a
        self.e = e
        self.v = None
        self.G = None
        self.t = 0
        self.num_layers = num_layers
        self.weights = weights
        self.grads = grads
    def optimize(self):
        v = [np.zeros_like(w) for w in self.weights]
        G = [np.zeros_like(w) for w in self.weights]
        t = 0
        for i in range(self.num_layers):
            v[i] = self.y * v[i] + (1 - self.y) * self.grads[i]
            G[i] = self.a * G[i] + (1 - self.a) * np.square(self.grads[i])
            v_norm = v[i] / (1 - self.a**(t+1))
            G_norm = G[i] / (1 - self.a**(t+1))
            weight_update = -self.learning_rate * v_norm / (np.sqrt(G_norm) + self.e)
            self.weights[i] += weight_update
            t += 1
        return self.weights

    
class RMSPropOptimizer:
    def __init__(self, learning_rate=0.001, a=0.9, e=1e-8,num_layers=[],weights=[],grads =[]):
        self.learning_rate = learning_rate
        self.a = a
        self.e = e
        self.v = None
        self.G = None
        self.t = 0
        self.num_layers = num_layers
        self.weights = weights
        self.grads = grads

    def optimize(self):
        G = [np.zeros_like(w) for w in self.weights]
        for i in range(self.num_layers):
            G[i] = self.a * G[i] + (1 - self.a) * np.square(self.grads[i])
            weight_update = -self.learning_rate * self.grads[i] / (np.sqrt(G[i]) + self.e)
            self.weights[i] += weight_update
        return self.weights

class SGD:
    def __init__(self, learning_rate=0.001,num_layers=[],weights=[],grads =[]):
            self.learning_rate = learning_rate
            self.num_layers = num_layers
            self.weights = weights
            self.grads = grads
    def optimize(self):
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate*self.grads[i]
        return self.weights

