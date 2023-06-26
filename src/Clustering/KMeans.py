import numpy as np

class myKMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            labels = self.get_labels(X)

            new_centroids = self.update_centroids(X, labels)

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def get_labels(self, X):
        dists = self.distance(X)
        return np.argmin(dists, axis=0)
    def distance(self, X):
        return np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
    
    def update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            centroids[i, :] = np.mean(X[labels == i, :], axis=0)
        return centroids

    def predict(self, X):
        return self.get_labels(X)
