import numpy as np

def dbscan(X, eps=0.5, min_samples=5):
    n_samples, n_features = X.shape
    visited = np.zeros(n_samples, dtype=bool)
    labels = np.zeros(n_samples, dtype=int)
    cluster = 0
    
    def region_query(X, idx):
        return np.where(np.sum((X - X[idx])**2, axis=1)**0.5 < eps)[0]
    
    def expand_cluster(X, idx, neighbors, labels, cluster):
        labels[idx] = cluster
        i = 0
        while i < len(neighbors):
            nidx = neighbors[i]
            if not visited[nidx]:
                visited[nidx] = True
                nneighbors = region_query(X, nidx)
                if len(nneighbors) >= min_samples:
                    neighbors = np.concatenate((neighbors, nneighbors))
            if labels[nidx] == 0:
                labels[nidx] = cluster
            i += 1
    
    for idx in range(n_samples):
        if not visited[idx]:
            visited[idx] = True
            neighbors = region_query(X, idx)
            if len(neighbors) < min_samples:
                labels[idx] = -1
            else:
                cluster += 1
                expand_cluster(X, idx, neighbors, labels, cluster)
                
    return labels