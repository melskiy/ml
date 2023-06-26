import numpy as np

def pca(X, n_components):
    X_meaned = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    cov_matrix = np.cov(X_meaned, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = sorted_eigenvectors[:, :n_components]
    X_pca = np.dot(X_meaned, principal_components)
    return X_pca

