'''
Description: PCA降维
Version: 1.0
Autor: xihuishaw
Date: 2022-08-17 00:23:09
LastEditors: xihuishaw
LastEditTime: 2022-08-17 00:31:02
'''


import numpy as np 

class PCA:
    def __init__(self,n_components) -> int:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self,X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov = np.cov(X.T)
        eigenvalues , eigenvectors = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvectors)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[0:self.n_components]

    def transform(self,X):
        X = X-self.mean
        return np.dot(X,self.components.T)

