'''
Description: PCA降维
Version: 1.0
Autor: xihuishaw
Date: 2022-08-17 00:23:09
LastEditors: xihuishaw
LastEditTime: 2022-08-17 10:12:39
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

        # 求协方差矩阵，并计算特征值与特征向量
        cov = np.cov(X.T)
        eigenvalues , eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        # 按特征值从大到小，特征向量依次排序
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # 取出前n_components个特征向量，组成【主成分】
        self.components = eigenvectors[:self.n_components]

    def transform(self,X):
        # 原数据均值中心化后，再乘以主成分矩阵实现降维
        X = X-self.mean
        return np.dot(X,self.components.T)


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    X , y = load_boston(return_X_y=True)
    
    pca = PCA(n_components=10)
    pca.fit(X)
    print(pca.transform(X))




