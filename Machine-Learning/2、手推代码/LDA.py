'''
Description: 
Version: 1.0
Autor: xihuishaw
Date: 2022-08-17 23:17:22
LastEditors: xihuishaw
LastEditTime: 2022-08-18 01:15:49
'''

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import ledoit_wolf
import numpy as np
from scipy import linalg


class LDA:
    def __init__(self,n_components):
        self.priors = None
        self.covariance = None
        self.n_components = n_components
        self._mean = None
        self._classes = None


    def _class_mean(self,X,y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # 计算每个类的特征均值
        self._mean = np.zeros((n_classes,n_features), dtype=np.float64)
        for idx,c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[idx,:] = X_c.mean(axis=0)
        return self._mean
    
    def _priors(self,X):
        # 计算类占比
        _ , y_t = np.unique(y,return_inverse=True)
        return np.bincount(y_t)/len(y)


    def _cov(self,X):
        sc = StandardScaler()  
        X = sc.fit_transform(X)
        # 标准化后，估计协方差
        s = ledoit_wolf(X)[0]
        s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
        return s
    
    # 类加权方差
    def _class_cov(self,X,y):
        self.priors = self._priors(X)
        classes = np.unique(y)
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors[idx] * np.atleast_2d(self._cov(Xg))
        return cov


    def _solve_eigen(self,X,y):
        self._mean = self._class_mean(X,y)
        self.covariance_ = self._class_cov(X,y)

        Sw = self.covariance_  # 类内方差
        St = self._cov(X)  # 总方差
        Sb = St - Sw  # 类间方差

        # 求实对称矩阵的特征值与特征向量
        evals, evecs = linalg.eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][: self.n_components]

        # 按特征值大小，特征向量排序
        evecs = evecs[:, np.argsort(evals)[::-1]]  
        self.scalings_ = evecs
        # self.coef_ = np.dot(self._mean, evecs).dot(evecs.T)
        # self.intercept_ = -0.5 * np.diag(np.dot(self._mean, self.coef_.T)) + np.log(self.priors)

    def fit(self,X,y):
        self._solve_eigen(X,y)
    
    def transform(self,X):
        X_new = np.dot(X,self.scalings_)
        return X_new[:,:self.n_components]
    


if __name__ == '__main__':
    X,y = load_iris(return_X_y=True)
    lda = LDA(n_components=2)
    lda.fit(X,y)
    X_new = lda.transform(X)
    print(X.shape,X_new.shape)

    



        

        # 
    

