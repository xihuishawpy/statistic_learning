'''
Description: 朴素贝叶斯
Version: 1.0
Autor: xihuishaw
Date: 2022-08-14 00:38:48
LastEditors: xihuishaw
LastEditTime: 2022-08-14 00:40:27
'''

# https://www.cnblogs.com/geo-will/p/10468401.html

import numpy as np



class NaiveBayes:

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # 计算每个类的特征均值、方差、先验占比
        self._mean = np.zeros((n_classes,n_features), dtype=np.float64)
        self._var = np.zeros((n_classes,n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes,dtype=np.float64)

        for idx,c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[idx,:] = X_c.mean(axis=0)
            self._var[idx,:] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)


    def _predict(self , x):
        posteriors = []
        # 对于每个类 
        for idx , c in enumerate(self._classes):
            # 计算先验占比
            prior = np.log(self._priors[idx])
            # 计算x的条件概率
            posterior = np.sum(np.log(self._pdf(idx,x)))
            # log相加，方便相乘
            posterior = prior + posterior
            posteriors.append(posterior)
        # 返回后验概率最大对应的类
        return self._classes[np.argmax(posteriors)]

    # 计算x相对于每个类的条件概率（正态分布）
    def _pdf(self,class_idx,x):
        mean = self._mean(class_idx)
        var = self._var[class_idx]
        numerator = np.exp(-((x-mean)**2)/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator / denominator



    