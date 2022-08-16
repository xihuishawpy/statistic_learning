'''
Description: KNN最近邻
Version: 1.0
Autor: xihuishaw
Date: 2022-08-16 17:17:27
LastEditors: xihuishaw
LastEditTime: 2022-08-17 00:34:27
'''

from collections import Counter
import numpy as np


class KNN:
    def __init__(self , k=3):
        self.k = k
    
    # 欧式距离
    def _distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    
    def _predict(self,x):
        # 对于每个数据点，计算与其他数据点的欧式距离
        distances = [self._distance(x,x_train) for x_train in self.X_train]
        # 返回最近的k个数据点索引及对应的y
        k_idxs = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_idxs]
        # 找出类别数最多的类
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    
    
from sklearn.datasets import load_iris

if __name__ == '__main__':
    X , y  = load_iris(return_X_y=True)
    knn = KNN()
    knn.fit(X,y)
    y_pred = knn.predict(X)
    print(y_pred)   
    # print(np.sum(y_pred==y) / len(y)) # 看看聚类效果 0.96
    
    