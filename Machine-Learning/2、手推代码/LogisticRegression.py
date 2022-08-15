'''
Description: 逻辑回归
Version: 1.0
Autor: xihuishaw
Date: 2022-08-15 10:12:26
LastEditors: xihuishaw
LastEditTime: 2022-08-15 10:35:06
'''

import numpy as np

class LogisticRegression:
    def __init__(self ,learning_rate=0.001,n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def fit(self, X , y):
        n_samples , n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for _ in range(self.n_iters):
            y_predicted = np.dot(X,self.weights) + self.bias
            y_predicted = self._sigmoid(y_predicted)
            
            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T , (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            # 更新paras
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
           
    def predict(self , X):
        y_predicted = np.dot(X,self.weights) + self.bias
        y_predicted = self._sigmoid(y_predicted)
        
        y_predicted = [1 if i>0.5 else 0 for i in y_predicted]
        return np.array(y_predicted)



from sklearn.datasets import load_iris

if __name__ == '__main__':
    X , y  = load_iris(return_X_y=True)
    LR = LogisticRegression()
    LR.fit(X,y)
    y_pred = LR.predict(X)
    acc = np.sum(y_pred==y) / len(y)
    print(acc)   