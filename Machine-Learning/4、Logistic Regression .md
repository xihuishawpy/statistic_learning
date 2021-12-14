<!--
 * @Description: LR
 * @Version: 1.0
 * @Autor: xihuishaw
 * @Date: 2021-12-14 00:06:59
 * @LastEditors: xihuishaw
 * @LastEditTime: 2021-12-15 00:06:54
-->

# LR

Logistic回归的本质：假设数据服从这个分布，然后使用极大似然估计做参数的估计。

## Logistic分布

Logistic分布是一种概率分布，其分布函数与密度函数如下：

![20211214224812](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214224812.png)

其图像特征（分布与正态分布相似）：

![20211214225019](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214225019.png)

## Logistic回归

我们知道感知机的原理是用一条线完成线性分割，假设某个样本点w1x1+w2x2+b>0，那么可以判断它的类别为 1。

而Logistic回归则是在回归函数的基础上得到分类的概率与分类之间的联系，所以才叫Logistic回归。

也就是将回归函数代入Logistic分布中去：

![20211214231100](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214231100.png)

将回归函数表达为（y/(1-y) ，为几率）：
![20211214231208](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214231208.png)
再次重写，回归函数可表达为`对数几率`：
![20211214231327](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214231327.png)

逻辑回归的思路:
>先拟合决策边界(不局限于线性，还可以是多项式)，再建立这个边界与分类的概率联系，从而得到了二分类情况下的概率。

1. 先拟合回归函数（或者其他多项式，甚至其他非线性函数，类似于感知机的过程）；
2. 决策边界拟合后，再通过Logistic分布转化为概率分布；

对数几率的意义：

1. 对分类建模，不需假设数据的分布情况；
2. 预测出类别，还能预测出概率；
3. 对数几率为可导凸函数，方便优化得到最优解；

## 损失函数

估计模型参数，Logistic回归采用极大似然估计

对数似然函数：

![20211214232643](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214232643.png)

随机梯度下降或牛顿法求解。

## 正则化

1、L1正则--Lasso回归

似然函数：

![20211214233242](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214233242.png)

目标函数：

![20211214233314](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214233314.png)

等价于原始损失函数的后面加上了 L1 正则，因此 `L1 正则的本质其实是为模型增加了“模型参数服从零均值拉普拉斯分布”这一先验知识`。

2、L2正则--Ridge回归

似然函数：

![20211214233431](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214233431.png)

目标函数：

![20211214233510](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211214233510.png)

等价于原始的损失函数后面加上了 L2 正则，因此 `L2 正则的本质其实是为模型增加了“模型参数服从零均值正态分布”这一先验知识`。

- **L1 正则化**就是在 loss function 后边所加正则项为 L1 范数，加上 L1 范数容易得到稀疏解（0 比较多）。它能实现`特征自动选择`，**去除无用特征**。

- **L2 正则化**就是 loss function 后边所加正则项为 L2 范数的平方，加上 L2 正则相比于 L1 正则来说，得到的解比较平滑（不是稀疏），但是同样能够保证解中`接近于 0（但不是等于 0，所以相对平滑）`的维度比较多，**降低模型的复杂度**。

------

相关阅读：

1. <https://zhuanlan.zhihu.com/p/74874291>（值得反复翻看）
