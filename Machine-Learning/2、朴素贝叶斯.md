<!--
 * @Description: Naive Bayes
 * @Version: 1.0
 * @Autor: xihuishaw
 * @Date: 2021-12-11 17:15:06
 * @LastEditors: xihuishaw
 * @LastEditTime: 2021-12-12 01:41:29
-->

# 朴素贝叶斯（Naive Bayes）

## 背景

在概率论与统计学中，贝叶斯定理 (Bayes' theorem) 表达了一个事件发生的概率，而确定这一概率的方法是基于与该事件相关的条件先验知识 (prior knowledge)。而利用相应先验知识进行概率推断的过程为贝叶斯推断 (Bayesian inference)。

## 贝叶斯公式

贝叶斯公式是由条件概率推导而来：

![20211211232812](https://s2.loli.net/2021/12/11/1BVLKYOQ8JNpI9l.png)

## 贝叶斯推断

P(A)--先验概率（Prior probability），即在B事件发生之前，对A事件概率的一个判断。

P(B|A)--条件概率（似然概率），一般通过历史数据统计得到。

P(A|B)--后验概率（Posterior probability），即在B事件发生之后，对A事件概率的重新评估。或者说事情已经发生，求这件事情发生的原因是由某个因素引起的可能性的大小。后验概率的计算要以先验概率为基础。

P(B|A)/P(B)--可能性函数（Likelyhood），这是一个调整因子，使得预估概率更接近真实概率。

so，`后验概率＝先验概率 ｘ 调整因子`·

>先预估一个"先验概率"，然后加入实验结果，看这个实验到底是增强还是削弱了"先验概率"，由此得到更接近事实的"后验概率"。

## “朴素”的含义及算法思想

换另一个角度看贝叶斯公式：

一般在监督学习中，训练数据的类别（label）出现的概率，每个特征取值的概率以及在每个类别（label）对应的特征取值的概率，都是已知的，代入贝叶斯公式，就能得出在已知特征条件下的类别概率。

![20211211172101](https://s2.loli.net/2021/12/11/BmwH9ZDXQpV4r7n.png)

思想基础：

以`自变量之间的独立（条件特征独立）性`和`连续变量的正态性假设`为前提：

对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别c

![20211212001903](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211212001903.png)

举例，比如下面2个类别，取最终概率最大的类别作为输出类别：

![20211211172938](https://s2.loli.net/2021/12/11/eJ8dIvlMjic4gW6.png)

![20211212004757](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211212004757.png)

再从数据角度通俗理解：

>对于一条测试集数据来说，特征是确定的，也就是分母是一定的，现在就是想让分子最大化，最后预测类别的概率才最大；
>
>对于分子，根据训练集数据可知，p（类别）是已知的，不同类别下的不同特征的条件概率p（特征|类别）也是可得知的。
>
>目标：计算**在不同类别下，使p（类别）×p（特征|类别）最大化**，这个过程需使用`极大似然估计（MLE)`来估计对应概率。

## 算法流程

![20211212005513](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211212005513.png)

实际应用：

- 若任务对预测速度要求较高，则对给定的训练集，可将朴素贝叶斯分类器涉及的所有概率估值事先计算好存储起来，在进行预测时只需 “查表” 即可进行判别；（提前计算好训练集中的相关概率值）
- 若任务数据更替频繁，则可采用 “懒惰学习” (lazy learning) 方式，先不进行任何训练，待收到预测请求时再根据当前数据集进行概率估值；（数据频繁变化，概率值也会相应变化，只能等到数据不变或需要进行预测的时候再去计算概率进行推断）
- 若数据不断增加，则可在现有估值的基础上，仅对新增样本的属性值所涉及的概率估值进行计数修正即可实现增量学习。（数据不再变动，只是增量数据，调整增量后的概率值即可）

## scikit-learn

文档给出了6种朴素贝叶斯算法：

![20211212010901](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211212010901.png)

- GaussianNB ：特征是高斯分布，可选
- MultinomialNB ：多项式分布，可选
- ComplementNB ：适用于不平衡数据集
- BernoulliNB ：伯努利分布，可选
- CategoricalNB ：适用于离散特征

另外， MultinomialNB, BernoulliNB,  GaussianNB 这3个算法都有一个针对大规模数据处理的方法-`partial_fit`。如果训练集数据量非常大，一次不能全部载入内存的时候。我们可以把训练集分成若干等分，重复调用partial_fit来一步步的学习训练集，非常方便。

## 算法优缺点

优点：

1. `对缺失数据不太敏感`，算法也比较简单，常用于文本分类；
2. 对小规模的数据表现很好，能处理多分类任务，`适合增量式训练`（即可以实时对新增样本进行训练）；

缺点：

1. 使用了样本属性独立性的假设，在实际场景里，`很难满足各个特征之间独立`，所以如果样本属性有关联时其效果不好。

---

参考：

1. <https://www.cnblogs.com/leoo2sk/archive/2010/09/17/naive-bayesian-classifier.html>

2. <https://www.cnblogs.com/geo-will/p/10468401.html>

3. <https://zhuanlan.zhihu.com/p/26262151>
4. <https://scikit-learn.org/stable/modules/naive_bayes.html>
