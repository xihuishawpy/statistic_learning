<!--
 * @Description: LDA
 * @Version: 1.0
 * @Autor: xihuishaw
 * @Date: 2021-12-13 22:54:50
 * @LastEditors: xihuishaw
 * @LastEditTime: 2021-12-13 23:30:47
-->

# LDA - 线性判别分析

算法思想：**投影后类内方差最小，类间方差最大。**

>将数据在低维度上进行投影，投影后希望每一种类别数据的投影点尽可能的接近，而不同类别的数据的类别中心之间的距离尽可能的大。

![20211213225936](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211213225936.png)

<center><font size=2 >右边的划分更好</font></center>

换个方式看，哪一条直线对应的投影更好？答案肯定是左侧的直线，因为无重叠，分布比较瘦，2个分布的分离度较大。

![20211213230530](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211213230530.png)

用一个数学表达式，去描述2个正态分布的组间方差大，组内方差小，其分子代表组间方差，分母代表组内方差：
<div align=center>
<img src="https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211213231158.png">
</div>

`决策边界`就是`两个分布的均值的中垂线`

<div align=center>
<img src="https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211213231619.png">
</div>

从求解过程可以看出，LDA算法比较`依赖于分布的均值`：

![20211213231837](https://cdn.jsdelivr.net/gh/xihuishawpy/PicBad@main/blogs/pictures/20211213231837.png)

LDA分类基本思想

- 假设各个类别的样本数据符合`高斯分布`，这样利用LDA进行投影后，可以`利用极大似然估计`计算各个类别投影数据的均值和方差，进而得到该类别高斯分布的概率密度函数。

- 当一个新的样本到来后，我们可以将它投影，然后将投影后的样本特征分别带入各个类别的高斯分布概率密度函数，计算它属于这个类别的概率，最大的概率对应的类别即为预测类别。

一般，**LDA用来降维，最多降到类别数k-1的维数，如果数据分布不是高斯分布，则效果不好**。

--------

相关阅读：

1. <https://www.cnblogs.com/pinard/p/6244265.html>
2. <https://www.zhihu.com/question/34305879/answer/2182905513>
