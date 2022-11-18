
# 学习顺序
## CS231n(知乎上的翻译笔记)
首先学习cs231n，最近邻分类器->k最近邻分类器->多类SVM->softmax分类器(未实现)->全连接神经网络->卷积神经网络(pytorch实现)

首先在此过程中了解训练集、测试集、交叉验证、超参数、损失函数、激活函数、最优化、梯度、正向传播、反向传播、数据预处理等概念。

之后通过手写svm及全连接神经网络，着重感受矩阵乘法中维度的变换，从而加深对预测和训练过程的理解。

最后在学习完pytorch后，利用pytorch这个工具完成卷积神经网络

##pytorch的学习(pytorch手册+csdn)
首先学习pytorch中的张量tensor的各种基本操作(创建、加法、与nparray互相转换、cpu和gpu之间移动等)，特别注意的是nparray和tensor互相转换时，共用同一块内存。

之后了解pytorch的tenzor自带的求取梯度的方法。根据反向转播时最终的tensor是否为标量，决定是否必须要额外参数。

接下来在了解了tensor的求导的基础上，学习如何定义继承一个网络的类，并实现正向传播。之后了解如何进行反向传播及更新权重。这里需要注意的是喂入数据的格式为(batchsize,C,M,N)

最后根据已学习的知识，实现最开始学习CS231n中的最后一项任务，即用cnn实现分类。(这里需要注意的是：不同层之间的数据维度是怎么变换的，以方便后续自己设计网络结构)

补充：dataset和dataloeader(未进行)；GPU并行运算(未进行)
