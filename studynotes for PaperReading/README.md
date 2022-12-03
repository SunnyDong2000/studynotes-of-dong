# Dropout BN LN.ipynb
## Dropout
尝试解决一个过大的前馈神经网络应用到一个小的数据集上时对测试数据表现不佳的过拟合问题。

在训练网络时，随机失活部分神经元不参与本轮的权重更新。相当于训练了多个模型，最终以其结果的平均作为最终输出，但实际效果不如真的训练多个模型，不过仍然能起到防止过拟合的作用。

在pytorch中采用torch.nn.Dropout(p)函数来实现dropout层，p表示神经元有p的概率失活。

提出论文：Improving neural networks by preventing co-adaptation of feature detectors
## BN层
尝试解决内部协方差偏移(internal covariate shift)，某些情况下可以代替Dropout。

在向网络喂入数据时，我们会先将数据进行归一化等预处理。但网络中每一层的参数是随机且会更新的，因此与输入数据进行运算后这种数据的分布并不能得到保持，这种情况被称为内部协方差偏移。而内部协方差偏移可能使网络中每一层的输出值过大或过小，在激活时处于激活函数梯度较小的部分，在很深的神经网络中会出现梯度消失的现象，甚至出现了随着网络层数的增加，测试集效果会出现突降的现象。因此自然的想法是将每一层的输出进行归一化再进行激活，BN层是将输入的batch中图像的对应特征一起归一化。即在维度为(batchsize,C,M,N)的batch中对(:,i,:,:)进行归一化。可以起到减缓过拟合的效果，同时因为归一化后使输出处于激活函数梯度较大的部分，也可以加快网络的训练速度。

在pytorch中采用torch.nn.BatchNorm2d(n)函数来实现卷积网络的BN层，n是卷积核的个数，也就是该层网络输出的特征数目。

提出论文：Batch normalization: Accelerating deep network training by reducing internal covariate shift
## LN层
与BN层类似，实现了网络内部的归一化，但归一化维度不同。

解决的问题和效果与BN层大同小异，不同之处在于在维度为(batchsize,C,M,N)的batch中对(i,:,:,:)进行归一化。

在pytorch中采用torch.nn.LayerNorm([C,M,N])函数来实现卷积网络的LN层，[C,M,N]是该层网络输出的单个数据的维度。

提出论文：Layer normalization

## 对比
![$~~B226{JCDM)_C% F9EY)H](https://user-images.githubusercontent.com/116711111/205441126-258e5b97-3f89-4557-840f-1e64cd603120.png)

横轴为epoch数目，纵轴为分类准确率，虚线为训练集，实线为测试集。红色为对比数据(不加入dropout、BN或LN)，蓝色为dropout，绿色为BN，粉色为LN。其余的网络结构、初始化seed、训练集、测试集等保持一致。
