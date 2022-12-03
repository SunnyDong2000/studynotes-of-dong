# Dropout BN LN.ipynb
Dropout，尝试解决一个过大的前馈神经网络应用到一个小的数据集上时对测试数据表现不佳的过拟合问题。提出论文：Improving neural networks by preventing co-adaptation of feature detectors
BN层，尝试解决内部协方差偏移(internal covariate shift)，某些情况下可以代替Dropout。提出论文：Batch normalization: Accelerating deep network training by reducing internal covariate shift
LN层，与BN层类似，实现了网络内部的归一化，但归一化维度不同。提出论文：Layer normalization
