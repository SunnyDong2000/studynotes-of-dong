# 基本信息
本代码使用python版本的CIFAR-10数据集，下载地址为http://www.cs.toronto.edu/~kriz/cifar.html 数据集图片示意如下：
![cifar10](https://user-images.githubusercontent.com/116711111/197977393-5531bdff-ef91-4521-afc9-a8766fb6adfe.jpg)

# 代码流程
## NearestNeighborClassifier
读取CIFAR-10数据集，得到训练集和测试集。调用train函数训练，最近邻方法的train实际上为读取所有数据。调用predict函数预测，实际上为计算每个测试集图片与每个训练集图片之间的距离，距离最短的图片标签作为测试图片的预测标签。
## k-NearestNeighborClassifier
读取CIFAR-10数据集，分割训练集，得到训练集、验证集和测试集（未采用交叉验证）。调用train函数训练，最近邻方法的train实际上为读取所有数据。设置knum，调用predict函数分别预测并计算k = 1 到 k = 2 * knum - 1的所有准确率（本代码只取k为奇数）。取准确率最高的k，调用predict函数预测测试集标签。
***需要注意***的是，与NearestNeighborClassifier不同的地方在于，predict函数取距离最短的k张图片标签，再选取出现次数最多的标签作为测试图片的预测标签。
## MulticlassSupportVectorMachine
读取CIFAR-10数据集，得到训练集和测试集。调用train函数训练，每次迭代都在测试集上保存一下loss和准确率，最后作图可视化每次迭代的变化。

CIFAR-10数据集有10个分类，每张图像的大小为32*32*3 = 3072。根据y = w * x + b可知：x大小为(3072,1)、w为(10,3072)、b和y为(3072,1)，为使权重更新只需更新一个矩阵，将w和b结合起来，即w为(10,3073)、x为(3073，1)，即对所有x后面加入一个x[3072,0] = 1，这样在矩阵乘法中w[10,3072]即为b，从而式子简化为y = w * x。
