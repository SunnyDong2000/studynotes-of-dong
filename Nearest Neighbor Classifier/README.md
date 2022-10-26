# 基本信息
本代码使用python版本的CIFAR-10数据集，下载地址为http://www.cs.toronto.edu/~kriz/cifar.html 数据集图片示意如下：
![cifar10](https://user-images.githubusercontent.com/116711111/197977393-5531bdff-ef91-4521-afc9-a8766fb6adfe.jpg)

# 代码流程
## NearestNeighborClassifier
读取CIFAR-10数据集，得到训练集和测试集。调用train函数训练，最近邻方法的train实际上为读取所有数据。调用predict函数预测，实际上为计算每个测试集图片与每个训练集图片之间的距离，距离最短的图片标签作为测试图片的预测标签。
## k-NearestNeighborClassifier
读取CIFAR-10数据集，分割训练集，得到训练集、验证集和测试集（未采用交叉验证）。调用train函数训练，最近邻方法的train实际上为读取所有数据。设置knum，调用predict函数分别预测并计算k = 1 到 k = 2 * knum - 1的所有准确率（本代码只取k为奇数）。取准确率最高的k，调用predict函数预测测试集标签。
***需要注意***的是，与NearestNeighborClassifier不同的地方在于，predict函数取距离最短的k张图片标签，再选取出现次数最多的标签作为测试图片的预测标签。
