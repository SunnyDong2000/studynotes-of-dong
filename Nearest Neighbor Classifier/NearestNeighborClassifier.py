import numpy as np

#解码数据集 函数为CIFAR-10提供的python3版本
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#创建训练集和测试集
def CreatData(path):
    #依次加载CIFAR-10的5个batch_data,并将其合并为traindata和traindlabels
    x=[]
    y=[]
    for i in range(1,6):
        batch_path=path + 'data_batch_%d'%(i) #每个batch的地址
        batch_dict=unpickle(batch_path) #解码每个batch
        train_batch=batch_dict[b'data'].astype('float') #将每个batch的data部分以float形式存储于train_batch变量
        train_labels=np.array(batch_dict[b'labels']) #将每个batch的label部分以np.array的形式存储于train_labels变量
        x.append(train_batch)
        y.append(train_labels)
    #将5个训练样本batch(10000,3072)合并为(50000,3072)，标签合并为(50000,1)
    #np.concatenate默认axis=0:按行合并，axis=1则为:按列合并
    traindata=np.concatenate(x)
    trainlabels=np.concatenate(y)
    
    #加载测试集
    testpath=path + 'test_batch' #test_batch的地址
    test_dict=unpickle(testpath) #解码test_batch
    testdata=test_dict[b'data'].astype('float') #将test_dict的data部分以float形式存储于testdata变量
    testlabels=np.array(test_dict[b'labels']) #将test_dict的labels部分以np.array形式存储于testlabels变量
    
    #将训练集数据、训练集标签、测试集数据、测试集标签返回
    return traindata,trainlabels,testdata,testlabels

class NearestNeighborClassifier:
    def __init__(self):
        pass
    def train(self,traindata,trainlabels):
        #将traindata和trainlabels全部读取到类里面
        self.traindata = traindata
        self.trainlabels = trainlabels
 
    def predict(self,testdata):
        #得到测试集总图片数，并保存到testimg_num内
        testimg_num = testdata.shape[0]
        #创建一个维度为(testimg_num，)的np.array，用于存储预测的标签
        predlabels = np.zeros(testimg_num, dtype = self.trainlabels.dtype)
        #遍历训练集
        for i in range(testimg_num):
            #计算测试图片与训练集中所有图片的l1距离，并找到最近的图片
            distances = np.sum(np.abs(self.traindata - testdata[i,:]),axis = 1)#axis=0:列求和 axis=1:行求和
            mindistances_index = np.argmin(distances) # 取最近图片的下标
            predlabels[i] = self.trainlabels[mindistances_index] # 记录下最近图片的label
        #返回预测标签
        return predlabels

#读取训练集和测试集的数据和标签
traindata,trainlabels,testdata,testlabels = CreatData("D:/Personal_documents/DXY/code/LearningForML/cifar-10-batches-py/")

#实例化一个最近邻分类的类
nn = NearestNeighborClassifier()
#调用类中的train函数训练(实则为将训练数据全部读取)
nn.train(traindata, trainlabels)
#调用类中的predict函数预测标签
testlabels_predict = nn.predict(testdata)
# 比对测试集标签，计算准确率。测得准确率为0.385900，远低于人类识别能力;但高于空模型10%(CIFAR-10数据集共十个分类)，说明起到了一定的作用
print('accuracy:',np.mean(testlabels_predict == testlabels))
      
