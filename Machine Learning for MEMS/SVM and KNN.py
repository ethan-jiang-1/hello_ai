'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

'''
from numpy import *
import numpy as np 
from numpy import linalg as la
import operator
from os import listdir
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

'''将记录的传感器数据文件变成n*3的矩阵'''
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return, 
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        ##returnMat[index,:] = listFromLine[0:3]
        returnMat[index,:] = listFromLine[0:3]
       #classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat 
'''按列归一化每个特征向量'''    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   

#皮尔逊相关系数  
def pearsSim(inA, inB):
    if len(inA)<3:
        return 1.0
    else:
        return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]
    
def spectrum_magnitude(frames,NFFT):
    '''计算每一帧经过FFY变幻以后的频谱的幅度，若frames的大小为N*L,则返回矩阵的大小为N*NFFT
    参数说明：
    frames:即audio2frame函数中的返回值矩阵，帧矩阵
    NFFT:FFT变换的数组大小,如果帧长度小于NFFT，则帧的其余部分用0填充铺满
    '''
    complex_spectrum=np.fft.rfft(frames,NFFT) #对frames进行FFT变换
    return np.absolute(complex_spectrum)  #返回频谱的幅度值
    
def spectrum_power(frames,NFFT):
    '''计算每一帧傅立叶变换以后的功率谱
    参数说明：
    frames:audio2frame函数计算出来的帧矩阵
    NFFT:FFT的大小
    '''
    return 1.0/NFFT * np.square(spectrum_magnitude(frames,NFFT)) #功率谱等于每一点的幅度平方/NFFT    
class FeatureVal:
     def __init__(self):
         self.mean_val=0.0
         self.var_val=0.0
         self.power_val=0.0
         self.c12_val=0.0
         self.c13_val=0.0
         self.c23_val=0.0
'''从3个原始数据生成均值，方差，能量谱，3个相关系数的6位特征值'''
def GenFeatureVal(dataSet):
    DistanceVal=[]
    tmpFeatureVal=FeatureVal()
    N=dataSet.shape[0]
    for i in range(N):
        sv=dataSet[i,:]*dataSet[i,:]
        dp=sum(sv)
        mag=sqrt(dp)
        DistanceVal.append(mag)
    ##########start to calculate each feature
    ##mean    
    #print(DistanceVal)
    tmpFeatureVal.mean_val=mean(DistanceVal)
    ##var
    tmpFeatureVal.var_val=var(DistanceVal)
    ##power spec
    spec_power=spectrum_power(DistanceVal,512)
    tmpFeatureVal.power_val=np.sum(spec_power)
    ##corr
    tmpFeatureVal.c12_val=pearsSim(dataSet[:,0],dataSet[:,1]);
    tmpFeatureVal.c13_val=pearsSim(dataSet[:,0],dataSet[:,2]);
    tmpFeatureVal.c23_val=pearsSim(dataSet[:,1],dataSet[:,2]);
   # print(tmpFeatureVal.power_val)
    return tmpFeatureVal

     
def svmTest():
    hwLabels = []
    testLabels=[]
    ####Train data 
    #trainingFileList = listdir('TrainData')           #load the tap training set
    trainingFileList = listdir('accTrain')           #load the acc training set
    m = len(trainingFileList)
    FeatureValMat=zeros((m,6))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = fileStr.split('_')[0]
        hwLabels.append(classNumStr)
        RawData=file2matrix('accTrain/%s' % fileNameStr)
       ## FeatureValSet.append(FeatureVal())
        a=GenFeatureVal(RawData)
        FeatureValMat[i,0]=a.mean_val
        FeatureValMat[i,1]=a.var_val
        FeatureValMat[i,2]=a.power_val
        FeatureValMat[i,3]=a.c12_val
        FeatureValMat[i,4]=a.c13_val
        FeatureValMat[i,5]=a.c23_val
        print('TrainData/%s\r\n' % fileNameStr)
        #print(a.mean_val,a.var_val,a.power_val,a.c12_val)
        #print(FeatureValSet[i].mean_val,FeatureValSet[i].var_val,FeatureValSet[i].power_val,FeatureValSet[i].c12_val,FeatureValSet[i].c13_val,FeatureValSet[i].c23_val)
    #print(FeatureValMat)
    normTrainMat, ranges, minVals = autoNorm(FeatureValMat)
    print(normTrainMat)
    ##Test data
    trainingFileList = listdir('accTest')           #load the training set
    n = len(trainingFileList)
    TestValMat=zeros((n,6))
    for i in range(n):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = fileStr.split('_')[0]
        testLabels.append(classNumStr)
        RawData=file2matrix('accTest/%s' % fileNameStr)
       ## FeatureValSet.append(FeatureVal())
        a=GenFeatureVal(RawData)
        TestValMat[i,0]=a.mean_val
        TestValMat[i,1]=a.var_val
        TestValMat[i,2]=a.power_val
        TestValMat[i,3]=a.c12_val
        TestValMat[i,4]=a.c13_val
        TestValMat[i,5]=a.c23_val
        print('TestData/%s\r\n' % fileNameStr)
        #print(a.mean_val,a.var_val,a.power_val,a.c12_val)
        #print(FeatureValSet[i].mean_val,FeatureValSet[i].var_val,FeatureValSet[i].power_val,FeatureValSet[i].c12_val,FeatureValSet[i].c13_val,FeatureValSet[i].c23_val)
    print(TestValMat)
    
    ###start svm    
    print('SVM Estimation:')
    clf=svm.SVC(kernel='rbf', C=1, gamma=0.001) 
    clf.fit(normTrainMat,hwLabels)  # training the svc model 
    for i in range(n):
        c=np.vstack((FeatureValMat,TestValMat[i,:]))
        normTestMat, ranges, minVals = autoNorm(c)
        result = clf.predict(normTestMat[-1,:]) # predict the target of testing samples
        print("orignal label: %s,SVM result: %s"%(testLabels[i],result))
     ##start KNN
    print('KNN Estimation:')   
    neighbors = KNeighborsClassifier(n_neighbors=4)
    neighbors.fit(normTrainMat,hwLabels)
    for i in range(n):
        c=np.vstack((FeatureValMat,TestValMat[i,:]))
        normTestMat, ranges, minVals = autoNorm(c)
        result = neighbors.predict(normTestMat[-1,:]) # predict the target of testing samples
        print("orignal label: %s,KNN result: %s"%(testLabels[i],result))
  
