
import pandas as pd
import numpy as np
import random as rd
import buildtree as bn



def bootDataset(data):
    n = data.shape[0]
    r = np.random.randint(n,size=n)
    newData = data.iloc[r]
    OBdataset = data.drop(data.index[r])
    return [newData,OBdataset]



def makeForest(T,data,colName,featNum,dataDim,tol):
    n = data.shape[0]
    forest = np.array([])
    errVect = np.zeros((n,2)) 
    for t in range(T):
        [newData,OBdataset] = bootDataset(data)
        tree = bn.buildTree(newData,colName,featNum,dataDim,tol)
        forest = np.append(forest,tree)

        for i in range(OBdataset.shape[0]):
            j = OBdataset.iloc[i,:].name
            errVect[j,1] = errVect[j,1] + 1 
            c = getClassForTree(OBdataset.iloc[i,:],tree)
            if OBdataset.iloc[i][colName] not in c:
                errVect[j,0] = errVect[j,0] + 1

    testDim = np.count_nonzero(errVect[:,1])
    errorRate = 0
    for i in range(n):
        if errVect[i,0] > (errVect[i,1]/2):
            errorRate = errorRate + 1
    errorRate = errorRate/testDim
    return forest, errorRate



def getClassForTree(sample,tree):
    tempNode = tree
    while(tempNode.getSplitVal() != None):
        [i,val,col] = tempNode.getValues()
        if(sample[col]<=val):
            tempNode = tempNode.getLeft()
        else:
            tempNode = tempNode.getRight()
    return tempNode.getMajClass()




def getClassForForest(forest,sample,colName,classes):
    d = {c:0 for c in classes}
    for t in range(forest.shape[0]):
        c = getClassForTree(sample,forest[t])[0]
        d[c] = d[c] + 1
    maxx = max(d.values())
    keys = list(filter(lambda x: d[x]==maxx,d.keys()))
    return keys[0]
    




def accuracy(forest,data,colName):
    n = data.shape[0]
    classes = data[colName].unique() 
    errorRate = 0
    for i in range(data.shape[0]):
        c = getClassForForest(forest,data.iloc[i,:],colName,classes)
        if data.iloc[i][colName] != c:
            errorRate = errorRate + 1
    return 1-errorRate/n




def makeTrainSample(data,prop):
    n = data.shape[0]
    n = int(prop*n)
    xTrain = data.sample(n=n,axis=0)
    xTest = data.drop(data.index[xTrain.index])
    xTrain.index = [i for i in range(n)]
    return xTrain,xTest




