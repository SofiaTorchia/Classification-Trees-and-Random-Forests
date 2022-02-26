import numpy as np
import pandas as pd
import gini
import random as rd
import node


def buildTree(data, colName, featNum, dataDim, tol):  
    if data.empty:
        return None
    else: 
        classes = data[colName].unique()   
        n = data.shape[0]
        imp = 1.0 - sum((data[colName].value_counts(sort=False)/n)**2)    
        if imp <= tol or (dataDim and data.shape[0] <= dataDim):
            maj = list(data[colName].mode())                    
            return node.Node(imp,None,None,majClass=maj)
        else:
            sData = subData(data, featNum, colName)    
            giniV, splitV = gini.gini(sData, colName, classes)  
            h = splitCouple(giniV, splitV, imp);    
  
            if h:
                newDataL = data[data[h[2]] <= h[1]]  
                newDataR = data[data[h[2]] > h[1]]    
                NL = buildTree(newDataL, colName, featNum, dataDim, tol)  
                NR = buildTree(newDataR, colName, featNum, dataDim, tol)  
                newNode = node.Node(imp, h[1], h[2], NL, NR)
            else:
                maj = list(data[colName].mode())    
                newNode = node.Node(imp, None, None, majClass=maj)
            return newNode


def subData(data,featNum,colName):
    d = data[[c for c in data if c != colName]]    
    x = d.sample(n=featNum-1,axis=1)
    x[colName]=data[colName]
    return x


def splitCouple(giniVector,splitValue,imp):   
    if imp == 0.0:
        return None
    h = None
    g = float('inf')
    for col in splitValue:
        if (splitValue[col]) and (not np.isnan(splitValue[col])) and giniVector[col] < g:
            g = giniVector[col]
            h = g, splitValue[col], col  
    return h



