
import numpy as np


def gini(dataset,colName,classes):
    n = dataset.shape[0]
    k = classes.shape[0]
    
    freqVector = np.zeros(k)
    giniVector = {col: 2.0 for col in dataset if col != colName}
    splitValue = {col: -float('inf') for col in dataset if col != colName}
    
    for col in dataset:
        if col != colName:     
            data = dataset[[col,colName]]     
            data = data.sort_values(by=col)   

            d1 = {c:0 for c in classes}
            d2 = {c:0 for c in classes}
            for i in range(n):
                c = data.iloc[i][colName]
                d2[c] = d2[c]+1
                
            for i in range(n-1):
                c = data.iloc[i][colName]
                d1[c] = d1[c] + 1
                d2[c] = d2[c] - 1

                G1 = 1.0 - sum([(d1[c]/(i+1))**2 for c in d1])
                G2 = 1.0 - sum([(d2[c]/(n-1-i))**2 for c in d2]) 

                G = ((i+1)*G1 + (n-i-1)*G2)/n
                s = (data.iloc[i,0] + data.iloc[i+1,0])/2

                if G < giniVector[col]:
                    if data.iloc[0,0] == data.iloc[n-1,0]:
                        giniVector[col] = G
                        splitValue[col] = None
                    elif data.iloc[i,0] != data.iloc[i+1,0]:
                        giniVector[col] = G
                        splitValue[col] = s
    return giniVector,splitValue                             

                    
