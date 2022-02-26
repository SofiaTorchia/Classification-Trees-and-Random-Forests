from graphviz import Digraph

class Node:
    def __init__(self,imp,splitVal,splitCol,left=None,right=None,majClass=None):
        self._left = left
        self._right = right
        self._imp = imp
        self._splitVal = splitVal
        self._splitCol = splitCol
        self._majClass = majClass
    
    def getValues(self):
        return [self._imp,self._splitVal,self._splitCol]
    
    def getRight(self):
        return self._right

    def getMajClass(self):
        return self._majClass
    
    def getLeft(self):
        return self._left

    def getSplitVal(self):
        return self._splitVal

    def plotTree(self,dot):
        tempNode = self
        if(tempNode.getSplitVal() == None):
            h = hex(id(tempNode))
            m = tempNode.getMajClass()
            dot.node(h,f'Output class: {m}')
            return h
        else:
            [i,val,col] = tempNode.getValues()
            pr = f'If x({col}) <= {val} turn right.\nGini impurity: {round(i,3)}'
            h = hex(id(tempNode))

            tempNodeL = tempNode.getLeft()
            tempNodeR = tempNode.getRight()
            
            R = tempNodeR.plotTree(dot)
            L = tempNodeL.plotTree(dot)

            dot.node(h,pr)
            dot.edge(h,R)
            dot.edge(h,L)
            return h
            
            
