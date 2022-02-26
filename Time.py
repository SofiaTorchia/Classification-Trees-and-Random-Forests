import time

def TicTocGenerator():
    ti = 0          
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti

TicToc = TicTocGenerator()

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        return tempTimeInterval

def tic():
    toc(False)
 

