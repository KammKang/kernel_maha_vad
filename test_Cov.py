import python_speech_features 
import scipy.io.wavfile as  wav
import matplotlib.pyplot as plt
import numpy as np
def Maha_distance(y1,y2,C1,C2):
    
   
    a =(y1 - y2 )
    #print("a:",a)
    
    #print("inv:",np.linalg.pinv(C1)+np.linalg.pinv(C2))
    ret = np.dot(np.dot(np.transpose(y1-y2),np.linalg.pinv(C1)+np.linalg.pinv(C2)),(y1-y2))
    #ret = np.dot(np.transpose(y1-y2),(y1-y2))
    return ret
                                                                      

    

def Covariance( y,i,R=7):
    """
    y: length* mfcc np.array
    R: default =7
    i : 第几帧
    
    """
    (row,) = y[i].shape
    _sum = np.zeros_like(y[i])
    u = np.zeros_like(y[i])
    #print(row)
    for j in range(i-R,i+R):
         _sum += y[j]
    
    u = _sum/(2*R+1)
    C = np.zeros((row,row),dtype = float)
    for j in range(i-R,i+R):
        C += np.dot((y[j]-u),np.transpose(y[j]-u)) #  注意这里的 shape * 东西。
    C = C/(1+2*R)
    return C 
    
