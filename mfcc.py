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
    
    for j in range(i-R,i+R+1):
        C +=np.dot(np.transpose(np.matrix(y[j]-u)),np.matrix(y[j]-u)) #  注意这里的 shape * 东西。
    C = C/(1+2*R)
    return C 
    
def K_(y1,y2,C1,C2):
    
    return np.exp(-Maha_distance(y1,y2,C1,C2))

def K_buile(y,C,R):
    """
    y:  lentg* 
    
    """
    K = np.zeros([2*R+1,2*R+1])
    for i in range(2*R+1):
        for j in range(2*R+1):
            K[i][j] = K_(y[i],y[j],C[i],C[j])
           
            
    return K

(rate,sig) = wav.read("./slice.wav")

mfcc = 13

mfcc_ = python_speech_features.mfcc(sig,rate)
print(mfcc_.shape)
R = 7
N = 2*R+1
 # 不知道带不带 1， 2 阶

C = np.zeros((15,mfcc,mfcc),dtype = float)

for i in range(R,3*R+1):
    C[i-R] = Covariance(mfcc_,i,R)
print("*************************C")
print(C)
(mfcc_numb,mfcc) = mfcc_.shape
print("MFCC")
for i in range (0,mfcc_numb):
    print(mfcc_[i])
out = [n for n in range(1,300)]
# 考虑下  每个 N做一次。
a=0;
print(mfcc_numb)

for i in range(R,mfcc_numb-3*R,15):
    
    #shuff  一下 C
    
    K = K_buile(mfcc_[i:],C,R)
    print("********************K*********")
    #print(K)
    D  = np.zeros([2*R+1,2*R+1]) # 全 1 ？
    _K = np.sum(K,0)
    print("***********_K****************")
    #print(_K)
    for j in range(2*R+1):
        D[j][j] = _K[j]
    print("***********D*****************")
    #print(D)
    M = np.dot( np.linalg.inv(D) ,K)
    print("***********M*****************")
    #print(M)
    #print("d:", np.linalg.inv(D))
    #print("K;",K)
    #print("M:",M)
    (w,v) = np.linalg.eig(M)

    for k in range(0,15):
        out[a+k]= v[k][1]
        print(a,out[a+k])  
    
    a+=15
    for c in range(0,15):
        C[c] = Covariance(mfcc_,i+R+c,R)
        #print(C[c])
        
    #print("v0: ",v)
    
plt.plot(out)
plt.ylabel("value")
plt.show()





















