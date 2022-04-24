import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
np.random.seed(2022)

def Euclidean(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
def CSE(x1,x2,t1,t2,b,distance=Euclidean):
    if(x1==x2):
        a=1
    else:
        a=0
    return t1**2 * np.exp(-0.5 * (distance(x1,x2)/b)**2)+t2**2*a

def CM52(x1,x2,t1,t2,b,distance=Euclidean):
    if (x1 == x2):
        a = 1
    else:
        a = 0
    d=distance(x1,x2)
    return t1**2*(1+np.sqrt(5)*d/b+5*d**2/(3*b**2))*np.exp(-np.sqrt(5)*d/b)+t2**2*a
def m(x1):
    return 0

def GP(N,m,C,X,t1,t2,b,distance=Euclidean):
    n = X.shape[0]
    mean = np.zeros((n,))
    cov =np.zeros((n,n))
    for i in range(n):
        mean[i] = m(X[i])
        for j in range(n):
            cov[i,j]=C(X[i],X[j],t1,t2,b,distance)
    return sps.multivariate_normal.rvs(mean=mean,cov=cov,size=N)

X = np.linspace(0,1,100)

t1=1
t2=10
b=1e-1
N=10

f,(ax1,ax2) = plt.subplots(1, 2, figsize=(12, 5))

Ys1 = GP(10,m,CSE,X,t1,t2,b)
for i in range(N):
    ax1.plot(X,Ys1[i])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title(r'CSE $\tau_1={}$, $\tau_2={}$, $b={}$'.format(t1,t2,b))
ax1.grid(True)
# ax1.legend()



Ys2 = GP(10,m,CM52,X,t1,t2,b)
for i in range(N):
    ax2.plot(X,Ys2[i])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title(r'CM52 $\tau_1={}$, $\tau_2={}$, $b={}$'.format(t1,t2,b))
ax2.grid(True)
# ax2.legend()
plt.show()




