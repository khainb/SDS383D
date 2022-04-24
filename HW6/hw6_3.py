import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
np.random.seed(2022)
def f1(x):
    return x*np.cos(x)#x**3+ x**2+x+1
def f2(x):
    return x**3 +x**2+x+1
def K(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2/2)
def f_smooth(x,X,Y,h):
    weight =1./h*K((X-x)/h)
    weight=weight/np.sum(weight)
    return np.sum(weight*Y)

def validate(Xtrain,Ytrain, Xtest,Ytest,hs):
    errors=[]
    for h in hs:
        preds = np.array([f_smooth(Xtest[i],Xtrain,Ytrain,h) for i in range(Xtest.shape[0])])
        errors.append(np.mean((np.array(Ytest)-preds)**2))
    return errors


X=np.linspace(-10,10,500)
inds =np.random.permutation(range(500))
Xtest=X[inds[400:]]
X=X[inds[:400]]

Ylow_smooth=[f2(X[i])+np.random.randn(1)[0]*2 for i in range(X.shape[0])]
Yhigh_smooth=[f2(X[i])+np.random.randn(1)[0]*200 for i in range(X.shape[0])]
Ylow_wig=[f1(X[i])+np.random.randn(1)[0]*2   for i in range(X.shape[0])]
Yhigh_wig=[f1(X[i])+np.random.randn(1)[0]*10 for i in range(X.shape[0])]

Ytestsmooth=[f2(Xtest[i]) for i in range(Xtest.shape[0])]
Ytestwig=[f1(Xtest[i]) for i in range(Xtest.shape[0])]

hs = [0.1,1,2,5,10]
print(validate(X,Ylow_smooth,Xtest,Ytestsmooth,hs))
print(validate(X,Yhigh_smooth,Xtest,Ytestsmooth,hs))
print(validate(X,Ylow_wig,Xtest,Ytestwig,hs))
print(validate(X,Yhigh_wig,Xtest,Ytestwig,hs))

