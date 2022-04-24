import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
np.random.seed(2022)
df =pd.read_csv('data/utilities.csv')
data=df[['temp','gasbill','billingdays']].to_numpy()
X = data[:,0]
Y = data[:,1]/data[:,2]
def GaussianKernel(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2/2)

def s(x,h,j):
    return np.sum(GaussianKernel((x-X)/h) *( (X-x)**j ) )
def w(x,h):
    return GaussianKernel( (x-X)/h )*(s(x,h,2)-(X-x)*s(x,h,1))

def fhat(x,h):
    weight= w(x,h)
    return np.sum(weight*Y/np.sum(weight))
def f_smooth(x,h):
    weight =1./h*GaussianKernel((x-X)/h)
    weight=weight/np.sum(weight)
    return np.sum(weight*Y)
hs = np.linspace(0.1,10,100)
# hs = np.array([1,3,10,15,30])
loocvs=[]

for i in range(hs.shape[0]):
    h = hs[i]
    Yhat = [fhat(X[j],h) for j in range(X.shape[0])]
    # Yhat = [f_smooth(X[j], h) for j in range(X.shape[0])]
    # print(np.sum((Yhat-Y)**2))
    Hii=[]
    for j in range(X.shape[0]):
        weight = w(X[j], h)
        weight=weight/np.sum(weight)
        Hii.append(weight[j])
    Hii=np.array(Hii)
    loocvs.append( np.sum(((Y-Yhat)/( (1-Hii)+1e-6))**2))
Xlin = np.linspace(np.min(X),np.max(X),200)
Yhat = [fhat(Xlin[j],hs[np.argmin(loocvs)]) for j in range(Xlin.shape[0])]
print("Optimal h: {}".format(hs[np.argmin(loocvs)]))
plt.scatter(X,Y,color='black')
plt.plot(Xlin,Yhat)
plt.xlabel('Temp')
plt.ylabel('Daily Bill')
plt.show()
plt.close()
plt.clf()
Ypred = [fhat(X[j],hs[np.argmin(loocvs)]) for j in range(X.shape[0])]
residual = np.array(Ypred)-Y
plt.scatter(X,residual)
plt.xlabel('Temp')
plt.ylabel('Daily Bill Residual')
plt.show()
plt.close()
plt.clf()
H = np.zeros((X.shape[0],X.shape[0]))
for i in range(X.shape[0]):
    weight = w(X[i], hs[np.argmin(loocvs)])
    weight = weight / np.sum(weight)
    H[i]=weight
print(H)
sigmahat = np.sum(residual**2)/(X.shape[0]-2*np.trace(H)+np.trace(H.T.dot(H)))
print('Sigma2hat:{}'.format(sigmahat))
plt.scatter(X,Y,color='black')
plt.plot(Xlin,Yhat)
# print(H.dot(H.T))
# print(np.diag(H.dot(H.T)).shape)
# print(Ypred.shape)
inds = np.argsort(X)
plt.plot(X[inds],(Ypred-2*np.sqrt(sigmahat*np.diag(H.dot(H.T))))[inds],color='red',alpha=0.5)
plt.plot(X[inds],(Ypred+2*np.sqrt(sigmahat*np.diag(H.dot(H.T))))[inds],color='red',alpha=0.5)
plt.fill_between(X[inds],(Ypred-2*np.sqrt(sigmahat*np.diag(H.dot(H.T))))[inds],(Ypred+2*np.sqrt(sigmahat*np.diag(H.dot(H.T))))[inds],color='tab:blue',alpha=0.5)
# Ypred=np.array(Ypred)
# for i in range(X.shape[0]):
#     print(Ypred)
#     plt.fill_between(X[i],Ypred[i]-2*np.sqrt(sigmahat),Ypred[i]+2*np.sqrt(sigmahat))
plt.xlabel('Temp')
plt.ylabel('Daily Bill')
plt.show()
plt.close()
plt.clf()


