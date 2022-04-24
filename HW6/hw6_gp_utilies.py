import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
np.random.seed(2022)
df =pd.read_csv('data/utilities.csv')
data=df[['temp','gasbill','billingdays']].to_numpy()
X = data[:,0]
Y = data[:,1]/data[:,2]
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


def mean_var_f(x,X,Y,m,C,mean,cov,t1,t2,b,distance=Euclidean):
    n=X.shape[0]
    covtilde = np.zeros((n,))
    for i in range(n):
        covtilde[i] = C(x, X[i], t1, t2, b, distance)
    covtilde= covtilde.reshape(-1,1)
    Cinv = np.linalg.pinv(cov)
    pred = m(x)+ covtilde.T.dot(Cinv).dot( (Y-mean).reshape(-1,1))
    var = C(x, x, t1, t2, b, distance) -covtilde.T.dot(Cinv).dot(covtilde)
    return np.array([pred[0][0],var[0][0]])

def mean_cov(X,m,C,t1,t2,b,distance=Euclidean):
    n = X.shape[0]
    mean = np.zeros((n,))
    cov = np.zeros((n, n))
    for i in range(n):
        mean[i] = m(X[i])
        for j in range(n):
            cov[i, j] = C(X[i], X[j], t1, t2, b, distance)
    return mean,cov+0.61*np.eye(n)

t1=10
t2=1e-6
b=10

mean,cov = mean_cov(X,m,CSE,t1,t2,b)
# Xlin = np.linspace(np.min(X),np.max(X),100)
inds = np.argsort(X)
Xlin = X[inds]
Yhatvar = np.array([mean_var_f(Xlin[j],X,Y,m,CSE ,mean,cov,t1,t2,b) for j in range(Xlin.shape[0])])
# print(Yhatvar[:,1])
plt.scatter(X,Y,color='black')
plt.plot(Xlin,Yhatvar[:,0])
plt.plot(Xlin,Yhatvar[:,0]-2*np.sqrt(Yhatvar[:,1]),color='red',alpha=0.5)
plt.plot(Xlin,Yhatvar[:,0]+2*np.sqrt(Yhatvar[:,1]),color='red',alpha=0.5)
plt.fill_between(Xlin,Yhatvar[:,0]-2*np.sqrt(Yhatvar[:,1]),Yhatvar[:,0]+2*np.sqrt(Yhatvar[:,1]),color='tab:blue',alpha=0.5)
plt.title(r'CSE $\tau_1={}$, $\tau_2={}$, $b={}$'.format(t1,t2,b))
plt.xlabel('Temp')
plt.ylabel('Daily Bill')
plt.show()
plt.close()
plt.clf()



