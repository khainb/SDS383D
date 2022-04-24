import pandas as pd
import numpy as np
from scipy.special import gammaln
np.random.seed(2022)
#Read data
data =pd.read_csv('data/wdbc.csv', header=None).iloc[:,1:12]
data[1]=data[1].replace({'M': 0, 'B': 1})
df = data.to_numpy()
X = df[:,1:]
X = (X-np.mean(X,axis=0,keepdims=True))/(np.sqrt(np.var(X,axis=0,keepdims=True)))
X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
Y = df[:,0].astype(np.int32).reshape(-1,1)
w= np.ones((X.shape[0],1))
#Gradient
def gradient(X,w,y,mu,phi):
    return X.T.dot((np.diag(w.reshape(-1,))/phi).dot(y-mu))
#Sigmoid function
def cal_p(X,beta):
    return np.exp(X.dot(beta))/(1+np.exp(X.dot(beta)))
#Initialize Beta
beta0 = np.ones((11,1))
#Loglikelihood function
def loglikelihood(beta):
    xtbeta = X.dot(beta)
    return np.sum(Y * np.log(np.exp(xtbeta) / (1 + np.exp(xtbeta))) + (1 - Y) * np.log(1. / (1 + np.exp(xtbeta))))
#list for store loglikelihood
list_ll=[loglikelihood(beta0)]
#Initialze Learning Rate
lr=1e-2
i=0
#Gradient descent
while(True):
    beta=beta0
    p = cal_p(X,beta )
    g = gradient(X,w,Y,p,phi=1)
    beta += lr*g
    list_ll.append(loglikelihood(beta))
    print(list_ll[-1])
    if(len(list_ll)>=2):
        if(np.abs((list_ll[-1]-list_ll[-2])/list_ll[-2])<=1e-8):
            break
print(beta)
#Check with sklearn
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0,fit_intercept=False,penalty='none').fit(X, Y.reshape(-1,))
print(clf.coef_)

#Plot log-likleihood per Iterations
import matplotlib.pyplot as plt
plt.plot(range(len(list_ll)),list_ll,label='Gradient Descent')
plt.axhline(y = loglikelihood(clf.coef_.reshape(beta.shape)), color = 'r', linestyle = '-',label='Sklearn')
plt.xlabel(r'Iterations')
plt.ylabel('Log-Likelihood')
plt.ylim([-300,0])
plt.legend()
plt.show()
