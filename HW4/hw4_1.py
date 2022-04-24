import pandas as pd
import numpy as np
from scipy.special import gammaln
import scipy.stats as sps
import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.random.seed(2022)
df =pd.read_csv('data/mathtest.csv')#, header=None)

data= df[['school','mathscore']]
X = data.to_numpy()
Z=np.unique(X[:,0])
Ybar= np.array([np.mean(X[:,1][np.where(X[:,0]==Z[i])]) for i in range(Z.shape[0])])
Ni = np.array([X[:,1][np.where(X[:,0]==Z[i])].shape[0] for i in range(Z.shape[0])])
Y= X[:,1]
# plt.scatter(Ni,Ybar)
# plt.xlabel('$N_i$')
# plt.ylabel(r'$\bar{y}_i$')
# plt.show()

def sample_theta(Ni,Ybar,sigma2,tau2,mu):
    variances = 1./(1./(sigma2*tau2)+ Ni/sigma2)
    means = variances*(mu/(sigma2*tau2)+ Ni*Ybar/sigma2)
    return sps.multivariate_normal.rvs(mean=means.reshape(-1,), cov=np.diag(variances.reshape(-1,)),size=1).reshape(-1,)
def sample_sigma2(Ni,Y,theta,tau2,mu):
    a = (Ni.shape[0]+np.sum(Ni))/2
    b= np.sum((theta-mu)**2)/(2*tau2)+1./2
    for i in range(Ni.shape[0]):
        b+=np.sum( (Y[np.where(X[:,0]==Z[i])] - theta[i])**2)/2
    return sps.invgamma.rvs(a=a,scale=b,size=1)
def sample_tau2(Ni,sigma2,theta,mu):
    a =(Ni.shape[0]+1)/2
    b= np.sum( (theta-mu)**2)/(2*sigma2)
    return sps.invgamma.rvs(a=a, scale=b, size=1)
def sample_mu(Ni,theta,sigma2,tau2):
    P=Ni.shape[0]
    return sps.norm.rvs(loc=np.mean(theta),scale=np.sqrt(sigma2*tau2/P))

T=1000
thetas=[]
sigma2s=[]
tau2s=[]
mus=[]
thetas.append(Ybar)
sigma2s.append(np.var(Y))
mus.append(np.mean(thetas))
kappas=[]
for t in tqdm.tqdm(range(T)):
    tau2s.append(sample_tau2(Ni,sigma2s[-1],thetas[-1],mus[-1]))
    sigma2s.append(sample_sigma2(Ni,Y,thetas[-1],tau2s[-1],mus[-1]))
    thetas.append(sample_theta(Ni,Ybar,sigma2s[-1],tau2s[-1],mus[-1]))
    mus.append(sample_mu(Ni,thetas[-1],sigma2s[-1],tau2s[-1]))
    kappas.append(1./(1+Ni*tau2s[-1]))
kappa=np.mean(kappas,axis=0)
plt.scatter(Ni,kappa)
plt.xlabel('$N_i$')
plt.ylabel(r'$\bar{\kappa}_i$')
plt.show()


