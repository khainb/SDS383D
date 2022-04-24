import pandas as pd
import numpy as np
from scipy.special import gammaln
import scipy.stats as sps
import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.random.seed(2022)
#Read data
df =pd.read_csv('data/greenbuildings.csv')#, header=None)
data= df[['Rent','leasing_rate','green_rating','City_Market_Rent','age','class_a','class_b']]
X = data.to_numpy()
X=X[np.where(X[:,1]!=0)[0]]
Y=X[:,[0]]*X[:,[1]]/100
X= np.concatenate([np.ones((X.shape[0],1)),X[:,2:]],axis=1)
#Priors
n=X.shape[0]
p=X.shape[1]
Lamda= np.eye(n)
K= np.eye(p)
m = np.ones((p,1))
d=4
eta=4
h=4
def sample_beta(omega,Lamda):
    Lamda=np.diag(Lamda)
    cov=np.linalg.inv(omega *X.T.dot(Lamda).dot(X) +omega*K)
    mean = cov.dot(omega*X.T.dot(Lamda).dot(Y)+omega *K.dot(m))
    new_beta = sps.multivariate_normal.rvs(mean=mean.reshape(-1,),cov=cov,size=1)
    return new_beta.reshape(-1,)
def sample_omega(beta,Lamda):
    beta=beta.reshape(-1,1)
    Lamda = np.diag(Lamda)
    lambdastar = X.T.dot(Lamda).dot(X) + K
    Ainv = np.linalg.inv(lambdastar)
    b = Y.T.dot(Lamda).dot(X) + m.T.dot(K)
    etastar = eta + Y.T.dot(Lamda).dot(Y) + m.T.dot(K).dot(m) - b.dot(Ainv).dot(b.T)
    new_omega = sps.gamma.rvs((n+d)/2,scale=1./(etastar/2),size=1)[0]
    return new_omega
def sample_Lamda(omega,beta):
    beta = beta.reshape(-1, 1)
    a= (h+1)/2
    b= omega* ((Y-X.dot(beta))**2).reshape(-1,)+h
    b=b/2
    lams= []
    for i in range(n):
        lams.append(sps.gamma.rvs(a,scale=1./b[i],size=1)[0])
    return np.array(lams).reshape(-1,)
T= 1000
#Initialize Paramters
betas=[np.random.randn(p,)]
omegas=[sps.gamma.rvs(d/2,scale=1./(eta/2),size=1)[0]]
Lamdas = [np.ones((n,)) * sps.gamma.rvs(h/2,scale=1./(h/2),size=1)[0]]
#Gibbs Sampling
for t in tqdm.tqdm(range(T)):
    betas.append(sample_beta(omegas[-1],Lamdas[-1]))
    omegas.append(sample_omega(betas[-1],Lamdas[-1]))
    Lamdas.append(sample_Lamda(omegas[-1],betas[-1]))


np.save('betas.npz',np.array(betas).reshape(-1,p)[500:])
np.save('omegas.npz',np.array(omegas).reshape(-1,)[500:])
np.save('Lamdas.npz',np.array(Lamdas).reshape(-1,n)[500:])
betas = np.load('betas.npz.npy')
omegas = np.load('omegas.npz.npy')
Lamdas = np.load('Lamdas.npz.npy')
#Plot 1/lambda_i
plt.scatter(Y,np.mean(1./Lamdas,axis=0))
plt.xlabel(r'$y_i$')
plt.ylabel(r'$\frac{1}{\lambda_i}$')
plt.show()
#Plot residuals
beta_mean = np.mean(betas,axis=0).reshape(-1,1)
resdual = Y-X.dot(beta_mean)
plt.hist(resdual,bins=100,label='Heteroskedastic')
plt.xlabel('Residual')
plt.ylabel('Histogram')
#
nustar=n+d
lambdastar=X.T.dot(Lamda).dot(X)+K
Ainv=np.linalg.inv(lambdastar)
b=X.T.dot(Lamda).dot(Y)+K.dot(m)
mustar=Ainv.dot(X.T.dot(Lamda).dot(Y)+K.dot(m))
etastar = eta + Y.T.dot(Lamda).dot(Y) + m.T.dot(K).dot(m) - b.T.dot(Ainv).dot(b)
Sigmastar = etastar/nustar*Ainv
resdual = Y-X.dot(mustar)
plt.hist(resdual,bins=100,label='Homoskedastic',alpha=0.8)
plt.legend()
plt.show()
#95% credible interval
import arviz as az
for i in range(p):
    print('Homo:{}'.format(i))
    print(az.hdi(sps.t.rvs(nustar, loc=mustar[i, 0], scale=Sigmastar[i, i],size=1000),hdi_prob=0.95))
for i in range(p):
    print('Hete:{}'.format(i))
    print(az.hdi(betas[:,i], hdi_prob=0.95))
#