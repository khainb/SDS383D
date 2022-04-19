import numpy as np
import pandas as pd
import scipy.stats as sps
from polyagamma import random_polyagamma
import tqdm
np.random.seed(2022)

#Read data
df =pd.read_csv('data/polls.csv')#, header=None)
data=df[['state','edu','age','female','black','weight']]
Y=df[['bush']].to_numpy()

#Remove NaN
inds = ~np.isnan(Y)
Y = Y[inds].astype(np.float32)
data = data.loc[inds]
data = data.to_numpy()
states=data[:,0]
u_states= np.unique(data[:,0])
n=u_states.shape[0]
u_edu = np.unique(data[:,1] )
u_age = np.unique(data[:,2] )
X_edu = np.zeros((data.shape[0],u_edu.shape[0]))
X_age = np.zeros((data.shape[0],u_age.shape[0]))

#Change categorical data into indicators
for i in range(data.shape[0]):
    X_edu[i,np.where(u_edu==data[i,1])[0][0]] =1
    X_age[i, np.where(u_age == data[i, 2])[0][0]] = 1
X = np.concatenate([np.ones((data.shape[0],1)),X_edu,X_age,data[:,3:]],axis=1)
# X = np.concatenate([np.ones((data.shape[0],1)),data[:,3:]],axis=1)
X[:,-1] = X[:,-1]/np.max(X[:,-1])
X= X.astype(np.float32)
p=X.shape[1]

#Sigmoid function
def sigmoid(a):
    return 1/(1+np.exp(-a))

#Sample omega for store i
def sample_omegai(Xi,betai):
    Ni=Xi.shape[0]
    betai=betai.reshape(-1,1)
    etai = Xi.dot(betai)
    etai = etai.reshape(-1,)
    nis=np.ones(Ni).astype(np.int32)
    return random_polyagamma(nis,etai,size=(1,Ni))

#Sample beta for store i
def sample_beta_i(yi,wi,Xi,Sigma,mubetai):
    Omegai= np.diag(wi.reshape(-1,))
    # print(Xi.T.dot(Omegai).dot(Xi))
    V= np.linalg.inv(Xi.T.dot(Omegai).dot(Xi)+np.linalg.inv(Sigma))
    kappa = (yi-np.ones(yi.shape)/2).reshape(-1,1)
    mubetai=mubetai.reshape(-1,1)
    m= V.dot(Xi.T.dot(kappa)+ np.linalg.inv(Sigma).dot(mubetai))
    return sps.multivariate_normal.rvs(mean=m.reshape(-1,),cov=V,size=1).reshape(-1,)

#Sample mu_beta
def sample_mubeta(Sigmas,beta):
    mu=[]
    for i in range(n):
        Sigmainv = np.linalg.inv(Sigmas[i])
        betai= beta[i]
        var = np.linalg.inv(Sigmainv + np.linalg.inv( np.eye(p)))
        mean = var.dot(Sigmainv.dot(betai.reshape(-1, 1)))
        mu.append( sps.multivariate_normal.rvs(mean=mean.reshape(-1, ), cov=var, size=1).reshape(-1, ))
    return np.array(mu)

#Sample Sigmas for all stores
def sample_Sigma(beta,mubeta):
    beta.reshape(-1, p)
    mubeta = mubeta.reshape(-1, p)
    Sigmas = []
    for i in range(n):
        inds = np.where(states == u_states[i])
        Xi = X[inds]
        Ni=Xi.shape[0]
        a = Ni + p + 1
        betai =beta[i]
        mubetai=mubeta[i]
        vec = (betai - mubetai).reshape(-1,p)
        b = np.eye(p) + vec.T.dot(vec)
        Sigmas.append(sps.invwishart.rvs(df=a, scale=b, size=1).reshape(p, p))
    return np.array(Sigmas)
#Sample omegas for all stores
def sample_Omega(beta):
    Omega=[]
    for i in range(n):
        inds = np.where(states==u_states[i])
        Xi= X[inds]
        omegai = sample_omegai(Xi,beta[i])
        Omega.append(omegai)
    return Omega
#Sample beta for all stores
def sample_betas(Omega,Sigmas,mubeta):
    beta=[]
    for i in range(n):
        inds = np.where(states == u_states[i])
        Xi = X[inds]
        Yi = Y[inds]
        betai=sample_beta_i(Yi,Omega[i],Xi,Sigmas[i],mubeta[i])
        beta.append(betai)
    return np.array(beta)

#Initialization
betao=[]
for i in range(n):
    inds = np.where(states == u_states[i])
    Xi = X[inds]
    betao.append(np.mean(Xi,axis=0))
mubetas=[np.random.randn(n,p)]
Omegas=[]
betas=[np.random.randn(n,p)]
Sigmas=[]

#The Gibbs sampler
def gibbs_sampling(T=1000):
    for t in tqdm.tqdm(range(T)):
        # from sklearn.metrics import accuracy_score
        # from sklearn.linear_model import LogisticRegression
        # accs = []
        # for i in range(n):
        #     inds = np.where(states == u_states[i])
        #     Xi = X[inds]
        #     Yi = Y[inds]
        #     betai = betas[-1][i].reshape(-1, 1)
        #     pred = sigmoid(Xi.dot(betai))
        #     Ypred = (pred >= 0.5).astype(np.int32)
        #     accs.append(accuracy_score(Ypred, Yi))
        # print(np.mean(accs))
        Sigmas.append(sample_Sigma(betas[-1], mubetas[-1]))
        Omegas.append(sample_Omega(betas[-1]))
        betas.append(sample_betas(Omegas[-1],Sigmas[-1],mubetas[-1]))
        mubetas.append(sample_mubeta(Sigmas[-1],betas[-1]))

#4000 iterations and 2000 thinned
gibbs_sampling(4000)
beta = np.mean(betas[2000:],axis=0)

#Plotiing things
import matplotlib.pyplot as plt
plt.bar(u_states,beta[:,0])
plt.xlabel('State')
plt.ylabel('Intercept')
plt.show()
plt.clf()

plt.bar(u_states,beta[:,1])
plt.xlabel('State')
plt.ylabel(r'$\beta_1$'+' ('+u_edu[0]+')')
plt.show()
plt.clf()

plt.bar(u_states,beta[:,2])
plt.xlabel('State')
plt.ylabel(r'$\beta_2$'+' ('+u_edu[1]+')')
plt.show()
plt.clf()

plt.bar(u_states,beta[:,3])
plt.xlabel('State')
plt.ylabel(r'$\beta_3$'+' ('+u_edu[2]+')')
plt.show()
plt.clf()
plt.bar(u_states,beta[:,4])
plt.xlabel('State')
plt.ylabel(r'$\beta_4$'+' ('+u_edu[3]+')')
plt.show()
plt.clf()
plt.bar(u_states,beta[:,5])
plt.xlabel('State')
plt.ylabel(r'$\beta_5$'+' ('+u_age[0]+')')
plt.show()
plt.clf()


plt.bar(u_states,beta[:,6])
plt.xlabel('State')
plt.ylabel(r'$\beta_6$'+' ('+u_age[1]+')')
plt.show()
plt.clf()
plt.bar(u_states,beta[:,7])
plt.xlabel('State')
plt.ylabel(r'$\beta_7$'+' ('+u_age[2]+')')
plt.show()
plt.clf()


plt.bar(u_states,beta[:,8])
plt.xlabel('State')
plt.ylabel(r'$\beta_8$'+' ('+u_age[3]+')')
plt.show()
plt.clf()


plt.bar(u_states,beta[:,9])
plt.xlabel('State')
plt.ylabel(r'$\beta_9$ (female)')
plt.show()
plt.clf()


plt.bar(u_states,beta[:,10])
plt.xlabel('State')
plt.ylabel(r'$\beta_{10}$ (black)')
plt.show()
plt.clf()

plt.bar(u_states,beta[:,11])
plt.xlabel('State')
plt.ylabel(r'$\beta_{11}$ (weight)')
plt.show()
plt.clf()
from sklearn.metrics import accuracy_score
# for i in range(n):
#     inds = np.where(states == u_states[i])
#     Xi = X[inds]
#     Yi = Y[inds]
#     betai=beta[i].reshape(-1,1)
#     pred = sigmoid(Xi.dot(betai))
#     Ypred= (pred>0.5).astype(np.int32)
#     print(accuracy_score(Ypred,Yi))