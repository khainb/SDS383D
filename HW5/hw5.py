import numpy as np
import pandas as pd
import scipy.stats as sps
import tqdm
import matplotlib.pyplot as plt

#Read data
np.random.seed(2022)
df =pd.read_csv('data/cheese.csv')#, header=None)
stores= df['store'].tolist()
stores_names= [store.split('-')[1][1:] for store in stores ]
data=df[['price','vol','disp']].to_numpy()
Y=np.log(data[:,[1]])
X=np.log(data[:,[0]])
disp=data[:,2]

#Count unique stores
u_store_names, s_counts= np.unique(stores,return_counts=True)
store_ints=[]
for store in stores:
    for j in range(len(u_store_names)):
        if store ==u_store_names[j]:
            store_ints.append(j)

#Save indices of stores
store_ints= np.array(store_ints)
u_store_int= np.unique(store_ints)

#Cat data into a mtrix
X =np.concatenate([np.ones((X.shape[0],1)),X,disp.reshape(-1,1),disp.reshape(-1,1)*X],axis=1)
n= u_store_int.shape[0]
d = X.shape[1]

#Sample beta for all stores
def sample_betai(sigma,s,mubeta):
    mubeta=mubeta.reshape(-1,1)
    sigma = sigma.reshape(-1,)
    Sigma = np.diag(s.reshape(-1,))
    betais=[]
    for i in range(n):
        inds = np.where(store_ints==i)
        Xi = X[inds]
        Yi = Y[inds]
        var = np.linalg.inv(Xi.T.dot(Xi)/sigma[i] + np.linalg.inv(Sigma))
        mean =var.dot(Xi.T.dot(Yi)/sigma[i] + np.linalg.inv(Sigma).dot(mubeta))
        betai = sps.multivariate_normal.rvs(mean=mean.reshape(-1,),cov=var,size=1)
        betais.append(betai.reshape(-1,))
    return np.array(betais)

#Sample mu beta for all stores
def sample_beta(s,betai):
    Sigma= np.diag(s.reshape(-1,))
    mean = np.mean(betai,axis=0)
    beta= sps.multivariate_normal.rvs(mean=mean.reshape(-1,), cov=Sigma/n,size=1)
    return beta.reshape(-1,)

#Sample sigma for all stores
def sample_sigmai(betai,a=1/2,b=1/2):
    sigma=[]
    for i in range(n):
        inds = np.where(store_ints == i)
        Xi = X[inds]
        Yi = Y[inds]
        Ni = Xi.shape[0]
        anew= a/2+ Ni/2
        bnew= b/2+ np.sum( (Yi-Xi.dot(betai[i].reshape(-1,1)))**2)/2
        sigma.append(sps.invgamma.rvs(anew,scale=bnew,size=1)[0])
    return np.array(sigma).reshape(-1,)
#Sample for variances
def sample_s(betai,mubeta):
    d = betai.shape[1]
    tau=[]
    for i in range(d):
        anew = 1/2+n/2
        bnew = 1/2 + np.sum((betai[:,i]-mubeta[i])**2)/2
        tau.append(sps.invgamma.rvs(anew,scale=bnew,size=1)[0])
    return np.array(tau).reshape(-1, )

#MCMC for a and b
def sample_ab(sigma,olda,oldb):
    anew = np.random.uniform(1,10,size=1)[0]
    bnew = np.random.uniform(1,10,size=1)[0]
    alpha= sps.gamma.pdf(anew,a=3,scale=1)*sps.gamma.pdf(bnew,a=3,scale=1)*np.prod(sps.invgamma.pdf(sigma,a=anew/2,scale=bnew/2)) \
           / (sps.gamma.pdf(oldb,a=3,scale=1)*sps.gamma.pdf(olda,a=3,scale=1)*np.prod(sps.invgamma.pdf(sigma,a=olda/2,scale=oldb/2)))
    if(np.random.uniform(0,1,size=1)<alpha):
        newa= anew
        newb=bnew
    else:
        newa=olda
        newb=oldb
    return newa,newb

#Initialization
betai0=np.array([ np.mean(X[np.where(u_store_int == i)],axis=0)  for i in range(n)] )
betas=[betai0]
mubetas=[np.mean(betai0,axis=0)]
taus=[]
sigmas=[]
lista=[1/2]
listb=[1/2]

#The Gibbs sampling
def gibbs_sampling(T=100):
    for t in tqdm.tqdm(range(T)):
        taus.append(sample_s(betas[-1], mubetas[-1]))
        sigmas.append(sample_sigmai(betas[-1],lista[-1],listb[-1]))
        betas.append(sample_betai(sigmas[-1],taus[-1],mubetas[-1]))
        mubetas.append(sample_beta(taus[-1],betas[-1]))
        a,b = sample_ab(sigmas[-1],lista[-1],listb[-1])
        lista.append(a)
        listb.append(b)

#10000 iterations and 5000 thinned
gibbs_sampling(10000)
mubetas= np.array(mubetas[5000:])
sigmas= np.array(sigmas[5000:])
betas = np.array(betas[5000:])

#Plotting things
plt.hist(mubetas[:,0],bins=100)
plt.xlabel(r'$\mu_{\beta_1}$')
plt.ylabel('Histogram')
plt.show()
plt.clf()

plt.hist(mubetas[:,1],bins=100)
plt.xlabel(r'$\mu_{\beta_2}$')
plt.ylabel('Histogram')
plt.show()
plt.clf()

plt.hist(mubetas[:,2],bins=100)
plt.xlabel(r'$\mu_{\beta_3}$')
plt.ylabel('Histogram')
plt.show()
plt.clf()

plt.hist(mubetas[:,3],bins=100)
plt.xlabel(r'$\mu_{\beta_4}$')
plt.ylabel('Histogram')
plt.show()
plt.clf()


beta= np.mean(betas,axis=0)
plt.scatter(range(n),beta[:,0])
plt.xlabel('Store')
plt.ylabel(r'$\beta_1$')
plt.show()
plt.clf()

plt.scatter(range(n),beta[:,1])
plt.xlabel('Store')
plt.ylabel(r'$\beta_2$')
plt.show()
plt.clf()

plt.scatter(range(n),beta[:,2])
plt.xlabel('Store')
plt.ylabel(r'$\beta_3$')
plt.show()
plt.clf()

plt.scatter(range(n),beta[:,3])
plt.xlabel('Store')
plt.ylabel(r'$\beta_4$')
plt.show()
plt.clf()


meansig= np.mean(sigmas,axis=0)
plt.scatter(range(meansig.shape[0]),meansig)
plt.xlabel('Store')
plt.ylabel(r'$\sigma_i^2$')
plt.show()