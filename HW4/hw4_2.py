import pandas as pd
import numpy as np
from scipy.special import gammaln
import scipy.stats as sps
import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
np.random.seed(2022)
df =pd.read_csv('data/bloodpressure.csv')#, header=None)
df = df[["date","systolic","subject","treatment"]]
data = df.to_numpy()

N1 = data[:,1][np.where(data[:,3]==1)].shape[0]
N2 = data[:,1][np.where(data[:,3]==2)].shape[0]
mean_1 = np.mean(data[:,1][np.where(data[:,3]==1)])
mean_2 = np.mean(data[:,1][np.where(data[:,3]==2)])
var_1 = np.var(data[:,1][np.where(data[:,3]==1)],ddof=1)
var_2 = np.var(data[:,1][np.where(data[:,3]==2)],ddof=1)

t_value = (mean_1-mean_2)/np.sqrt((var_1/N1+var_2/N2))
dfre= N1+N2-2
# print(sps.ttest_ind(data[:,2][np.where(data[:,3]==1)],data[:,2][np.where(data[:,3]==2)]))
# print(dfre)
# print(sps.t.ppf(0.975,df=dfre))
print(t_value)
Z=np.unique(data[:,2])
for i in range(Z.shape[0]):
    fig = tsaplots.plot_acf(data[:,1][np.where(data[:,2]==Z[i])])
    plt.xlabel('Subject {}'.format(Z[i]))
    plt.show()

    plt.clf()
    plt.close()





Ybar= np.array([np.mean(data[:,1][np.where(data[:,2]==Z[i])]) for i in range(Z.shape[0])])
Ni = np.array([data[:,1][np.where(data[:,2]==Z[i])].shape[0] for i in range(Z.shape[0])])
Y= data[:,1]
X = np.zeros((Ni.shape[0],))
treatment= np.array([np.mean(data[:,3][np.where(data[:,2]==Z[i])]) for i in range(Z.shape[0])])
X[np.where(treatment==2)]=1

N1 = Ybar[np.where(treatment==1)].shape[0]
N2 = Ybar[np.where(treatment==2)].shape[0]
mean_1 = np.mean(Ybar[np.where(treatment==1)])
mean_2 = np.mean(Ybar[np.where(treatment==2)])
var_1 = np.var(Ybar[np.where(treatment==1)],ddof=1)
var_2 = np.var(Ybar[np.where(treatment==2)],ddof=1)

t_value =(mean_1-mean_2)/np.sqrt((var_1/N1+var_2/N2))
print(t_value)

# plt.scatter(Ni,Ybar)
# plt.xlabel('$N_i$')
# plt.ylabel(r'$\bar{y}_i$')
# plt.show()

# def sample_theta(Ni,Ybar,sigma2,tau2,mu,beta,X):
#     variances = 1./(1./(sigma2*tau2)+ Ni/sigma2)
#     means = variances*((mu+beta*X)/(sigma2*tau2)+ Ni*Ybar/sigma2)
#     return sps.multivariate_normal.rvs(mean=means.reshape(-1,), cov=np.diag(variances.reshape(-1,)),size=1).reshape(-1,)
# def sample_sigma2(Ni,Y,theta,tau2,mu,beta,X):
#     a = (Ni.shape[0]+np.sum(Ni))/2
#     b= np.sum((theta-mu-X*beta)**2)/(2*tau2)+1./2
#     for i in range(Ni.shape[0]):
#         b+=np.sum( (Y[np.where(data[:,2]==Z[i])] - theta[i])**2)/2
#     return sps.invgamma.rvs(a=a,scale=b,size=1)[0]
# def sample_tau2(Ni,sigma2,theta,mu,beta,X):
#     a =(Ni.shape[0]+1)/2
#     b= np.sum( (theta-mu-X*beta)**2)/(2*sigma2)
#     # print('Start')
#     # print(mu)
#     # print(theta)
#     # print(X*beta)
#     return sps.invgamma.rvs(a=a, scale=b, size=1)[0]
# def sample_mu(Ni,theta,sigma2,tau2,beta,X):
#     P=Ni.shape[0]
#     return sps.norm.rvs(loc=np.mean(theta-X*beta),scale=np.sqrt(sigma2*tau2/P))
# def sample_beta(Ni,theta,sigma2,tau2,mu,X):
#     return sps.norm.rvs(loc=np.sum( (theta-mu)*X )/np.sum(X**2),scale=np.sqrt( (sigma2*tau2)/np.sum(X**2)))
# T=1000
# thetas=[]
# sigma2s=[]
# tau2s=[]
# mus=[]
# betas=[]
# thetas.append(Ybar)
# sigma2s.append(np.var(Y))
# mus.append(np.mean(thetas))
# betas.append(  (Ybar[1]-Ybar[0])/2)
# kappas=[]
#
# for t in tqdm.tqdm(range(T)):
#     tau2s.append(sample_tau2(Ni,sigma2s[-1],thetas[-1],mus[-1],betas[-1],X))
#     sigma2s.append(sample_sigma2(Ni,Y,thetas[-1],tau2s[-1],mus[-1],betas[-1],X))
#     thetas.append(sample_theta(Ni,Ybar,sigma2s[-1],tau2s[-1],mus[-1],betas[-1],X))
#     mus.append(sample_mu(Ni,thetas[-1],sigma2s[-1],tau2s[-1],betas[-1],X))
#     betas.append(sample_beta(Ni, thetas[-1], sigma2s[-1], tau2s[-1], mus[-1], X))
# print(np.mean(betas[600:]))
# print(np.sqrt(np.var(betas[600:],ddof=1) ))
# plt.hist(betas[600:],bins=30)
# plt.xlabel(r'$\beta$')
# plt.ylabel(r'Histogram')
# plt.show()
# plt.clf()
# plt.close()


# for i in range(Ni.shape[0]):
#
#     human1= data[np.where(data[:,2]==Z[i])]
#     if (i < 10):
#         plt.scatter(human1[:,0],human1[:,1],label='Subject '+str(Z[i]))
#     else:
#         plt.scatter(human1[:, 0], human1[:, 1], label='Subject ' + str(Z[i]),marker='P')
# plt.grid(True)
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Systolic')
# plt.show()





