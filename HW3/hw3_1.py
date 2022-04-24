import pandas as pd
import numpy as np
from scipy.special import gammaln
import scipy.stats as sps
import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
np.random.seed(2022)
#Red data
df =pd.read_csv('data/greenbuildings.csv')#, header=None)
data= df[['Rent','leasing_rate','green_rating','City_Market_Rent','age','class_a','class_b']]
X = data.to_numpy()
X=X[np.where(X[:,1]!=0)]
Y=X[:,[0]]*X[:,[1]]/100
# Add intercept
X= np.concatenate([np.ones((X.shape[0],1)),X[:,2:]],axis=1)

#Priors
n=X.shape[0]
p=X.shape[1]
Lamda= np.eye(n)
K= np.eye(p)
m = np.ones((p,1))
d=4
eta=4

#Compute paramter of posterior
nustar=n+d
lambdastar=X.T.dot(Lamda).dot(X)+K
Ainv=np.linalg.inv(lambdastar)
b=X.T.dot(Lamda).dot(Y)+K.dot(m)
mustar=Ainv.dot(X.T.dot(Lamda).dot(Y)+K.dot(m))
etastar = eta + Y.T.dot(Lamda).dot(Y) + m.T.dot(K).dot(m) - b.T.dot(Ainv).dot(b)
Sigmastar = etastar/nustar*Ainv
#95% interval
import arviz as az
print(az.hdi(sps.t.rvs(nustar, loc=mustar[1, 0], scale=Sigmastar[1, 1],size=1000),hdi_prob=0.95))
print(mustar)

#Traditional 95%
import statsmodels.api as sm
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary(alpha=0.05)
print(print_model)
#Plot residual
resdual = Y-X.dot(mustar)
plt.hist(resdual,bins=100)
plt.xlabel('Residual')
plt.ylabel('Histogram')
plt.show()