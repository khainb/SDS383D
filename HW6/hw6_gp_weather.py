import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import tqdm
np.random.seed(2022)
df =pd.read_csv('data/weather.csv')
data=df[['pressure','temperature','lon','lat']].to_numpy()
X = data[:,[2,3]]
Y1 = data[:,[0]]
Y2 = data[:,[1]]



def Euclidean(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
def CSE(x1,x2,t1,t2,b,distance=Euclidean):
    if(np.sum(np.abs(x1-x2))==0):
        a=1
    else:
        a=0
    return t1**2 * np.exp(-0.5 * (distance(x1,x2)/b)**2)+t2**2*a

def m(x1):
    return 0


def mean_var_f(x,X,Y,m,C,mean,cov,t1,t2,b,distance=Euclidean):
    n=X.shape[0]
    d= X.shape[1]
    mean = mean.reshape(Y.shape)
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
def loglikelihood(X,y,m,C,t1,t2,b):
    _, cov = mean_cov(X, m, C, t1, t2, b)
    n= X.shape[0]
    y= y.reshape(-1,1)
    return -n/2*np.log(2*np.pi) - 1/2*np.log(np.linalg.det(cov+1*np.eye(n))) - 1/2*y.T.dot(np.linalg.inv(cov+1*np.eye(n))).dot(y)[0][0]
grid=10000
t1s= np.linspace(0.1,10000,grid)
t2=1e-6
bs =np.linspace(0.1,10000,grid)
lls =np.zeros((grid,grid))
for i in tqdm.tqdm(range(grid)):
    for j in range(grid):

        lls[i,j]=loglikelihood(X,Y1,m,CSE,t1s[i],t2,bs[j])
ind = np.argmax(lls)
print('t1star={}, bstar={}'.format(t1s[int(ind/grid)],bs[int(ind%grid)] ))



t1=t1s[int(ind/grid)]
b=bs[int(ind%grid)]
t2=0

mean,cov = mean_cov(X,m,CSE,t1,t2,b)

grid=100
lon_pred = np.linspace(np.min(X[:,0]), np.max(X[:,0]), grid)
lat_pred = np.linspace(np.min(X[:,1]), np.max(X[:,1]), grid)
x, y = np.meshgrid(lon_pred, lat_pred)
points = np.stack((x, y), axis=-1)
X_pred = points.reshape(-1,2)


Yhatvar1 = np.array([mean_var_f(X_pred[j],X,Y1,m,CSE ,mean,cov,t1,t2,b) for j in tqdm.tqdm(range(X_pred.shape[0]))])
Yhatvar2 = np.array([mean_var_f(X_pred[j],X,Y2,m,CSE ,mean,cov,t1,t2,b) for j in tqdm.tqdm(range(X_pred.shape[0]))])

plt.contourf(lon_pred, lat_pred, Yhatvar1[:,0].reshape(grid,grid),100, cmap='coolwarm')
plt.colorbar()
plt.title('Predicted Pressure')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

plt.contourf(lon_pred, lat_pred, Yhatvar1[:,1].reshape(grid,grid),100, cmap='coolwarm')
plt.colorbar()
plt.title('Variance Pressure')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


plt.contourf(lon_pred, lat_pred, Yhatvar2[:,0].reshape(grid,grid),100, cmap='coolwarm')
plt.colorbar()
plt.title('Predicted Temperature')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

plt.contourf(lon_pred, lat_pred, Yhatvar2[:,1].reshape(grid,grid),100, cmap='coolwarm')
plt.colorbar()
plt.title('Variance Temperature')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
plt.scatter(X[:,0], X[:,1],c=Y1, label='Data', cmap='coolwarm')
plt.colorbar()
plt.title('Pressure')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()


plt.scatter(X[:,0], X[:,1],c=Y2, label='Data', cmap='coolwarm')
plt.colorbar()
plt.title('Temperature')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
