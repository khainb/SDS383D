import numpy as np
import time
import scipy
from scipy.linalg import solve_triangular
warm = np.random.randn(10,10)#warm up
#Vectors to store time
timesinv = np.zeros((4,4))
timesdec = np.zeros((4,4))
for i,d in enumerate([5,10,100,1000]):
    for j,N in enumerate([100,500,1000,5000]):
        #Sample data
        X=np.random.randn(N,d)
        truebeta=np.random.randn(d,1)
        truebeta=truebeta/np.sum(truebeta)
        Y = X.dot(truebeta)+np.random.randn(N,1)
        #Inverse method
        start=time.time()
        XTX= X.T.dot(X)
        beta = np.linalg.inv(XTX).dot(X.T).dot(Y)
        count=time.time()-start
        print('Inv={}'.format(count))
        timesinv[i,j]=count
        # Decomposition method
        start=time.time()
        P,L,U = scipy.linalg.lu(XTX)
        Z = solve_triangular(L,X.T.dot(Y),lower=True)
        beta = solve_triangular(U,Z)
        count = time.time() - start
        print('Dec={}'.format(count))
        timesdec[i, j] = count
print(timesinv)
print(timesdec)


