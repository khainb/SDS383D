import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
np.random.seed(2022)
def f(x):
    return x**3+ x**2+x+1#x*np.cos(x)#x**3+ x**2+x+1

X=np.linspace(-10,10,20)
Y=[f(X[i])+np.random.randn(1)[0]*200 for i in range(X.shape[0])]
plt.scatter(X,Y,label='data',color='black')

def f_smooth(x,X,Y,h):
    weight =1./h*1/np.sqrt(2*np.pi)*np.exp( -((X-x)/h)**2/2 )
    weight=weight/np.sum(weight)
    return np.sum(weight*Y)
x=np.linspace(-10,10,200)

y=[f(x[i]) for i in range(x.shape[0])]
plt.plot(x,y,label='$x^3+x^2+x+1$ ')
y0=[f_smooth(x[i],X,Y,0.1) for i in range(x.shape[0])]
plt.plot(x,y0,label='h=0.1')
y1=[f_smooth(x[i],X,Y,1) for i in range(x.shape[0])]
plt.plot(x,y1,label='h=1')
y2=[f_smooth(x[i],X,Y,2) for i in range(x.shape[0])]
plt.plot(x,y2,label='h=2')
y5=[f_smooth(x[i],X,Y,5) for i in range(x.shape[0])]
plt.plot(x,y5,label='h=5')
y10=[f_smooth(x[i],X,Y,10) for i in range(x.shape[0])]
plt.plot(x,y10,label='h=10')
# x=np.linspace(-10,10,100)
# y=[f(x[i]) for i in range(x.shape[0])]
plt.title('Gaussian smoothing: $y=x^3+x^2+x+1$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()