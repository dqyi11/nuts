'''
Created on Dec 6, 2016

@author: daqingy
'''

from hmc import HMC
from hmc import HMC_DA
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # step size
    eps = 0.3
    sample_num = 1000
    L = 20
    
    mu = np.zeros((2,1))
    #var = np.matrix([[1, .8],[.8, 1]])
    cov = np.asarray([[1, .8],[.8, 4]])

    # potential energy function
    def U_func(X):    
        tmp = - np.dot(X, np.linalg.inv(cov))
        U = 0.5 * np.dot( tmp, X.T )
        return U
    
    # gradient potential energy function
    def dU_func(X):
        grad = - np.dot(X, np.linalg.inv(cov))
        return grad
        
    x0 = np.array([0,6])
    
    X = HMC(x0, eps, L, sample_num, U_func, dU_func)
    
    delta = 0.2
    lam = 1.0
    X = HMC_DA(x0, delta, lam, sample_num, U_func, dU_func)
    
    s_mu = np.mean(X,0)
    s_cov = np.cov(X,rowvar=False)
    print s_mu
    print s_cov
    
    temp = np.random.multivariate_normal(mu[:,0], cov, size=sample_num)
    #print temp
           
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X[:,0],X[:,1],'b.')
    ax.plot(temp[:,0],temp[:,1], 'r+')
    #ax.plot(X[0,0:50],X[1,0:50],'r')
    #ax.set_xlim([-6,6])
    #ax.set_ylim([-6,6])
    
    
    plt.show()