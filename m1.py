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
    sample_num = 500
    L = 20
    
    mu = np.zeros((2,1))
    var = np.matrix([[1, .8],[.8, 1]])
    M = np.matrix([[1,0],[0,1]])

    # potential energy function
    def U_func(X):    
        U = X.T * np.linalg.inv(var) * X / 2
        return np.sum(U)
    
    # gradient potential energy function
    def dU_func(X):
        A = np.linalg.inv(var) * X
        return A
        
    # kinetic energy function    
    def K_func(P):
        s = np.sum( P.T * M * P / 2 )
        return s
    
    # gradient kinetic energy function
    def dK_func(P):
        return M * P
    
    x0 = np.matrix([[0],[6]])
    
    X = HMC(x0, eps, L, sample_num, U_func, K_func, dU_func, dK_func)
    X = HMC_DA(x0, eps, lam, sample_num, U, K, dU, dK)
    
    s_mu = np.mean(X,1)
    s_var = np.cov(X)
    
    print s_mu
    print s_var
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[0,:],X[1,:])
    #ax.plot(X[0,0:50],X[1,0:50],'r')
    #ax.set_xlim([-6,6])
    #ax.set_ylim([-6,6])
    
    plt.show()