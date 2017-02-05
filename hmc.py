'''
Created on Feb 2, 2017

@author: daqingy
'''

import numpy as np

def LeapFrog(x, q, eps, dU):

    new_q = q + 0.5 * eps * dU( x ) 
    new_x = x + 0.5 * eps * new_q
    new_q = new_q + 0.5 * eps * dU( new_x )
    
    return new_x, new_q

def HMC(x0, eps, L, M, U, dU):
    
    dim = len(x0)
    X = np.zeros((M+1, dim))
    X[0, :] = x0
    for m in range(M):
        q = np.random.randn(dim)
        x = X[m, :]
        H0 = U(x) - 0.5 *  np.dot(q, q.T)
        for i in range(L):
            x, q = LeapFrog(x, q, eps, dU)
        H1 = U(x) - 0.5 *  np.dot(q, q.T) 
            
        u = np.random.rand()
        if np.log(u) <= np.min([0.0, H1-H0]):
            X[m+1,:] = x
        else:
            X[m+1,:] = X[m,:]       
    
    return X

def HMC_DA(x0, delta, lam,  M, U, dU, M_adapt=None):
    
    if M_adapt == None:
        M_adapt = int(M/2)
        
    eps0 = FindReasonableEpsilon(x0, U, dU)
    dim = len(x0)
    X = np.zeros((M+1, dim))
    X[0, :] = x0
    Eps = np.zeros(M+1)
    Eps[0] = eps0
    
    mu = np.log(10*eps0)
    eps_b = 1.0
    H_b = 0
    gamma = 0.05
    t0 = 10
    k = 0.75
    
    for m in range(M):
        q = np.random.randn(dim)
        x = X[m,:]
        H0 = U(x) - 0.5 *  np.dot(q, q.T)
        
        L = np.max([1, int( lam / Eps[m] )])
        for i in range(L):
            x, q = LeapFrog(x, q, Eps[m], dU)
        H1 = U(x) - 0.5 *  np.dot(q, q.T)
            
        u = np.random.rand()
        if np.log(u) <= np.min([0.0, H1-H0]):
            X[m+1,:] = x
        else:
            X[m+1,:] = X[m,:]       
    
        if m <= M_adapt:
            alfa = np.exp(H1-H0)
            eta = 1./float(m+t0)
            H_b = (1 - eta) * H_b + eta * (delta - alfa)
            Eps[m+1] = np.exp( mu - np.sqrt(m) / gamma * H_b )
            eta = (m+1)**(-k) # m start from 0
            eps_b = np.exp( eta * np.log( Eps[m+1] ) + (1. - eta) * np.log( eps_b ) )
        else:
            Eps[m+1] = eps_b 
            
    return X
    
def FindReasonableEpsilon(x, U, dU):
    eps = 1.0
    dim = len(x)
    q = np.random.randn(dim)
    
    log_prob = U(x) - 0.5 * np.dot(q, q.T)   
    new_x, new_q = LeapFrog(x, q, eps, dU) 
    new_log_prob = U(new_x) - 0.5 * np.dot(new_q, new_q.T)
    a = 2. * float( new_log_prob - log_prob > np.log(0.5) ) - 1
    
    while a * (new_log_prob - log_prob) > - a * np.log(2):
        eps = 2.0**2 * eps
        new_x, new_q = LeapFrog(x, q, eps, dU) 
        log_prob = U(x) - 0.5 * np.dot(q, q.T)
        new_log_prob = U(new_x) - 0.5 * np.dot(new_q, new_q.T)
        x , q = new_x, new_q
    return eps    