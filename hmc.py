'''
Created on Feb 2, 2017

@author: daqingy
'''

import numpy as np

def LeapFrog(x, q, eps, dU, dK):

    new_q = q - (eps/2) * dU( x ) 
    new_x = x + (eps/2) * dK( new_q )
    new_q = new_q - (eps/2) * dK( new_x )
    
    return new_x, new_q

def HMC(x0, eps, L, M, U, K, dU, dK):
    
    dim = len(x0)
    X = np.zeros((dim, M+1))
    X[:,0] = x0
    for m in range(M):
        q = np.random.randn(dim,1)
        x = X[:, m]
        U0 = U(x)
        K0 = K(q)
        for i in range(L):
            x, q = LeapFrog(x, q, eps, dU, dK)
        U1 = U(x)
        K1 = K(q)  
            
        alfa = np.min([1.0, np.exp((U0+K0)-(U1+K1))])
        u = np.random.rand()
        if u <= alfa:
            X[:,m+1] = x
        else:
            X[:,m+1] = X[:,m]       
    
    return X

def HMC_DA(x0, delta, lam,  M, M_adapt=None, U, K, dU, dK):
    
    if M_adapt == None:
        M_adapt = int(M/2)
        
    eps0 = FindReasonableEpsilon(x0, dU, dK)
    dim = len(x0)
    X = np.zeros((dim, M+1))
    Eps = np.zeros(M+1)
    X[:,0] = x0
    Eps[0] = eps0
    
    mu = np.log(10*eps0)
    eps_b = 1.0
    H_b = 0
    gamma = 0.05
    t0 = 10
    k = 0.75
    
    for m in range(M):
        q = np.random.randn(dim,1)
        x = X[:, m]
        U0 = U(x)
        K0 = K(q)
        L = np.max([1, int( lam / Eps[m] )])
        for i in range(L):
            x, q = LeapFrog(x, q, Eps[m], dU, dK)
        U1 = U(x)
        K1 = K(q)  
            
        alfa = np.min([1.0, np.exp((U0+K0)-(U1+K1))])
        u = np.random.rand()
        if u <= alfa:
            X[:,m+1] = x
        else:
            X[:,m+1] = X[:,m]       
    
        if m <= M_adapt:
            H_b = (1 - 1/(m+t0)) * H_b + (1/(m+t0)) * (delta - alfa)
            Eps[m+1] = np.exp( mu - np.sqrt(m) / gamma * H_b )
            eps_b = np.exp( m**(-k) * np.log( Eps[m+1] ) + (1 - m**(-k)) * np.log( eps_b ) )
        else:
            Eps[m+1] = eps_b 
        
    return X
    
def FindReasonableEpsilon(x, dU, dK):
    eps = 1.0
    dim = len(x)
    q = np.random.randn(dim,1)
    new_x, new_q = LeapFrog(x, q, eps, dU, dK) 
    log_prob = 0.0
    new_log_prob = 0.0
    if new_log_prob - log_prob > np.log(0.5):
        a = 1
    else:
        a = -1
    while a * (new_log_prob - log_prob) > - a * np.log(2):
        eps = 2.0**2 * eps
        new_x, new_q = LeapFrog(x, q, eps, dU, dK) 
        log_prob = 0.0
        new_log_prob = 0.0
        x , q = new_x, new_q
    return eps    