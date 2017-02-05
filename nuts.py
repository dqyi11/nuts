'''
Created on Feb 2, 2017

@author: daqingy
'''

from hmc import *

def NUTS(x0, delta, M, M_adapt=None, U, K, dU, dK):
    
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
        log_u = U(x) + K(q) + np.log(np.random.random())
        
        U0 = U(x)
        K0 = K(q)
        
        neg_x = x
        pos_x = x 
        neg_q = q 
        pos_q = q 
        j = 0
        X[:,m+1] = x
        
        n = 1
        s = 1
        while s == 1:
            v = 2*np.random.random() - 1.
            if v == -1:
                neg_x, neg_q, None, None, new_x, new_n, new_s, alfa, n_alfa = build_tree(neg_x, neg_q, log_u, v, j, Eps[m], X[:,m])
            else:
                None, None, pos_x, pos_q, new_x, new_n, new_s, alfa, n_alfa = build_tree(pos_x, pos_q, log_u, v, j, Eps[m], X[:,m])
            
            if new_s == 1:
                if np.random.rand() < np.min([1.0, new_n/n]):
                    X[:,m+1] = new_x
            
            n += new_n
            s = new_s * (np.dot(pos_x - neg_x, neg_q) >= 0) * (np.dot(pos_x - neg_x, pos_q) >=0)
            j += 1
    
        if m <= M_adapt:
            H_b = (1 - 1/(m+t0)) * H_b + (1/(m+t0)) * (delta - alfa)
            Eps[m+1] = np.exp( mu - np.sqrt(m) / gamma * H_b )
            eps_b = np.exp( m**(-k) * np.log( Eps[m+1] ) + (1 - m**(-k)) * np.log( eps_b ) )
        else:
            Eps[m+1] = eps_b 
            
        return X, Eps
            
delta_max = 1000            
            
def build_tree(x, q, log_u, v, j, eps, theta0, r0, U, dU):
    
    if j == 0:
        # base case - take one leapfrog step in the direction v
        H0 = U(x) - 0.5 * np.dot(q, q.T)
        new_x, new_q = LeapFrog(x, q, v*eps, dU )
        H1 = U(new_x) - 0.5 * np.dot(new_q, new_q.T)
        # Is the new point in the slice?
        new_n = int(log_u <= H1)    
        # Is the simulation widly inaccurate?
        new_s = int((log_u - delta_max) < H1)
        
        # Compute the acceptance probability
        new_alfa = np.min([1, np.exp(H1-H0)])
        new_n_alfa = 1.0
        return new_x, new_q, new_x, new_q, new_x, new_n, new_s, new_alfa, new_n_alfa
    else:
        # recursion - implicitly build the left and right subtree
        neg_x, neg_q, pos_x, pos_q, new_x, new_n, new_s, new_alfa, new_n_alfa = build_tree( x, q, log_u, v, j-1, eps, theta0, r0, U, K, dU, dK )
        
        # no need to keep going if the stopping criteria were met in the first subtree
        if new_s == 1:
            if v == -1:
                neg_x, neg_q, _, _, new_new_x, new_new_n, new_new_s, new_new_alfa, new_new_n_alfa = build_tree( neg_x, neg_q, log_u, v, j-1, eps, theta0, r0, U, K, dU, dK )    
            else:
                None, None, pos_x, pos_q, new_new_x, new_new_n, new_new_s, new_new_alfa, new_new_n_alfa = build_tree( pos_x, pos_q, log_u, v, j-1, eps, theta0, r0, U, K, dU, dK )  
           
            if np.random.random() < new_new_n / (new_n + new_new_n):
                new_x = new_new_x
            
            new_alfa += new_new_alfa
            new_n_alfa += new_new_n_alfa
            new_s = new_new_s * (np.dot( pos_x - neg_x , neg_q ) >= 0 ) * (np.dot(pos_x-neg_x, pos_q) >=0)
            new_n += new_new_n
        return neg_x, neg_q, pos_x, pos_q, new_x, new_n, new_s, new_alfa, new_new_alfa