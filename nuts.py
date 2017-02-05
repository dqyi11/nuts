'''
Created on Feb 2, 2017

@author: daqingy
'''

from hmc import *

def NUTS(x0, delta, M, U, dU, M_adapt=None):
    
    if M_adapt == None:
        M_adapt = int(M/2)
        
    eps0 = FindReasonableEpsilon(x0, dU)
    dim = len(x0)
    X = np.zeros((dim, M+M_adapt))
    Eps = np.zeros(M+M_adapt)
    X[0,:] = x0
    Eps[0] = eps0
    
    mu = np.log(10.*eps0)
    eps_b = 1.0
    H_b = 0
    gamma = 0.05
    t0 = 10
    k = 0.75
    
    for m in range(1, M+M_adapt):
        # Resample momenta
        q = np.random.randn(dim)
        x = X[:, m-1]
        
        log_u = float( U(x) - 0.5* np.dot(q, q.T) + np.random.exponential(1, size=1) )
        
        neg_x = x[:]
        pos_x = x[:] 
        neg_q = q[:] 
        pos_q = q[:]
        X[m,:] = x
        
        j = 0
        n = 1 # Initially the only valid point is the initial point
        s = 1
        while s == 1:
            # choose a direction. -1 = backwards, 1 = forwards
            v = int(2 * (np.random.random() < 0.5) - 1)
            # Double the size of the tree
            if v == -1:
                neg_x, neg_q, None, None, new_x, new_n, new_s, alfa, new_alfa = build_tree(neg_x, neg_q, log_u, v, j, Eps[m], X[:,m])
            else:
                None, None, pos_x, pos_q, new_x, new_n, new_s, alfa, new_alfa = build_tree(pos_x, pos_q, log_u, v, j, Eps[m], X[:,m])
            
            # Use Metropolis-Hastings to decide whether or not to move to a point from the half-tree we just generated
            _tmp = np.min( [1. , float(new_n)/float(n)] )
            if new_s == 1 and np.random.rand() < _tmp:
                X[m,:] = new_x
                
            # Update number of valid points we've seen            
            n += new_n
            s = new_s * (np.dot(pos_x - neg_x, neg_q) >= 0) * (np.dot(pos_x - neg_x, pos_q) >=0)
            j += 1
    
        # Do adaptation of epsilon if we're still doing burn-in
        if m <= M_adapt:
            eta = 1. / float(m+t0)
            H_b = (1. - eta) * H_b + eta * (delta - alfa / float(new_alfa) )
            Eps[m] = np.exp( mu - np.sqrt(m) / gamma * H_b )
            eta = m**(-k)
            eps_b = np.exp( eta * np.log( Eps[m] ) + (1. - eta) * np.log( eps_b ) )
        else:
            Eps[m] = eps_b 
            
        return X[M_adapt:,:], Eps[M_adapt:]
    
def stop_criterion(pos_x, neg_x, pos_q, neg_q):
    # Compute the stop condition in the main loop
    dx = pos_x - neg_x
    return (np.dot(dx, neg_q.T) >= 0) & (np.dot(dx, pos_q.T) >= 0)      
            
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
        neg_x, neg_q, pos_x, pos_q, new_x, new_n, new_s, new_alfa, new_n_alfa = build_tree( x, q, log_u, v, j-1, eps, theta0, r0, U, dU )
        
        # no need to keep going if the stopping criteria were met in the first subtree
        if new_s == 1:
            if v == -1:
                neg_x, neg_q, _, _, new_new_x, new_new_n, new_new_s, new_new_alfa, new_new_n_alfa = build_tree( neg_x, neg_q, log_u, v, j-1, eps, theta0, r0, U, dU)    
            else:
                _, _, pos_x, pos_q, new_new_x, new_new_n, new_new_s, new_new_alfa, new_new_n_alfa = build_tree( pos_x, pos_q, log_u, v, j-1, eps, theta0, r0, U, dU)  
            
            # choose which subtree to propagate a sample up from
            if np.random.random() < float(new_new_n) / (new_n + new_new_n):
                new_x = new_new_x
            
            new_alfa += new_new_alfa
            new_n_alfa += new_new_n_alfa
            new_s = new_new_s * (np.dot( pos_x - neg_x , neg_q.T ) >= 0 ) * (np.dot(pos_x-neg_x, pos_q.T) >=0)
            new_n += new_new_n
        return neg_x, neg_q, pos_x, pos_q, new_x, new_n, new_s, new_alfa, new_new_alfa