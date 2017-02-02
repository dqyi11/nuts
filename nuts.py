'''
Created on Feb 2, 2017

@author: daqingy
'''

from hmc import *, LeapFrog

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
        
        n = 1
        s = 1
        while s == 1:
            v = 2*np.random.random() - 1.
            if v == -1:
                build_tree()
            else:
                build_tree()
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
            
delta_max = 1000            
            
def build_tree(x, q, log_u, v, j, eps, theta0, r0, U, K, dU, dK):
    if j = 0:
        # base case - take one leapfrog step in the direction v
        new_x, new_q = LeapFrog(x, q, v*eps, dU, dK )
        U0 = U(x)
        K0 = K(q)
        U1 = U(new_x)
        K1 = K(new_q)
        if log_u <= np.log(U1 + K1):
            new_n = 1
        else:
            new_n = 0
        if log_u < delta_max + U1 + K1:
            new_s = 1
        else:
            new_s = 0
        new_alfa = np.min([1, np.exp((U0+K0)-(U1+K1))])
        new_n_alfa = 1.0
        return new_x, new_q, new_x, new_q, new_x, new_n, new_s, new_alfa, new_n_alfa
    else:
        # recursion - implicitly build the left and right subtree
        neg_x, neg_q, pos_x, pos_q, new_x, new_n, new_s, new_alfa, new_n_alfa = build_tree( x, q, log_u, v, j-1, eps, theta0, r0, U, K, dU, dK )
        if new_s == 1:
            if v == -1:
                neg_x, neg_q, None, None, new_new_x, new_new_n, new_new_alfa, new_new_n_alfa = build_tree( neg_x, neg_q, log_u, v, j-1, eps, theta0, r0, U, K, dU, dK )    
            else:
                None, None, pos_x, pos_q, new_new_x, new_new_n, new_new_alfa, new_new_n_alfa = build_tree( pos_x, pos_q, log_u, v, j-1, eps, theta0, r0, U, K, dU, dK )  
           
            if np.random.random() < new_new_n / (new_n + new_new_n):
                new_x = new_new_x
            
            new_alfa += new_new_alfa
            new_n_alfa += new_new_n_alfa
            new_s = new_new_s * 
            new_n += new_new_n
        return neg_x, neg_q, pos_x, pos_q, new_pos, new_n, new_s, new_alfa, new_new_alfa