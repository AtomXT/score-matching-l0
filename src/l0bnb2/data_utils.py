import numpy as np
import scipy.stats as st
from scipy.sparse import diags



"""
Theta = B + delta*I
delta chosen to have condition(Theta)=cond
Each coordinate of B is set to zero or 0.5 with probability 1-p0 and p0, respectively.
"""

def generate_uniform(n,p, p0, cond, normalize=True,rng=None):
    
    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
    B = 0.5*rng.binomial(1,p0,(p,p))
    B = B*np.tri(p,p,-1)
    np.fill_diagonal(B, 0)
    B = (B + np.transpose(B))
    evals = np.linalg.eigvalsh(B)
    delta = (cond*evals[0]-evals[p-1])/(1-cond)
    Theta = B + delta*np.identity(p)
    Sigma = np.linalg.inv(Theta)

    if normalize:
        q = np.max(np.diag(Sigma))
        Theta = Theta*q
    Sigma = np.linalg.inv(Theta)

        
    X = rng.multivariate_normal(np.zeros(p), Sigma, n)
    
    return X, Sigma, Theta
 
def generate_banded_Toeplitz_precision(n,p,half_bandwidth=1,rho=0.5,cond=2,normalize=True, rng=None):

    if rng is None:
        rng = np.random
    elif type(rng) == int:
        rng = np.random.RandomState(rng)
    
    if rho == 0 or cond == 1:
        Sigma = np.eye(p)
        Theta = np.eye(p)
        X = rng.randn(n,p)
        return X, Sigma, Theta
    
    Theta = np.abs(np.arange(p)-np.arange(p)[:,None])
    Theta = np.where(Theta <= half_bandwidth, np.power(float(rho), Theta), 0)
    np.fill_diagonal(Theta,0)
    evals = np.linalg.eigvalsh(Theta)
    delta = (cond*evals[0]-evals[-1])/(1-cond)
    np.fill_diagonal(Theta,delta)
    Sigma = np.linalg.inv(Theta)

    if normalize:
        q = np.max(np.diag(Sigma))
        Theta = Theta*q
    Sigma = np.linalg.inv(Theta)
    
    Sigma = np.linalg.inv(Theta)
    X = rng.randn(n,p)@np.linalg.cholesky(Sigma).T
    

    return X, Sigma, Theta

def generate_synthetic(n,p,model="uniform",normalize=True, rng=None, **kwargs):
    """
    Generate synthetic data set
    Parameters
    ----------
    model: str
        "banded_Toeplitz_precision" or "uniform"
        "uniform": Uniform random sparsity
        "banded_Toeplitz_precision": samples with partial correlation rho^{|i-j|}, where 'rho' is given by kwargs
    normalize: bool, default True
        whether to normalize the precision matrix to have maximum var = 1
    rng: None, int, or random generator
        If rng is None, then it becomes np.random
        If rng is int, then it becomes np.random.RandomState(rng)
    Returns
    -------
    X:  n x p numpy array
        simulated data
    Sigma: p x p numpy array
        population covariance matrix of sampled data
    Theta: p x p numpy array
        population precision matrix of sampled data
    """
    
    assert model in {"banded_Toeplitz_precision", "uniform"}
    if model == "uniform":
        p0 = kwargs.get("p0", 0.2)
        cond = kwargs.get("cond", 2)
        return generate_uniform(n,p, p0, cond, normalize,rng )
    elif model == "banded_Toeplitz_precision":
        cond = kwargs.get("cond",2)
        half_bandwidth = kwargs.get("half_bandwidth",5)
        rho = kwargs.get("rho", 0.5)
        return generate_banded_Toeplitz_precision(n,p,half_bandwidth,rho,cond,normalize,rng)




def preprocess(X, assume_centered=False, cholesky=False):
    if assume_centered:
        X_mean = np.zeros(X.shape[1])
    else:
        X_mean = np.mean(X, axis=0)
        X = X - X_mean


    n, p = X.shape
    if cholesky and n > p:
        S = X.T@X/n
        S_diag = np.diag(S)
        Y = np.linalg.cholesky(S).T
    else:
        Y = X/np.sqrt(n)
        S_diag = np.linalg.norm(Y, axis=0) ** 2
    return n,p,X,X_mean,Y,S_diag


"""
Centering and normalizing the training, validation and test data 
"""


def preprocess2(X, X_val, X_test, assume_centered=False):
    if assume_centered:
        X_mean = np.zeros(X.shape[1])
    else:
        X_mean = np.mean(X, axis=0)
        Y = X - X_mean
        Y_val = X_val - X_mean
        Y_test = X_test - X_mean

    n, _ = Y.shape
    Y = Y/np.sqrt(n)
    n, _ = Y_val.shape
    Y_val = Y_val/np.sqrt(n)
    n, _ = Y_test.shape
    Y_test = Y_test/np.sqrt(n)
    

    return X,X_mean,Y,Y_val, Y_test