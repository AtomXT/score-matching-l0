import numpy as np
from numba import njit
from numba.typed import List

from ..utils import compute_relative_gap
from ..oracle import Q_psi, Q_phi, R_nl
from ._cost import get_primal_cost

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@njit(cache=True)
def cd_loop(Y, Theta, l0, l2, M, ratio, threshold, S_diag, zlb, zub,
            active_set, R, ordering = 'coordinate'):
    p = Y.shape[1]
    for i,j in active_set:
        if zub[i,j] == 0:
            R[:,i] = R[:,i] - Theta[i,j]*Y[:,j]
            R[:,j] = R[:,j] - Theta[i,j]*Y[:,i]
            Theta[i,j] = Theta[j,i] = 0
        elif zlb[i,j] == 0:
            theta_old = Theta[i,j]
            Theta[i,j] = Theta[j,i] = Q_psi(S_diag[j]/Theta[i,i]+S_diag[i]/Theta[j,j], \
                                            (2*Y[:,j]@R[:,i]-2*Theta[i,j]*S_diag[j])/Theta[i,i]+(2*Y[:,i]@R[:,j]-2*Theta[i,j]*S_diag[i])/Theta[j,j], l0, l2, M, ratio, threshold, suff=True)
            R[:,i] = R[:,i] + (Theta[i,j]-theta_old)*Y[:,j]
            R[:,j] = R[:,j] + (Theta[i,j]-theta_old)*Y[:,i]
        elif zlb[i,j] > 0:
            theta_old = Theta[i,j]
            Theta[i,j] = Theta[j,i] = Q_phi(S_diag[j]/Theta[i,i]+S_diag[i]/Theta[j,j], \
                                            (2*Y[:,j]@R[:,i]-2*Theta[i,j]*S_diag[j])/Theta[i,i]+(2*Y[:,i]@R[:,j]-2*Theta[i,j]*S_diag[i])/Theta[j,j], 1, l2, M)
            R[:,i] = R[:,i] + (Theta[i,j]-theta_old)*Y[:,j]
            R[:,j] = R[:,j] + (Theta[i,j]-theta_old)*Y[:,i]
    
    for i in range(p):
        R[:,i] = R[:,i] - Theta[i,i]*Y[:,i]
        Theta[i,i] = R_nl(S_diag[i], R[:,i]@R[:,i])
        R[:,i] = R[:,i] + Theta[i,i]*Y[:,i]
    
    return Theta, R


@njit(cache=True)
def cd(Y, Theta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, R, ordering, rel_tol=1e-8, maxiter=3000, verbose=False):
    tol = 1
    old_cost = cost
    curiter = 0
    while tol > rel_tol and curiter < maxiter:
        old_cost = cost
        Theta, R = cd_loop(Y, Theta, l0, l2, M, ratio, threshold, S_diag, zlb, zub,
            active_set, R, ordering)
        cost, _ = get_primal_cost(Y, Theta, R, l0, l2, M, ratio, threshold, zlb, zub, active_set)
        if verbose:
            print(cost)
        tol = abs(compute_relative_gap(old_cost, cost))
        curiter += 1
    return Theta, cost, R

