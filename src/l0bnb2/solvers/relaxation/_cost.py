import numpy as np
from numba import njit

from ..utils import get_ratio_threshold, diag_index


@njit(cache=True)
def get_primal_cost(Y, Theta, R, l0, l2, M, ratio, threshold, zlb, zub, active_set):
    n,p = Y.shape
    cost = 0
    zlb_act = np.zeros((0,))
    zub_act = np.zeros((0,))
    abs_theta = np.zeros((0,))
    for i in range(p):
        cost -= np.log(Theta[i,i])
        cost += R[:,i]@R[:,i]/Theta[i,i]
    
    if len(active_set) == 0:
        return cost, np.zeros_like(zlb)
    
    for i,j in active_set:
        if zub[i,j] > 0:
            zlb_act = np.append(zlb_act, zlb[i,j])
            zub_act = np.append(zub_act, zub[i,j])
            abs_theta = np.append(abs_theta, abs(Theta[i,j]))
    s_act = np.where(zlb_act==1, abs_theta**2, \
                     abs_theta*M if ratio > M else np.maximum(abs_theta*ratio, abs_theta**2))
    
    z_act = abs_theta/M
    
    if l2 > 0:
        z_act[z_act>0] = np.maximum(z_act[z_act>0], abs_theta[z_act>0]**2/s_act[z_act>0])
    z_act = np.minimum(np.maximum(zlb_act, z_act), zub_act)
    
    z = np.zeros((p,p))
    k = 0
    for i,j in active_set:
        if zub[i,j] > 0:
            z[i,j] = z[j,i] = z_act[k]
            k += 1
    return cost+l0*np.sum(z_act[z_act>0]) + l2*np.sum(s_act[s_act>0]), z


@njit(cache=True)
def get_dual_cost(Y, Theta, R, NuY, l0, l2, M, ratio, threshold, zlb, zub, active_set):
    p = Y.shape[1]
    res = p
    tmp = 0.
    for i in range(p):
        tmp = -np.linalg.norm(R[:,i]/Theta[i,i], 2)**2 - NuY[i,i]
        if tmp <= 0:
            return -np.inf
        res += np.log(tmp)
    
    a = 2*M*l2 if l2 != 0 else 0
    c = a if ratio <= M else (l0/M+l2*M)
    pen = 0.
    cur_nuy = 0.
    for i,j in active_set:
        cur_nuy = abs(NuY[i,j])
        if zub[i,j] == 0:
            continue
        elif zlb[i,j] == 1:
            if l2 == 0:
                pen += (M*cur_nuy-l0)
            elif cur_nuy <= a:
                pen += (cur_nuy**2/(4*l2)-l0)
            else:
                pen += (M*cur_nuy-l0-l2*M**2)
        else:
            if cur_nuy <= threshold:
                continue
            elif cur_nuy <= c:
                pen += (cur_nuy**2/(4*l2)-l0)
            else:
                pen += (M*cur_nuy-l0-l2*M**2)
    return res - pen

