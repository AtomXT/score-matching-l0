import copy
from time import time
from collections import namedtuple

import numpy as np
from numba.typed import List
from numba import njit

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from ._coordinate_descent import cd_loop, cd
from ._cost import get_primal_cost, get_dual_cost
from ..utils import get_ratio_threshold, compute_relative_gap, support_to_active_set, trivial_soln


EPSILON = np.finfo('float').eps

def is_integral(z, tol):
    casted_sol = (z+0.5).astype(int)
    sol_diff = z - casted_sol
    max_ind = np.argmax(np.abs(sol_diff))
    if abs(sol_diff.flat[max_ind]) > tol:
        return True
    return False



def _initial_active_set(Y, Theta, zlb, zub):
    p = Y.shape[1]
    corr = np.corrcoef(Y.T)
    argpart = np.argpartition(-np.abs(corr), int(0.2*p), axis=1)[:,:int(0.2*p)]
    active_set = set()
    for i in range(p):
        for j in argpart[i]:
            if i!=j:
                active_set.add((i,j) if i < j else (j,i))

    argwhere = np.argwhere(zlb==1)
    for i,j in argwhere:
        if i != j:
            active_set.add((i,j) if i < j else (j,i))

    argwhere = np.argwhere(np.abs(Theta)>EPSILON*1e10)
    for i,j in argwhere:
        if i<j:
            active_set.add((i,j))

    active_set = np.array(sorted(active_set))
    return active_set



@njit(cache=True)
def _refined_initial_active_set(Y, Theta, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, support, R, ordering='coordinate'):
    support.clear()
    num_of_similar_supports = 0
    delta = 0
    while num_of_similar_supports < 3:
        delta = 0
        Theta, R = cd_loop(Y, Theta, l0, l2, M, ratio, threshold, S_diag, zlb, zub,\
                                active_set, R, ordering)
        for i,j in active_set:
            if (Theta[i,j]!=0)  and (i<j) and ((i,j) not in support):
                support.add((i,j))
                delta += 1
        if delta == 0:
            num_of_similar_supports += 1
        else:
            num_of_similar_supports = 0
    return support, Theta, R


def _initialize_active_set_algo(Y, l0, l2, M, ratio, threshold, S_diag, fixed_ones, fixed_zeros, warm_start, ordering):
    p = Y.shape[1]
    fixed_ones = support_to_active_set(fixed_ones)
    fixed_zeros = support_to_active_set(fixed_zeros)
    zlb = np.zeros((p,p))
    zlb[fixed_ones[:,0], fixed_ones[:,1]] = 1
    zlb[fixed_ones[:,1], fixed_ones[:,0]] = 1
    zub = np.ones((p,p))
    zub[fixed_zeros[:,0], fixed_zeros[:,1]] = 0
    zub[fixed_zeros[:,1], fixed_zeros[:,0]] = 0
    if S_diag is None:
        S_diag = np.linalg.norm(Y, axis=0)**2
    if warm_start is not None:
        support, Theta = warm_start['support'], np.copy(warm_start['Theta'])
        R = warm_start.get('R', Y@Theta)
        active_set = support_to_active_set(support)
    else:
        Theta, R = trivial_soln(Y, S_diag)
        active_set = _initial_active_set(Y, Theta, zlb, zub)
        support = {(0,0)}
        support, Theta, R = _refined_initial_active_set(Y, Theta, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, support, R, ordering)
        
    return Theta, R, support, zub, zlb, S_diag



@njit(cache=True, parallel=True)
def _above_threshold_indices(zub, R, Y, Theta, threshold):
    Nu = -2*R/np.diag(Theta)
    NuY = Y.T@Nu
    NuY = NuY + NuY.T
    np.fill_diagonal(NuY, np.diag(NuY)/2)
    above_threshold = np.argwhere(zub*np.abs(NuY)-threshold>0)
    return above_threshold, NuY

### zlb is fixed_ones, zub is fixed_zeros

def solve(Y, l0, l2, M, fixed_ones, fixed_zeros, 
          ratio=None, threshold=None, S_diag=None, warm_start=None, 
          rel_tol=1e-4, ordering='coordinate', tree_upper_bound=None, mio_gap=0,
              check_if_integral=True, cd_max_itr=100, kkt_max_itr=100, verbose =False):
    st = time()
    _sol_str = 'primal_value dual_value support Theta sol_time z R'
    Solution = namedtuple('Solution', _sol_str)
    
    if ratio is None or threshold is None:
        ratio, threshold = get_ratio_threshold(l0, l2, M)
    
    Theta, R, support, zub, zlb, S_diag = \
        _initialize_active_set_algo(Y, l0, l2, M, ratio, threshold, S_diag, \
                                    fixed_ones, fixed_zeros, warm_start, ordering)
    
    active_set = support_to_active_set(support)
    
    cost, _ = get_primal_cost(Y, Theta, R, l0, l2, M, ratio, threshold, zlb, zub, active_set)
    cd_tol = rel_tol / 2
    if verbose:
        print("cost", cost)
    curiter = 0
    while curiter < kkt_max_itr:
        Theta, cost, R = cd(Y, Theta, cost, l0, l2, M, ratio, threshold, S_diag, zlb, zub, active_set, R, ordering, cd_tol, cd_max_itr, verbose)
        if verbose:
            print("iter", curiter+1)
            print("cost", cost)
        
        above_threshold, NuY = _above_threshold_indices(zub, R, Y, Theta, threshold)
        
        outliers = [(i,j) for i,j in above_threshold if i < j and (i,j) not in support]
        
        if not outliers:
            if verbose:
                print("no outliers, checking dual_cost...")
            dual_cost = get_dual_cost(Y, Theta, R, NuY, l0, l2, M, ratio, threshold, zlb, zub, active_set)
            if verbose:
                print("dual", dual_cost)
            if not check_if_integral or tree_upper_bound is None:
                cur_gap = -2
                tree_upper_bound = dual_cost + 1
            else:
                cur_gap = compute_relative_gap(tree_upper_bound,cost)
    
            if cur_gap < mio_gap and tree_upper_bound > dual_cost:
                if (compute_relative_gap(cost, dual_cost) < rel_tol) or \
                        (cd_tol < 1e-8 and check_if_integral):
                    break
                else:
                    cd_tol /= 100
            else:
                break
        
        support = support | set(outliers)
        active_set = support_to_active_set(support)
        curiter += 1
        
    if curiter == kkt_max_itr:
        print('Maximum KKT check iterations reached, increase kkt_max_itr '
              'to avoid this warning')
    
    support = set([(i,j) for [i,j] in np.argwhere(Theta) if i<j])
    active_set = support_to_active_set(support)
    primal_cost, z = get_primal_cost(Y, Theta, R, l0, l2, M, ratio, threshold, zlb, zub, active_set)
    
    if dual_cost is not None:
        prim_dual_gap = compute_relative_gap(cost, dual_cost)
    else:
        prim_dual_gap = 1
        
    if check_if_integral:
        if prim_dual_gap > rel_tol:
            if is_integral(z, 1e-4):
                if verbose:
                    print("integral solution obtained: perform exact optimization")
                ws = dict()
                ws['support'] = support
                ws['Theta'] = Theta
                sol = solve(Y=Y, l0=l0, l2=l2, M=M, fixed_ones=fixed_ones, fixed_zeros=fixed_zeros,
                           ratio=ratio, threshold=threshold, S_diag=S_diag, warm_start=ws,
                           rel_tol=rel_tol, ordering=ordering, tree_upper_bound=tree_upper_bound,
                           mio_gap = mio_gap, check_if_integral = False, cd_max_itr=cd_max_itr, kkt_max_itr=kkt_max_itr, verbose=verbose)
                return sol

    sol = Solution(primal_value=primal_cost, dual_value=dual_cost,
                    support=support, Theta=Theta,
                    sol_time=time() - st, z=z, R=R)
    return sol
