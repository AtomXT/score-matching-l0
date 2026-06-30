import copy
import sys
import numpy as np

from .node import Node


def max_fraction_branching(z, tol):
    casted_sol = (z+0.5).astype(int)
    sol_diff = z - casted_sol
    max_ind = np.argmax(np.abs(sol_diff))
    if abs(sol_diff.flat[max_ind]) > tol:
        i,j = np.unravel_index(max_ind, z.shape)
        return (i,j) if i < j else (j,i)
    return (-1,-1)


def is_integral(solution, tol):
    return True if max_fraction_branching(solution, tol) == (-1,-1) else False


def new_z(node, index):
    new_zlb = node.zlb.copy()
    new_zlb.add(index)
    new_zub = node.zub.copy()
    new_zub.add(index)
    return new_zlb, new_zub




def branch(current_node, tol, branching_type):
    if branching_type == 'maxfrac':
        branching_variable = \
            max_fraction_branching(current_node.z, tol)
    else:
        raise ValueError(f'branching type {branching_type} is not supported')
    new_zlb, new_zub = new_z(current_node, branching_variable)
    right_ws = dict()
    right_ws['support'] = current_node.support.copy()
    right_ws['support'].add(branching_variable)
    right_ws['Theta'] = np.copy(current_node.primal_Theta)
    right_node = Node(current_node, new_zlb, current_node.zub, warm_start=right_ws)
    left_ws = dict()
    left_ws['support'] = current_node.support.copy()
    left_ws['support'].remove(branching_variable)
    left_ws['Theta'] = np.copy(current_node.primal_Theta)
    i, j = branching_variable
    left_ws['Theta'][i,j] = 0
    left_ws['Theta'][j,i] = 0
    left_node = Node(current_node, current_node.zlb, new_zub, warm_start=left_ws)
    return left_node, right_node
