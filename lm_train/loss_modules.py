"""Modules for dealing with the loss functions.

The actual loss functions are still defined in the main script.
"""
import numpy as np
import torch
from torch.func import vmap, jacrev


@torch.no_grad()
def cal_L_vec(params, *args):
    """Calculate the loss vector.
    
    args[i] is a tuple of (loss, data, all_targets), where all_targets is a tuple of (target_1, target_2, ...) that contains all the targets for the loss function with arbritary number of targets.
    """
    L_vec_list = list()
    factors = list()
    for i in range(len(args)):
        loss, targets, kwargs = args[i]
        batch_dim = tuple([None] + [0] * (len(targets)) + [None] * len(kwargs))
        if targets[0].shape[0] == 0:
            continue
        factors.append(np.sqrt(1 / targets[0].shape[0]))
        L_vec = vmap(loss, batch_dim)(params, *targets, **kwargs)
        L_vec_list.append(L_vec)
    L_vec = torch.cat(
        [factor * L_vec for factor, L_vec in zip(factors, L_vec_list)])
    return L_vec


@torch.no_grad()
def cal_J_part(params, loss, *targets, **kwargs):
    """Calculate the Jacobian matrix for a loss function."""
    batch_dim = tuple([None] + [0] * (len(targets)) + [None] * len(kwargs))
    per_sample_grads = vmap(jacrev(loss), batch_dim)(params, *targets,
                                                     **kwargs)
    cnt = 0
    for g in per_sample_grads.values():
        g = g.detach()
        J_d = g.reshape(len(g), -1) if cnt == 0 else torch.hstack(
            [J_d, g.reshape(len(g), -1)])
        cnt = 1
    return J_d


@torch.no_grad()
def cal_J_mat(params, *args):
    """Calculate the Jacobian matrix for the loss vector.
    
    Shares the same arguments as cal_L_vec.
    """

    J_mat_list = list()
    factors = list()
    for i in range(len(args)):
        loss, targets, kwargs = args[i]
        if targets[0].shape[0] == 0:
            continue
        factors.append(np.sqrt(1 / targets[0].shape[0]))
        J_mat = cal_J_part(params, loss, *targets, **kwargs)
        J_mat_list.append(J_mat)
    J_mat = torch.cat(
        [factor * J_mat for factor, J_mat in zip(factors, J_mat_list)])
    return J_mat
