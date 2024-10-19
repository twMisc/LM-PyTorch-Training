"""The training module.

More specifically, this file defines the Levenberg-Marquardt(LM) algorithm training loop using the loss functions defined in loss_modules.py.
"""
import numpy as np
import torch
import torch.nn as nn
import copy
import time
import functools
from .loss_modules import cal_L_vec, cal_J_mat


@torch.no_grad()
def training_LM(params, device, args, **kwargs):
    """The training loop for the LM algorithm."""
    # check arguments
    defaults = {
        'steps': 1000,
        'writePercent': 10,
        'args_test': None,
        'mode': 0,
        'args_func': None,
        'save_best_model_weight': False,
        'min_loss': None,
    }
    diff = set(kwargs.keys()) - set(defaults.keys())
    if diff:
        raise KeyError(f"Invalid arguments {diff}.")
    defaults.update(kwargs)
    steps = defaults['steps']
    writePercent = defaults['writePercent']
    args_test = defaults['args_test']
    mode = defaults['mode']
    args_func = defaults['args_func']
    save_best_model_weight = defaults['save_best_model_weight']
    min_loss = defaults['min_loss']
    if args_test is None:
        use_test = False
    else:
        use_test = True
    if steps == 0:
        return params, None, None, None

    # initialize
    cost_time = 0.
    lossval = []
    lossval_all = []
    lossval_test = []
    loss_running = []
    loss_old = 0
    loss_new = np.inf
    step = 0
    augmenting = True
    loss_min = np.inf
    if save_best_model_weight:
        best_params = copy.deepcopy(params)

    # LM parameters
    mu = 1e6
    decay_factor = 3
    grow_factor = 2
    lower_const = 1e-25
    upper_const = 1e8
    evaluate_checker = 1

    # training loop
    try:
        while step < steps:
            time_start = time.time()

            # evaluate loss
            if evaluate_checker >= 1:
                # Calculate loss term before mean square
                L_vec = cal_L_vec(params, *args)
                # old loss
                Ls = torch.sum(L_vec**2)
                loss_old = Ls.item()
            else:
                loss_old = lossval[-1]

            # (re)evaluate J
            if evaluate_checker < 2:
                J_mat = cal_J_mat(params, *args)

            # solve and update weights
            with torch.no_grad():
                if step == 0:
                    Id = torch.eye((J_mat.shape[1])).to(device)
                if evaluate_checker < 2:
                    J_product = J_mat.t() @ J_mat
                rhs = -J_mat.t() @ L_vec
                dp = torch.linalg.solve(J_product + mu * Id, rhs)

                cnt = 0
                for p in params.values():
                    mm = torch.Tensor([p.shape]).tolist()[0]
                    num = int(functools.reduce(lambda x, y: x * y, mm, 1))
                    p += dp[cnt:cnt + num].reshape(p.shape)
                    cnt += num

                if mode == 1:
                    args = args_func(args)

            # evaluate new loss
            L_vec = cal_L_vec(params, *args)
            Ls = torch.sum(L_vec**2)
            loss_new = Ls.item()

            # check loss
            with torch.no_grad():
                # accept update
                if loss_new < loss_old:
                    mu = max(mu / decay_factor, lower_const)
                    lossval.append(loss_new)
                    dp_old = dp
                    augmenting = False
                    evaluate_checker = 0
                else:
                    # checking whether to accpet update
                    if step == 0:
                        dp_old = dp
                        evaluate_checker = 0
                    cosine = nn.functional.cosine_similarity(dp,
                                                             dp_old,
                                                             dim=0,
                                                             eps=1e-15)
                    augmenting = (1. - cosine)**2 * loss_new > loss_min

                    # decline update
                    if augmenting:
                        mu = min(grow_factor * mu, upper_const)
                        cnt = 0
                        for p in params.values():
                            mm = torch.Tensor([p.shape]).tolist()[0]
                            num = int(
                                functools.reduce(lambda x, y: x * y, mm, 1))
                            p -= dp[cnt:cnt + num].reshape(p.shape)
                            cnt += num
                        if mode == 1:
                            args = args_func(args)
                        evaluate_checker += 1
                    # accept update
                    else:
                        mu = max(mu / decay_factor, lower_const)
                        lossval.append(loss_new)
                        dp_old = dp
                        evaluate_checker = 0

            time_end = time.time()
            time_c = time_end - time_start
            cost_time += time_c
            step = step + 1

            #### evaluate step ####
            with torch.no_grad():
                if loss_old < loss_min:
                    loss_min = loss_old
                    if save_best_model_weight:
                        best_params = copy.deepcopy(params)
                if loss_new < loss_min:
                    loss_min = loss_new
                    if save_best_model_weight:
                        best_params = copy.deepcopy(params)
                if augmenting:
                    loss_running.append(loss_old)
                else:
                    loss_running.append(loss_new)
                lossval_all.append(loss_min)

                # printing loss
                if (step + 1) % (steps * writePercent / 100) == 0:
                    print(
                        f"Step: {step+1}. loss: {lossval_all[-1]:.4e}. mu: {mu:.4e}."
                    )
                    if use_test:
                        L_vec_test = cal_L_vec(params, *args_test)
                        Ls_test = torch.sum(L_vec_test**2)
                        lossval_test.append(Ls_test.item())
                if min_loss is not None:
                    if loss_min <= min_loss:
                        print(f'loss_min<={min_loss}')
                        break

            #### evaluate step ####

    except KeyboardInterrupt:
        if save_best_model_weight:
            params = copy.deepcopy(best_params)
        print(f"training time: {cost_time} (s).")
        return params, lossval_all, loss_running, lossval_test

    if loss_new > loss_min:
        if save_best_model_weight:
            params = copy.deepcopy(best_params)
    else:
        pass

    print(f"training time: {cost_time} (s).")
    return params, lossval_all, loss_running, lossval_test
