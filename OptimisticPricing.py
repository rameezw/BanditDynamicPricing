from __future__ import division
import numpy as np
from BanditPricing import *
from argminCVX import *
from scipy.linalg import sqrtm
def MOPOL(demands_prev, eta, delta, k, s_radius, prev_state, barrier, hessian):
    """ Modified Low-rank dynamic pricing with *Unknown* product features (latent U is assumed orthonormal).
    TODO: initialize g_aggr to be 0 vec with dim d, set k
    Args:
            demands_prev (1D array): Observed product demands at the prices p_tilde chosen by this method in the last round.
                                     Is used to calculate: R_prev = -p_tilde * demands_prev
            eta (float): Positive value.
            delta (float): Positive value. Recall, if revenue is bounded by B and L-Lipshitz and r = s_radius,
                           we want to set: eta = r/(B*sqrt(T)), delta = T^(-1/4) * sqrt((BNr^2)/(3*(Lr + B))),
                           alpha parameter is automatically set = delta/r inside this function.
            s_radius (float): Postive radius of constraint-set, which must be centered Euclidean ball.
            prev_state (tuple): Internal bandit state from previous rounds that is required for this round
                                (was returned as next_state in previous round).
                                In first round, we want: prev_state = (initial_price, random_unit-vector)
            barrier (function): the self-concordant barrier R(x) (see barrier defs file)
            hessian (function): the hessian of the self concordant barrier \nabla^2 R(x)
    """
    alpha = delta / s_radius
    if (alpha <= 0) or (alpha > 1) or (eta <= 0) or (delta <= 0):
        raise ValueError("eta, delta, or alpha invalid")

    # read prev data
    x_prev_clean, Q, t, update_cnts, xi_prev, p_prev, g_aggr_prev = prev_state

    d = x_prev_clean.shape[0]
    N = p_prev.shape[0]
    # Q update here:
    col_ind = t % d
    cnt_ind = update_cnts[col_ind] + 1  # num times this column has been updated
    update_cnts[col_ind] = cnt_ind
    Q[:, col_ind] = (1.0 / cnt_ind) * demands_prev + ((cnt_ind - 1) / cnt_ind) * Q[:, col_ind]

    # while Q contains zero column (first d rounds), simply select next price randomly:
    if np.min(np.sum(np.abs(Q), axis=0)) == 0.0:
        p_rand = p_prev + delta * randUnitVector(N) / 10.0
        if np.linalg.norm(p_rand) > s_radius:
            p_rand = s_radius * (p_rand / np.linalg.norm(p_rand))
        Uhat, singval, right_singvec = np.linalg.svd(Q, full_matrices=False)
        x_prev_clean = np.dot(Uhat.transpose(), p_prev)  # first low-dimensional action.
        next_state = (x_prev_clean, Q, t + 1, update_cnts, xi_prev, p_prev, g_aggr_prev)
        return ((p_rand, next_state))

    # Otherwise run our algorithm: we have xi_prev from before
    Uhat, singval, right_singvec = np.linalg.svd(Q, full_matrices=False)  # Update product-feature estimates.
    R_prev = negRevenue(p_prev, demands_prev)

    #g_aggr_prev = (list of g_hats up to t-1, g_bar_1:t-1 (sum of subgradients up to t-1)
    hat_list, g_bar_prev = g_aggr_prev

    g_hat = d / delta * R_prev @ sqrtm(hessian(x_prev_clean)) @ xi_prev
    hat_list.append(g_hat)

    g_bar = set_g_bar(hat_list, k)
    g_tilde = set_g_tilde(hat_list, k)

    #set g_bar_1:t
    g_bar_aggr_t = g_bar_prev + g_bar

    # use cvxopt here to do argmin
    x_next_clean = argmin(x_prev_clean, eta, s_radius, g_bar_aggr_t, g_tilde) # approximate gradient step.

    #setting up next iteration + getting prices from prev
    xi_next = randUnitVector(d) #sample UAR from sphere

    x_tilde = x_next_clean + delta * sqrtm(hessian(x_prev_clean)) @ xi_next
    g_aggr_next = (hat_list, g_bar_aggr_t)

    p_tilde = findPrice(x_tilde, Uhat)
    if np.linalg.norm(p_tilde) > s_radius:
        raise ValueError("constraints violated, norm(p_tilde)=" + str(np.linalg.norm(p_tilde)))
    next_state = (x_next_clean, Q, t + 1, update_cnts, xi_next, p_tilde, g_aggr_next)
    return ((p_tilde, next_state))


def set_g_bar(hat_list, k):
    if len(hat_list) < k+1:  # if hat_list not long enough, just take avg.
        return np.add.reduce(hat_list)
    else:
        return np.add.reduce(hat_list[k+2:-1]) / (k + 1)

def set_g_tilde(hat_list,k):
    if len(hat_list) < k:  # if hat_list not long enough, just take avg.
        return np.add.reduce(hat_list)
    else:
        return np.add.reduce(hat_list[k+1:-1]) / (k + 1)
