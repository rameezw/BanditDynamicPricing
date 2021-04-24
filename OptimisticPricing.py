from __future__ import division
import numpy as np
from BanditPricing import *
from argminCVX import *

def MOPOL(demands_prev, eta, delta, s_radius, prev_state, barrier, hessian):
    """ Modified Low-rank dynamic pricing with *Unknown* product features (latent U is assumed orthonormal).
    TODO: initialize g_bar_aggr = 0 vector
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
            Hessian (function): the hessian of the self concordant barrier \nabla^2 R(x)
    """
    alpha = delta / s_radius
    if (alpha <= 0) or (alpha > 1) or (eta <= 0) or (delta <= 0):
        raise ValueError("eta, delta, or alpha invalid")
    x_prev_clean, Q, t, update_cnts, xi_prev, p_prev, g_bar_aggr_prev = prev_state
    d = x_prev_clean.shape[0]
    N = p_prev.shape[0]
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
        next_state = (x_prev_clean, Q, t + 1, update_cnts, xi_prev, p_prev, g_bar_aggr_prev)
        return ((p_rand, next_state))

    # Otherwise run our algorithm:
    Uhat, singval, right_singvec = np.linalg.svd(Q, full_matrices=False)  # Update product-feature estimates.
    R_prev = negRevenue(p_prev, demands_prev)

    #TODO: code for gradient estimate from Yang+Mohri
    #TODO: keep track of g_hats
    #g_hat =
    #g_bar =
    #g_tilde =

    g_bar_aggr += g_bar_aggr_prev + g_tilde

    #TODO: define r, radius of U^T(S)
    x_next_clean = argmin(x_prev_clean, eta, xi_prev, barrier) # approximate gradient step.

    #don't need projection:

    #setting up next iteration + getting prices from prev
    xi_next = randUnitVector(d) #sample UAR from sphere

    x_tilde = x_next_clean + delta * hessian(x_next_clean) ** 0.5 * xi_next

    p_tilde = findPrice(x_tilde, Uhat) #findPrice
    if np.linalg.norm(p_tilde) > s_radius:
        raise ValueError("constraints violated, norm(p_tilde)=" + str(np.linalg.norm(p_tilde)))
    next_state = (x_next_clean, Q, t + 1, update_cnts, xi_next, p_tilde, g_bar_aggr_next)
    return ((p_tilde, next_state))


    # g_bar_aggr = g_bar_1:t is sum of subgradients g_bar_s from s=1 to t

