"""implementation of argmin step using cvxopt"""

import cvxopt


# TODO: rameez? use cvx to do argmin here
def argmin(x, eta, barrier, g_bar, g_tilde):
    #implement argmin_ball(eta * (g_bar_1:t + g_tilde_t+1)^T x + barrier(x)
    #argmin is over ball with radius r (not necessary because of barrier?
    pass
