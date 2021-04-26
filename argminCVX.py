"""implementation of argmin step"""

from scipy import optimize
import numpy as np


def argmin(eta, s_radius, barrier, g_bar_aggr_t, g_tilde, d, max_iter = 1e4):
    #implement argmin_ball(eta * (g_bar_1:t + g_tilde_t+1)^T x + barrier(x)
    #argmin is over ball with radius r (not necessary because of barrier?
    TOL = 1e-10  # numerical error allowed
    cons = {'type': 'ineq', 'fun': lambda x: s_radius - np.linalg.norm(x),
            'jac': lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > TOL else np.zeros(x.shape)}
    res = optimize.minimize(fun=obj,
                            x0=np.zeros(d),
                            args=(eta, barrier, g_bar_aggr_t, g_tilde),
                            constraints=cons,
                            options={'disp': True, 'maxiter': max_iter})
    return res['x']


def obj(x, eta, barrier, g_bar_aggr_t, g_tilde):
    return eta * np.dot(g_bar_aggr_t + g_tilde, x) + barrier(x)