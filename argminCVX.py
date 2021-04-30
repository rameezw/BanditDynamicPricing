"""implementation of argmin step"""

from scipy import optimize
import numpy as np
from BanditPricing import randUnitVector

def argmin(eta, s_radius, barrier, g_bar_aggr_t, g_tilde, d, max_iter = 1e4):
    #implement argmin_ball(eta * (g_bar_1:t + g_tilde_t+1)^T x + barrier(x)
    #argmin is over ball with radius r
    TOL = 1e-10  # numerical error allowed
    #g_bar_aggr_t = complex_to_real(g_bar_aggr_t)
    #g_tilde = complex_to_real(g_tilde)
    #init_pt = complex_to_real(randUnitVector(d)*s_radius/2)
    cons = {'type': 'ineq', 'fun': lambda x: s_radius - np.linalg.norm(x),
            'jac': lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > TOL else np.zeros(x.shape)}
    res = optimize.minimize(fun=obj,
                            x0=randUnitVector(d)*s_radius/2,
                            args=(eta, barrier, g_bar_aggr_t, g_tilde),
                            constraints=cons,
                            options={'disp': True, 'maxiter': max_iter})
    return res['x']


def obj(x, eta, barrier, g_bar_aggr_t, g_tilde):
    return eta * np.dot(g_bar_aggr_t + g_tilde, x) + barrier(x)

def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))