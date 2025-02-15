"""file that defines barrier functions for OptimisticPricing"""

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import hessian


# TODO: easiest way: hardcode r into this file in ball_barrier r = radius of S
# see Lemma A.2 in Mueller

def unit_ball_barrier(x):
    return -1 * np.log(- np.linalg.norm(x) ** 2)


def ball_barrier_20(x): #  is a function
    return -1 * np.log(20 ** 2 - np.linalg.norm(x) ** 2)


def hessian_ball_20():  # returns a hessian as a funciton
    # print(hessian(ball_barrier_20))
    return hessian(ball_barrier_20)  # hessian of f


#def hessian(x, hess):
    # # to fix problem with NaN hessians
    # print(x)
    # is_all_zero = np.all((x == 0))
    # if is_all_zero:
    #     # perturb x
    #     x = x + np.random.normal(0, .1, x.shape)
    # elif np.isnan(x):
    #     x = np.where(np.isnan(x), 1, x)
    # print(x)
    #return hess(x, 0)
