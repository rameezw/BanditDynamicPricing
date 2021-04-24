"""file that defines barrier functions for OptimisticPricing"""


import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad

# TODO: easiest way: hardcode r into this file in ball_barrier r = radius of S


def ball_barrier(x):
    return -1 * np.log(1 - np.linalg.norm(x)**2)

def hessian_ball(): #need to define function in the simulation file somehow
    return grad(grad(ball_barrier)) #hessian of f