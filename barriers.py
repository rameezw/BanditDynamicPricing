"""file that defines barrier functions for OptimisticPricing"""


import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad

# TODO: easiest way: hardcode r into this file in ball_barrier r = radius of S
# see Lemma A.2 in Mueller

def ball_barrier_r(x):
    return -1 * np.log(1 - np.linalg.norm(x)**2)

def hessian_ball_r(): #need to define function in the simulation file somehow
    return grad(grad(ball_barrier)) #hessian of f