"""file that defines barrier functions for OptimisticPricing"""


import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import hessian

# TODO: easiest way: hardcode r into this file in ball_barrier r = radius of S
# see Lemma A.2 in Mueller

def ball_barrier_20(x):
    return -1 * np.log(20**2 - np.linalg.norm(x)**2)

def hessian_ball_20(): #need to define function in the simulation file somehow
    return hessian(ball_barrier_20) #hessian of f
#TODO:
def hessian(x, hess):
    # to fix problem with NaN hessians
    if x is all zeros
    #perturb x
    return hess(x)
