"""file that defines barrier functions for OptimisticPricing"""


import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad

def ball_barrier(r, x):
    return -1 * np.log(r**2 - np.linalg.norm(x)**2)

def hessian(f): #need to define function in the simulation file somehow
    return grad(grad(f)) #hessian of f