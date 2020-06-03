import numpy as np
import numpy.linalg as nplg
import scipy.linalg as sclg
from functional import cost_function,gradient,hessian
from modCholDec import modDec
from utilities import unvec

"""
Implemetation of the modified newton method for cost function minimization

"""


def rec(P0, D,W, conv):
    """
    Use the modified newton method to reconstruct the point configuration that
    generate D
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        P0: initial guess, is an (d,n) matrix, where d is the embedding
            dimension and n is the nuber of points
        D: measured (n,n) EDM
        W: (n,n) Mask matrix corresponding to D
        conv: convergence rate
    Returns:
        (d,n) matrix containing the estimated configuration that generates D
    """

    first = 1.#first term of convergence condition
    second = 0.#second term of convergence condition
    P = P0#initial guess
    iter = 0#iter counter

    #starting the cycle
    while first> second:
        #store the old configuration
        POld = P

        #compute hessian and gradient
        G = gradient(P,D,W)
        H = hessian(P,D,W)

        #compute LS via modified cholewsky decomposition
        S,L = modDec(H)

        #find the upgrade term by solving the systems
        epsII = sclg.solve_triangular(L,-G)
        epsI  = nplg.solve(S,epsII)
        eps   = sclg.solve_triangular(L,epsI)
        term = unvec(eps, np.shape(P0))

        #update the guess
        P = P + (2**-iter)*term
        iter +=1
        
        #update convergence terms
        first = abs(cost_function(P,D,W)-cost_function(POld,D,W))
        second = conv*nplg.norm(P-POld, ord = 2)

    return P
