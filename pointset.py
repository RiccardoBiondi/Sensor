import numpy as np
import numpy.random as rnd

"""
Implementation of some function useful to operate with point configuarion
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
Functions:
    generate: function to random generate an d-dimensional n-point configuration
    update:  function to move each point of the configuration in a random
        direction
"""

def generate(shape, side):
    """
    Generate a random point configuation uniformly distributed in an
     hypersquare.
     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
     Parameters:
        shape: configuration shape (d,n) where d is the embedding dimension and
                n is the number of points
        side: side of the hypersquare
    Return:
        (d,n) point configuarion as an np.ndarray

    """

    return side*rnd.random_sample(shape)


def update(P, step) :
    """
    Move each point of the configuration along a direction
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        P: (2,n) point configuration to update
        step: size of the movement
    """

    N = np.shape(P)[1]
    R = np.empty(np.shape(P), dtype = np.float64)
    for i in range(N) :
        #generate the angle
        #phase and angol to determine the movement direction
        phase = np.pi*rnd.sample() -2*np.pi*rnd.sample()
        teta = np.pi*rnd.sample()
        #move the pointset
        R[0,i] = P[0,i]+ step*np.cos(teta +phase)
        R[1,i] = P[1,i]+ step*np.sin(teta +phase)

    return R
