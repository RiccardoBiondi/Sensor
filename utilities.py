import numpy as np
import scipy.spatial.distance as dist

"""
Implementation of some usefull function to manage with EDM, mask and errors.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ __
Functions:
    EDM: generate the EDM corresponding to a particular point configuration
    mask: generate a mask matrix
    error: generate a matrix that contains noise
    add_error: sum the EDM and the error matrix, leave the diagonal to 0.
    unvec: reshape a dn vector to an (d,n) matrix
"""

def EDM (P):
    """
    Compute the EDM for a particular point configuration
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameter:
        P : (d,n) point configuration where n is the number of points, d is the
                embedding dimension
    Return
        (n,n) EDM
    """
    return dist.squareform(dist.pdist(P.transpose(),metric="euclidean"))**2


def mask(D, tr) :
    """
    Compute the mask of missing entries for a particular EDM
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        D: (n,n) EDM from which compute the mask
        tr: maximum distance that allow the measure
    Return:
        (n,n) matrix

    """
    W = np.zeros(np.shape(D),dtype = np.float64)
    for i in range(np.shape(D)[0]):
        for j in range(np.shape(D)[0]):
            if D[i,j] < tr :
                W[i,j] = 1.

    return W

def error(sigma, shape):
    """
    Compute a gaussian noise matrix
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameter:
        sigma: noise size
        shape: dimension of the noise matrix
    Return:
        a shape matrix containing normal distributed noise
    """

    return sigma*np.random.randn(shape[0],shape[1])

def add_error(D,E):
    """
    Add EDM and noise by setting the diagonal to 0.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        D: (n,n) noiseless EDM
        E: (n,n) noise matrix
    Return:
        hollow D+E
    """
    #add noise
    R = np.add(D,E)
    #make hollow
    for i in range(np.shape(D)[0]):
        R[i,i] = 0.
    return R


def unvec( V, shape):
    """
    reshape the vector V
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    V: nd vector
    shape: (d,n) new shape
    """

    R = np.empty(shape)
    d = shape[0]
    N = shape[1]

    for i in range(N):
        for k in range(d):
            R[k,i] = V[d*i+k]
    return R
