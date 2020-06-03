import numpy as np
import scipy.spatial.distance as dist

"""
Implementatio of the cost function, its gradient and its hessian
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
Functions:
    cost_function : Implementation of the cost function to minimize
    gradient : Numerical computation of the gradient of cost_function
    hessian : Numerical computation of the hessian of cost_function
"""

def cost_function (P, D,W) :
    """
    Cost function Implementation
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        P: dxn pointset matrix where n is the number of point, d is the
        embedding dimension

        D: nxn measuered EDM
        W: nxn mask matrix
    """
    #compute the EDM corresponding to P
    Dg = dist.squareform(dist.pdist(P.transpose(),metric="euclidean"))**2

    return np.linalg.norm(np.multiply(W,Dg-D))**2




def gradient(P,D,W):
    """
    Compute the gradient of the cost_function in P ny using the forward
    difference method
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        P: dxn point configuration matrix where n is the number of point, d is
            the embedding dimension. It is the point in which the gradient is
            compute
        D: nxn Measuered EDM
        W: nxn mask matrixù
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Reurn:
        nd array corresponding to the gradient
    """

    grad = []
    #save the cost function in P
    f0 = cost_function(P,D,W)

    for i in range(np.shape(P)[1]):
        for k in range(np.shape(P)[0]):
            #save the initial value of the kth coordinate of the ith point
            xx0 = 1.*P[k,i]
            #define eps
            if P[k,i] != 0:
                eps = abs(P[k,i])*np.finfo(np.float32).eps
            if P[k,i] == 0:
                eps = np.finfo(np.float32).eps
            #
            #move by a small step along the kth coordinate of the ith point
            P[k,i] = P[k,i]+eps
            #compute the function value in the incremented point
            f1 = cost_function(P,D,W)
            #compute the gradient along the kth coordinate of the ith point
            grad.append((f1-f0)/eps)
            #restore the inital situation
            P[k,i] = xx0
    return np.array(grad)



def hessian (P,D,W):
    """
    Compute the hessian of the cost_function in P by using the forward
    difference method
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        P: dxn point configuration matrix where n is the number of point, d is
            the embedding dimension. It is the point in which the gradient is
            compute
        D: nxn Measuered EDM
        W: nxn mask matrix
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Reurn:
        ndxnd array corresponding to the hessian matrix
    """
    N = np.size(P)
    #compute the gradient in P
    g0 = gradient(P,D,W)
    H = np.empty((N,N), dtype = np.float64)
    #define the size of the increment eps
    if np.linalg.norm(g0) != 0:
        eps = np.linalg.norm(g0)*np.finfo(np.float32).eps
    if np.linalg.norm(g0) == 0:
        eps = np.finfo(np.float32).eps
    #starting computation
    for i in range(np.shape(P)[1]):
        for k in range(np.shape(P)[0]):
            #store the initial value of the kth coordinate of the ith point
            xx0 = 1.*P[k,i]
            #increment the direction
            P[k,i] = P[k,i]+eps
            #compute the new gradient
            g1 = gradient(P,D,W)
            #compute the incremental ratio
            H[:,np.shape(P)[0]*i+k] = (g0-g1)/eps
            #restore initial situation
            P[k,i] = xx0

    return (H+H.transpose())/2.
