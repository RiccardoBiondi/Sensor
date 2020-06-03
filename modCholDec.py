import numpy as np
import math

"""
Implementation of the function to perform the Modified Cholewsky factorization
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
Functions:
    modDec: fucntion to perform the factorization.
Internal functions:
    constants: function to compute useful constants to perform the factorization
    L_row: function to update the row of the lower triangular matrix
    C_col: function to update the column of the internal matrix C
    updateC : function to update the internal matrix C
    updateD : function to update the diagonal matrix
    updateTeta : function to update the internal number teta
    unitary_diagonal : function to make unitary the diagonal of a matrix
"""



def constants (m_A) :
    """
    Function to compute useful constats to perform the modified Cholewsky factorization.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameter:
        m_A: symmetric matrix to factorize
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Returns:
        the constants beta, gamma and delta

    """
    n = np.shape(m_A)
    D = m_A.diagonal()
    #define delta as the epsilon of the machine
    delta = np.finfo(np.float64).eps
    #maximum element on the diagonal of the input matrix
    gamma = np.amax(D)
    zi = np.amax(m_A-np.diag(D))
    nu = np.amax([1., math.sqrt(n[0]**2-1)])
    #maximum between gamma,delta previously defined and the number zi/nu
    beta = np.amax([gamma,zi/nu,delta])

    return[beta,gamma,delta]


def L_row (i,t_C,t_D,t_L) :
    """
    Function to compute the ith row of the lower triangular matrix L
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        i: index corresponding to the row to compute
        t_C: matrix C
        t_D: diagonal matrix
        t_L: lower triangular matrix to compute the row
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Returns:
        L: input triangular matrix with the ith row updated
    """
    b = np.arange(0,i,1)
    L = t_L
    if(i > 0):
        for bb in b :
            L[i,bb] = t_C[i,bb]/t_D[bb,bb]
    else:
        pass
    return L



def C_col (i,t_A,t_C,t_L) :
    """
    Function to compute the ith column of the matrix C
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        i: index corresponding to the row to compute
        t_A: symmetric matrix to factorize
        t_C: matrix to update
        t_L: lower triangular matrix to compute the row
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Returns:
        update the ith column of the input matrix C
    """
    n = np.shape(t_A)[0]
    ee = np.arange(i+1,n,1,dtype=np.int8)
    if(i>0) :
        if (i< n-1):
            for e in ee :
                t_C[e,i] = t_A[e,i] -np.sum(t_L[i,b]*t_C[e,b] for b in range(i))

    else :
        for e in ee :
            t_C[e,i] = t_A[e,i]



def updateC(i,t_C,t_D):
    """
    Function to update the matrix C
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        i: indicator of the current iteration
        t_C: matrix to update
        t_D: diagonal matrix
        _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        Returns:
            update the input matrix C
        """
    n = np.shape(t_C)[0]
    ee = np.arange(i+1,n,1,dtype=np.int8)
    for e in ee :
        t_C[e,e] = t_C[e,e] - (t_C[e,i]**2)/t_D[i,i]




def updateD(i,t_D,t_C,t_teta,t_beta):
    """
    Function to update the diagonal matrix D
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        i: indicator of the current iteration
        t_D: diagonal matrix to update
        t_C: C matrix
        t_teta : vector of constants
        t_beta: beta constant frm function "constants"
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
        Returns:
            update the diagonal matrix D
            """
    eps = np.finfo(np.float64).eps
    c = np.absolute(t_C[i,i])
    term = float((t_teta[i]**2)/t_beta)
    a =np.array([eps,c,term])
    t_D[i,i] = np.amax(a)





def updateTeta(i,t_teta,t_C):
    """
    Function to compute the constant teta, useful to compute the elemnt of the
    diagonal matrix D
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        i: indicator of the element to update
        t_teta: array that contains the value to update
        t_C: matrix C
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Returns:
        update the ith element of the input array teta
    """
    n = np.shape(t_C)[0]
    if i == n-1 :
        t_teta[i] = 0.
    else :
        t_teta[i] = np.amax(np.absolute(t_C[i+1:n,i]))






def unitary_diagonal(m_L):
    """
    Funcion to set all the diagonal elements of m_L to 1.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        t_L: lower triangular matrix
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Returns:
        t_L with all 1. on the diagonal
        """
    L = m_L
    for i in range(np.shape(m_L)[0]):
        L[i,i] = 1.
    return L



def modDec (A):
    """
    Funcion to perform the modified cholwsky factorization of a symmetric
    matrix A.
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Parameters:
        A : symmetric matrix to factorize
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    Returns:
        [D,L] where :
            D -> diagonal matrix
            L -> lower triangula matrix
             such that A = LDL
    """

    #useful constants beta,gamma,delta
    [beta,gamma,delta] = constants(A)

    #dimension
    n = np.shape(A)[0]

    #matrix declarations
    C = np.diag(A.diagonal())
    D = np.zeros((n,n), dtype = np.float64)
    L = np.zeros((n,n), dtype = np.float64)
    teta = np.zeros(n, dtype = np.float64)

    #starting the algorithm
    for j in range(n) :
        #conpute L jth row
        L = L_row (j,C,D,L)
        #compute C jth column
        C_col (j,A,C,L)
        # update thete[j]
        updateTeta(j,teta,C)
        #update D
        updateD(j,D,C,teta,beta)
        #update C
        updateC(j,C,D)

    #set L diagonal to 1.
    L = unitary_diagonal(L)

    return[D,L]
