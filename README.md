# SENSOR POSITION ESTIMATION

In this repository I've implemented some function useful to reconstruct the
sensor position starting from an incomplete and noisy Euclidean distance matrix (EDM). I've also implemented two performances test and one simulation,

##Structure

The code is structured as follow:
Name | Contents
-----|---------
functional.py | contains the implementation of the cost function, its gradient and hessian

mdCholDec.py| contains the implementation of the modified Cholesky factorization

pointset.py | contains the implementation of function to generate and manage point configurations

utilities.py | contains useful tools to work with EDM
reconstruct.py | contains the implementation of the modified Newton method for cost function minimization

performances.py | contains the implementation of a routine to check the behaviour of the algorithm to the initial guess

performances_bis.py | contains the routine to check the behaviour of the algorithm depending on the structure of the mask matrix

simulatin.py | contains the implementation of a routine that reconstruct the positions of point moving in the space
