import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sys
from utilities import EDM, mask, error, add_error
from functional import cost_function,gradient,hessian
from pointset import generate, update
from reconstruct import rec


"""
Simulation of 10 points moving in a square of side 10.m with 1 anchor at eanch
vertex. The aim of the simulation is to estimate for each step the point
configuration. As initial guess I've used the result of the estimation of the
previous one.
"""

n_step = 20 #number of step
shape = (2,10) # matrix containing the position of the 10 sensor in 2-dim
side = 10. #side of the square in which the sensors are generated

OR = np.zeros((2,14), dtype = np.float64) #original point set
PR = np.zeros((2,14), dtype = np.float64) #estimated point set
RME = np.zeros(n_step, dtype = np.float64)#relative mean error

#define the anchors position, which is fixed and known
A = np.array([[0.,0.,10.,10.],[0.,10.,0.,10.]], dtype = np.float64)

#generate the initial point configuration
TAG = generate(shape,side)

#Initzialize original and reconstructed pointset
OR[:,0:4] = A
OR[:,4:]  = TAG
PR[:,0:4] = A
PR[:,4:]  = TAG

#compute the EDM and add error
D = EDM(OR)#
E = error(2.,(14,14))
D = add_error(D,E)
#define the mask. Only distances between sensors are missing
W = np.ones((14,14),dtype = np.float64)
W[4:,4:] = np.zeros((10,10), dtype = np.float64)

my_dpi = 96#constant for figure size

#init the progress bar
toolbar_width = n_step
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after

#start the simulatin
for i in range(n_step):

    #store the old pointset
    P0 = PR
    #update the current pointset
    OR = update(OR,1.)
    OR[:,:4] = A #enforce anchor locations
    #compute the new edm
    D = EDM(OR)
    E = error(2.,(14,14))
    D = add_error(D,E)

    #reconstruct the pointset
    PR = rec(P0,D,W,0.0001)
    #enforce anchor locations
    PR[:,:4] = A

    #conpute the RME
    RME[i] += np.linalg.norm(OR - PR)/np.linalg.norm(OR)

    #save the results
    fig = plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
    plt.scatter(OR[0],OR[1], c = "b")
    plt.scatter(PR[0,:4], PR[1,:4], c="g", marker = "s")
    plt.scatter(PR[0,4:], P0[1,4:], c ="r")
    plt.xlim(-3, 15)
    plt.ylim(-3, 15)
    plt.gca().set_title("step "+str(i))
    plt.gca().set_xlabel("x(m)")
    plt.gca().set_ylabel("y(m)")
    plt.legend(['Actual','Anchors', 'Estimated'])
    filename='step'+str(i)+'.png'
    plt.savefig(filename, dpi=96)
    plt.gca()

    sys.stdout.write("-")
    sys.stdout.flush()

np.savetxt("RME.txt", RME)
