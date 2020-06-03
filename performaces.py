import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sys
from utilities import EDM, mask, error, add_error
from functional import cost_function,gradient,hessian
from pointset import generate, update
from reconstruct import rec

"""
Routines to check the behaviour of the algorithm by changing the distances between the initial guess and the actual position for different noise
"""


n_init = 20 #number of different distances between actual position and initial guess
n_step = 25 #number of repetitions for each initial guess

# setup toolbar
toolbar_width = n_step



DIST = np.zeros(20, dtype = np.float64) #distance between the guess and the original

#rms error between the reconstruct and the original for first noise level
RMSE0 = np.zeros(20,dtype = np.float64)
#rms error between the reconstruct and the original for second noise level
RMSE1 = np.zeros(20,dtype = np.float64)
#rms error between the reconstruct and the original for third noise level
RMSE2 = np.zeros(20,dtype = np.float64)
#rms error between the reconstruct and the original for fourth noise level
RMSE3 = np.zeros(20,dtype = np.float64)
#rms error between the reconstruct and the original for fifth noise level
RMSE4 = np.zeros(20,dtype = np.float64)

#defining the mask matrix
W = np.ones((10,10),dtype=np.float64)

print("Starting the simulation")
for i in range(n_init) :
    print("Iteration number: "+str(i))
    #progression bar for each iteration
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    for j in range(n_step) :

        #generate the original pointset
        P = generate((2,10), 10.)
        #save the initial guess
        PI = P
        #update the pointset
        P = update(P,i/10.)
        #save the difference between guessed and original
        DIST[i] += np.linalg.norm(P-PI) /np.linalg.norm(P)
        #find the noiseless EDM
        D = EDM(P)

        #add the different errors
        D0 = add_error(D,error(0., (10,10)))
        D1 = add_error(D,error(.5, (10,10)))
        D2 = add_error(D,error(1., (10,10)))
        D3 = add_error(D,error(1.5,(10,10)))
        D4 = add_error(D,error(2., (10,10)))

        #find the five reconstruction
        P0 = rec(PI,D0,W,0.0001)
        P1 = rec(PI,D1,W,0.0001)
        P2 = rec(PI,D2,W,0.0001)
        P3 = rec(PI,D3,W,0.0001)
        P4 = rec(PI,D4,W,0.0001)

        #save the errors
        RMSE0[i] += np.linalg.norm(P-P0) /np.linalg.norm(P)
        RMSE1[i] += np.linalg.norm(P-P1) /np.linalg.norm(P)
        RMSE2[i] += np.linalg.norm(P-P2) /np.linalg.norm(P)
        RMSE3[i] += np.linalg.norm(P-P3) /np.linalg.norm(P)
        RMSE4[i] += np.linalg.norm(P-P4) /np.linalg.norm(P)

        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("]\n") # this ends the progress bar

    #compute the mean
    DIST[i] = DIST[i]/25.
    RMSE0[i] = RMSE0[i]/25.
    RMSE1[i] = RMSE1[i]/25.
    RMSE2[i] = RMSE2[i]/25.
    RMSE1[i] = RMSE3[i]/25.
    RMSE4[i] = RMSE4[i]/25.



#save the results
print("Saving data")

np.savetxt("distance.txt", DIST)
np.savetxt("RMSE0.txt", RMSE0)
np.savetxt("RMSE1.txt", RMSE1)
np.savetxt("RMSE2.txt", RMSE2)
np.savetxt("RMSE3.txt", RMSE3)
np.savetxt("RMSE4.txt", RMSE4)
