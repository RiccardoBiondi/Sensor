import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sys
from utilities import EDM, mask, error, add_error
from functional import cost_function,gradient,hessian
from pointset import generate, update
from reconstruct import rec


"""
Routine to check the behaviour of the algorithm for different mask structure.
I've considered three cases:
a) No missing entrance
b) missing the distances between sensors locations
c) missing the distances between sensors locations and  the distances between
    anchors locations
I've  checked each case for different noise
"""


shape = (14,14)
#Define the contaners

#define the array that contains the RME
AW0 = np.zeros(5,dtype =np.float64)
AW1 = np.zeros(5,dtype = np.float64)
AW2 = np.zeros(5, dtype = np.float64)



#define the mask
W0 = np.ones(shape, dtype = np.float64)
W1 = np.zeros(shape, dtype = np.float64)
W2 = np.zeros(shape, dtype = np.float64)

#initzialize the mask for the three cases
W1[:4,:4] = W0[:4,:4]
W1[4:, :4] = W0[4:,:4]
W1[:4, 4:] = W0[:4, 4:]

W2[4:, :4] = W0[4:,:4]
W2[:4, 4:] = W0[:4, 4:]

print("starting the simulation")
#progression bar
toolbar_width = 50
sys.stdout.write("[")
sys.stdout.flush()

#starting the simulation
for _ in range(25):

    #generate the pointset
    P = generate((2,14),10.)
    #store the initial guess
    PI = P
    #update the pointset
    P = update(P, 1.)
    #compute the EDM
    D = EDM(P)

    #add the different errors

    D0 = add_error(D,error(0.,shape))
    D1 = add_error(D,error(.5,shape))
    D2 = add_error(D,error(1.5,shape))
    D3 = add_error(D,error(2.,shape))
    D4 = add_error(D,error(2.5,shape))

    sys.stdout.write("-")
    sys.stdout.flush()

    #reconstruct the pointsets
    W0P0 = rec(PI,D0,W0,0.0001)
    W0P1 = rec(PI,D1,W0,0.0001)
    W0P2 = rec(PI,D2,W0,0.0001)
    W0P3 = rec(PI,D3,W0,0.0001)
    W0P4 = rec(PI,D4,W0,0.0001)

    W1P0 = rec(PI,D0,W1,0.0001)
    W1P1 = rec(PI,D1,W1,0.0001)
    W1P2 = rec(PI,D2,W1,0.0001)
    W1P3 = rec(PI,D3,W1,0.0001)
    W1P4 = rec(PI,D4,W1,0.0001)

    W2P0 = rec(PI,D0,W2,0.0001)
    W2P1 = rec(PI,D1,W2,0.0001)
    W2P2 = rec(PI,D2,W2,0.0001)
    W2P3 = rec(PI,D3,W2,0.0001)
    W2P4 = rec(PI,D4,W2,0.0001)
    sys.stdout.write("-")
    sys.stdout.flush()
    #update the errors
    AW0[0] += np.linalg.norm(P - W0P0)/np.linalg.norm(P)
    AW0[1] += np.linalg.norm(P - W0P1)/np.linalg.norm(P)
    AW0[2] += np.linalg.norm(P - W0P2)/np.linalg.norm(P)
    AW0[3] += np.linalg.norm(P - W0P3)/np.linalg.norm(P)
    AW0[4] += np.linalg.norm(P - W0P4)/np.linalg.norm(P)

    AW1[0] += np.linalg.norm(P - W1P0)/np.linalg.norm(P)
    AW1[1] += np.linalg.norm(P - W1P1)/np.linalg.norm(P)
    AW1[2] += np.linalg.norm(P - W1P2)/np.linalg.norm(P)
    AW1[3] += np.linalg.norm(P - W1P3)/np.linalg.norm(P)
    AW1[4] += np.linalg.norm(P - W1P4)/np.linalg.norm(P)


    AW2[0] += np.linalg.norm(P - W2P0)/np.linalg.norm(P)
    AW2[1] += np.linalg.norm(P - W2P1)/np.linalg.norm(P)
    AW2[2] += np.linalg.norm(P - W2P2)/np.linalg.norm(P)
    AW2[3] += np.linalg.norm(P - W2P3)/np.linalg.norm(P)
    AW2[4] += np.linalg.norm(P - W2P4)/np.linalg.norm(P)
sys.stdout.write("]\n") # this ends the progress bar
#divide

#compute the mean
AW0[0] = AW0[0]/25.
AW0[1] = AW0[1]/25.
AW0[2] = AW0[2]/25.
AW0[3] = AW0[3]/25.
AW0[4] = AW0[4]/25.

AW1[0] = AW1[0]/25.
AW1[1] = AW1[1]/25.
AW1[2] = AW1[2]/25.
AW1[3] = AW1[3]/25.
AW1[4] = AW1[4]/25.

AW2[0] = AW2[0]/25.
AW2[1] = AW2[1]/25.
AW2[2] = AW2[2]/25.
AW2[3] = AW2[3]/25.
AW2[4] = AW2[4]/25.

#save the results
np.savetxt("AW0.txt",AW0)
np.savetxt("AW1.txt",AW1)
np.savetxt("AW2.txt",AW2)
