import numpy as np
import numba as nb

def CreateLinearCompound(n):
    #create structure of hydrocarbon compound
    Omega = 3*np.ones((n,1))
    Omega[1:-1] = 2
    #B = np.ones((n-1,1))
    #C = {'Omega': Omega, 'B':B}

    return Omega

#@nb.jit(nopython=False)
def DataInMolecules(x,y,centers):
    #Compute important data
    SigmaSplitX, SigmaSplitY, dist, q, n = DataCalc(x,y,centers)
    SigmaSplit = {'X': SigmaSplitX, 'Y': SigmaSplitY}
    moleculeNumber = np.argmin(dist, axis=1)
    for g in range(0, q):
        SigmaSplit['X'][moleculeNumber[g]].append(x[g,:])
        SigmaSplit['Y'][moleculeNumber[g]].append(y[g,:])
    for u in range(0,n):
        SigmaSplit['X'][u] = np.asarray(SigmaSplit['X'][u])
        SigmaSplit['Y'][u] = np.asarray(SigmaSplit['Y'][u])
    SigmaSplitX = SigmaSplit['X']
    SigmaSplitY = SigmaSplit['Y']
    return SigmaSplitX, SigmaSplitY

@nb.jit(nopython=True, parallel=False)
def DataCalc(x,y,centers):
    #Compute important data
    q = x.shape[0]
    n = centers.shape[0]
    #Initialize splitting structure and distance matrix
    SigmaSplitX = [[np.float64(x) for x in range(0)] for _ in range(0,n)]
    SigmaSplitY = [[np.float64(x) for x in range(0)] for _ in range(0,n)]
    #SigmaSplit = {'X': [], 'Y': []}
    dist = np.zeros((q,n))
    for i in range(0,n):
        for j in range(0, q):
            dist[j][i] = np.sqrt(np.sum((x[j][:] - centers[i][:])**2))

    return SigmaSplitX, SigmaSplitY, dist, q, n


#@nb.jit(nopython=True, parallel=False)
def ComputeMoleculeParameters(Xi, Yi, Omegai):
    #Calculate phi-values in the series of the molecular function
    Phi = CH_X(Xi, Omegai)

    #Find the molecular parameters using least-squared error (LSE)
    H = np.linalg.lstsq(Phi, Yi)[0]

    #Determine error between the targets and estimates
    error = np.sqrt(np.mean(np.abs((Phi @ H)-Yi)**2))

    return H, error

@nb.jit(nopython=True, parallel=False)
def CH_X(X,h):
    #Calculate important data
    q = X.shape[0]
    numDims = X.shape[1]

    # Calculate phi-values in the seires of the molecular function
    Phi = np.zeros((q,1))
    for i in range(0, numDims):
        varphi = np.ones((q,1))
        for k in range(1,h+1):
            #varphi = [varphi] + [X[:,i]**k]
            varphi = np.concatenate((varphi, np.reshape((X[:,i]**k),(q,1))), axis=1)

        Phi = np.concatenate((Phi, varphi), axis=1)

    Phi = Phi[:,1:]

    return Phi