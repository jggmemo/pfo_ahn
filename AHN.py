import numpy as np
from models.pso_func import ParticleSwarmAlgorithm as PSO
from models.gwo_func import GreyWolfOptimizer as GWO
from models.ba_func import BatAlgorithm as BAT
from models.pfo_func import PiranhaFeastOptimizer as PFO
import numba as nb

def train(x,y,n, iterations, model='PSO'):
    numDims = np.size(x,1)
    numVars = numDims * n
    if model == 'PSO':
        #optimize the center of molecules in the hydrocarbon compound
        #learn_alg = PSO(D=numVars, NP=100, nFES=iterations, C1=1.49, C2=1.49, w=-0.35, vMin=-0.5, vMax=0.5, benchmark=ModelEval(n, x, y))
        lowerAux = list(x.min(axis=0))
        lowerPi = [e for _ in range(n) for e in lowerAux]
        Lower = np.asarray(lowerPi)

        upperAux = list(x.max(axis=0))
        upperPi = [e for _ in range(n) for e in upperAux]
        Upper = np.asarray(upperPi)

        error = PSO(D=numVars, nFES=iterations, NP=100, C1=1.49, C2=1.49, w=-0.35, vMin=-0.5, vMax=0.5, Lower=Lower, Upper=Upper, x=x, y=y, n=n)
        #Define the best ahn structure based on the center of molecules
        #ahn = learn_alg.gBestAhn
        #error = learn_alg.gBestFitness
    if model == 'GWO':
        lowerAux = list(x.min(axis=0))
        lowerPi = [e for _ in range(n) for e in lowerAux]
        Lower = np.asarray(lowerPi)

        upperAux = list(x.max(axis=0))
        upperPi = [e for _ in range(n) for e in upperAux]
        Upper = np.asarray(upperPi)

        error = GWO(D=numVars, NP=10, nFES=iterations, Lower=Lower, Upper=Upper, x=x, y=y, n=n)
        #error = learn_alg.run()
        #ahn = learn_alg.Alpha_ahn
        #error = learn_alg.Alpha_score
    if model == 'BAT':
        lowerAux = list(x.min(axis=0))
        lowerPi = [e for _ in range(n) for e in lowerAux]
        Lower = np.asarray(lowerPi)

        upperAux = list(x.max(axis=0))
        upperPi = [e for _ in range(n) for e in upperAux]
        Upper = np.asarray(upperPi)

        error = BAT(A=1.5, r=0.5, Qmin=10, Qmax=15, D=numVars, NP=5, nFES=iterations, Lower=Lower, Upper=Upper, x=x, y=y, n=n)

        #learn_alg = BAT(D=numVars, NP=5, nFES=iterations, Qmin=13, Qmax=15, A=0.7, r=0.35, benchmark=ModelEval(n,x,y))
        #error = learn_alg.run()
        #ahn = learn_alg.f_ahn
        #error = learn_alg.f_min

    if model == 'PFO':
        lowerAux = list(x.min(axis=0))
        lowerPi = [e for _ in range(n) for e in lowerAux]
        Lower = np.asarray(lowerPi)

        upperAux = list(x.max(axis=0))
        upperPi = [e for _ in range(n) for e in upperAux]
        Upper = np.asarray(upperPi)

        error = PFO(D=numVars, NP=30, R=3, b=0.5, c=0.5, nFES=iterations, Lower=Lower, Upper=Upper, x=x, y=y, n=n)
    return error #[error, ahn]
"""""
def CreateLinearCompound(n):
    #create structure of hydrocarbon compound
    Omega = 3*np.ones((n,1))
    Omega[1:-1] = 2
    B = np.ones((n-1,1))
    C = {'Omega': Omega, 'B':B}

    return C


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

    return SigmaSplit

#@nb.jit(nopython=True, parallel=True)
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


#@nb.jit(nopython=True, parallel=True)
def ComputeMoleculeParameters(Xi, Yi, Omegai):
    #Calculate phi-values in the series of the molecular function
    Phi = CH_X(Xi, Omegai)

    #Find the molecular parameters using least-squared error (LSE)
    H = np.linalg.lstsq(Phi, Yi)[0]

    #Determine error between the targets and estimates
    error = np.sqrt(np.mean(np.abs((Phi @ H)-Yi)**2))

    return H, error

#@nb.jit(nopython=True, parallel=True)
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


#@nb.jitclass(spec)
class ModelEval(object):
    def __init__(self, n, x0, y0):
        self.n = n
        # Create a prototype structure of the hydrocarbon compund
        self.C = CreateLinearCompound(self.n)
        self.x = x0
        self.y = y0
        # define lower bound of benchmark function
        #lowerAux = list(x0.min(axis=0))
        #lowerPi = [e for _ in range(self.n) for e in lowerAux]
        #self.Lower = lowerPi
        # define upper bound of benchmark function
        #upperAux = list(x0.max(axis=0))
        #upperPi = [e for _ in range(self.n) for e in upperAux]
        #self.Upper = upperPi

    # function which returns evaluate function
    def function(self):
        def evaluate(D, sol): #sol is the possible solution
            # Reshape the particle for matching the prototype structure
            sol = np.asarray(sol).reshape((self.n, self.x.shape[1]))
            # Split training data over molecules
            SigmaSplit = DataInMolecules(self.x, self.y, sol)

            # initialize procedure to fit molecular parameters
            error = np.zeros((self.n + 1, np.size(self.y, axis=1)))
            omega = self.C['Omega']
            H = [[] for i in range(self.n)]
            # Find molecular parameters
            for j in list(range(0, self.n)):
                Xi = SigmaSplit['X'][j]
                Yi = SigmaSplit['Y'][j]
                if Xi.any():
                    [H[j], error[j][:]] = ComputeMoleculeParameters(Xi, Yi, omega[j].__int__())
            ahn = {'H': H, 'Pi': sol, 'n': self.n, 'C': self.C}
            fitness = np.sum(error)
            return fitness, ahn
        return evaluate
"""""
