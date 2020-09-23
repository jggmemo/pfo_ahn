import numba as nb
from numba import prange
from models.AHNFunctions import CreateLinearCompound
from models.AHNFunctions import DataInMolecules
from models.AHNFunctions import ComputeMoleculeParameters
import numpy as np

def BatAlgorithm(A, r, Qmin, Qmax, D, NP, nFES, Lower, Upper, x, y, n):
    r"""Implementation of Bat algorithm.

    **Algorithm:** Bat algorithm

    **Date:** 2015

    **Authors:** Iztok Fister Jr. and Marko Burjek

    **License:** MIT

    **Reference paper:**
        Yang, Xin-She. "A new metaheuristic bat-inspired algorithm."
        Nature inspired cooperative strategies for optimization (NICSO 2010).
        Springer, Berlin, Heidelberg, 2010. 65-74.
    """

    #def __init__(, D, NP, nFES, A, r, Qmin, Qmax, benchmark):
    r"""**__init__(, D, NP, nFES, A, r, Qmin, Qmax, benchmark)**.

    Arguments:
        D {integer} -- dimension of problem

        NP {integer} -- population size

        nFES {integer} -- number of function evaluations

        A {decimal} -- loudness

        r {decimal} -- pulse rate

        Qmin {decimal} -- minimum frequency

        Qmax {decimal } -- maximum frequency

        benchmark {object} -- benchmark implementation object

    Raises:
        TypeError -- Raised when given benchmark function which does not exists.

    """
    """
        @nb.jit(nopython=True, parallel=False)
        def Fun(D, sol, x, y, C):
            #@nb.jit(nopython=True, parallel=True)
            def evaluate(D, sol, x, y, C):  # sol is the possible solution
                # Reshape the particle for matching the prototype structure
                sol = np.asarray(sol).reshape((n, x.shape[1]))
                # Split training data over molecules
                SigmaSplitX, SigmaSplitY = DataInMolecules(x, y, sol)
                # initialize procedure to fit molecular parameters
                error = np.zeros((n + 1, np.size(y, axis=1)))
                omega = C
                H = [[0] for i in range(n)]
                # Find molecular parameters
                for j in list(range(0, n)):
                    Xi = SigmaSplitX[j]
                    Yi = SigmaSplitY[j]
                    if Xi.any():
                        [H[j], error[j][:]] = ComputeMoleculeParameters(Xi, Yi, int(omega[j]))
                #ahn = {'H': H, 'Pi': sol, 'n': n, 'C': C}
                fitness = np.sum(error)
                return fitness
            fitness = evaluate(D=D, sol=sol, x=x, y=y, C=C)
            return fitness
    """

    #benchmark = Utility().get_benchmark(benchmark)
    #D = D  # dimension
    #NP = NP  # population size
    #nFES = nFES  # number of function evaluations
    #A = A  # loudness
    #r = r  # pulse rate
    #Qmin = Qmin  # frequency min
    #Qmax = Qmax  # frequency max
    #Lower = benchmark.Lower  # lower bound
    #Upper = benchmark.Upper  # upper bound

    f_min = 0.0  # minimum fitness

    Lb = np.array([0] * D)  # lower bound
    Ub = np.array([0] * D)  # upper bound
    Q = np.array([0] * NP)  # frequency

    v = np.array([[0 for _i in range(D)] for _j in range(NP)])  # velocity
    Sol = np.array([[0 for _i in range(D)] for _j in range(NP)])  # population of solutions
    Fitness = np.array([0] * NP)  # fitness
    best = np.array([0] * D)  # best solution
    #f_ahn = [] # best ahn structure
    evaluations = 0  # evaluations counter
    eval_flag = True  # evaluations flag
    C = CreateLinearCompound(n)

    @nb.jit
    def Fun(D, sol, x, y, C):
        # @nb.jit(nopython=True, parallel=True)
        #def evaluate(D, sol, x, y, C):  # sol is the possible solution
        # Reshape the particle for matching the prototype structure
        sol = np.asarray(sol).reshape((n, x.shape[1]))
        # Split training data over molecules
        SigmaSplitX, SigmaSplitY = DataInMolecules(x, y, sol)
        # initialize procedure to fit molecular parameters
        error = np.zeros((n + 1, np.size(y, axis=1)))
        omega = C
        H = [[0] for i in range(n)]
        # Find molecular parameters
        for j in list(range(0, n)):
            Xi = SigmaSplitX[j]
            Yi = SigmaSplitY[j]
            if Xi.any():
                [H[j], error[j][:]] = ComputeMoleculeParameters(Xi, Yi, int(omega[j]))
        # ahn = {'H': H, 'Pi': sol, 'n': n, 'C': C}
        fitness = np.sum(error)
            #return fitness
        return fitness

    #@nb.jit(nopython=True, parallel=False)
    def best_bat(NP, D, Fitness, Sol): #best
        """Find the best bat."""
        z = 0
        j = 0
        for z in range(NP):
            if Fitness[z] < Fitness[j]:
                j = z
        for z in range(D):
            best[z] = Sol[j][z]
        f_min = Fitness[j]

        return f_min, best

    @nb.jit(nopython=True, parallel=False)
    def eval_true(evaluations, nFES, eval_flag):
        """Check evaluations."""
        if evaluations == nFES:
            eval_flag = False
        return eval_flag

    @nb.jit(nopython=True, parallel=False)
    def init_bat(D, Lower, Upper, Q, v, Sol, Lb, Ub):
        """Initialize population."""
        for i in range(D):
            Lb[i] = Lower[i]
            Ub[i] = Upper[i]

        for i in range(NP):
            Q[i] = 0
            for j in range(D):
                rnd = np.random.random()
                v[i][j] = 0.0
                Sol[i][j] = Lb[j] + (Ub[j] - Lb[j]) * rnd
            #Fitness[i] = Fun(D, Sol[i])
            #evaluations = evaluations + 1
        #best_bat()
        return Sol#, Fitness, v

    @nb.jit(nopython=True, parallel=False)
    def simplebounds(D, Lower, Upper, val):
        """Keep it within bounds."""
        for i in prange(D):
            if val[i] < Lower[i]:
                val[i] = Lower[i]
            if val[i] > Upper[i]:
                val[i] = Upper[i]
        return val

    @nb.jit(nopython=True, parallel=True)
    def move_bat(D, NP, nFES, Lower, Upper, Fun, Sol, eval_flag, evaluations, Q, v, Lb, Ub, Qmin, Qmax):
        """Move bats in search space."""
        S = [[0.0 for i in range(D)] for j in range(NP)]

        auxSol = init_bat(D=D, Lower=Lower, Upper=Upper, Q=Q, v=v, Sol=Sol, Lb=Lb, Ub=Ub)
        Sol = auxSol
        #f_min, best = best_bat(NP, D, Fitness, Sol)
        """Find the best bat."""
        z = 0
        j = 0
        for z in range(NP):
            if Fitness[z] < Fitness[j]:
                j = z
        for z in range(D):
            best[z] = Sol[j][z]
        f_min = Fitness[j]

        while eval_flag is not False:
            for i in range(NP):
                # aqui deben ir los bounds
                auxSol2 = simplebounds(D=D, Lower=Lower, Upper=Upper, val=Sol[i])
                Sol[i] = auxSol2
                # se valida eval_flag
                #eval_flag = eval_true(evaluations=evaluations, nFES=nFES, eval_flag=eval_flag)
                #if eval_flag is not True:
                #    break
                #Se calcula Fit
                Fitness[i] = Fun(D=D, sol=Sol[i], x=x, y=y, C=C)
                #evaluations = evaluations + 1
                #jerarquias

                # Cambio de posiciones
                rnd = np.random.rand()
                Q[i] = Qmin + (Qmin - Qmax) * rnd
                for j in range(D):
                    v[i][j] = v[i][j] + (Sol[i][j] -
                                                   best[j]) * Q[i]
                    S[i][j] = Sol[i][j] + v[i][j]

                    S[i] = simplebounds(D=D, Lower=Lb[j], Upper=Ub[j], val=S[i])

                rnd = np.random.random()

                if rnd > r:
                    for j in range(D):
                        S[i][j] = best[j] + 0.001 * np.random.rand()
                        S[i] = simplebounds(D=D, Lower=Lb[j], Upper=Ub[j], val=S[i])

                eval_flag = eval_true(evaluations=evaluations, nFES=nFES, eval_flag=eval_flag)

                if eval_flag is not True:
                    break

                Fnew = Fun(D=D, sol=Sol[i], x=x, y=y, C=C)
                evaluations = evaluations + 1
                rnd = np.random.random()

                if (Fnew <= Fitness[i]) and (rnd < A):
                    for j in range(D):
                        Sol[i][j] = S[i][j]
                    Fitness[i] = Fnew

                if Fnew <= f_min:
                    for j in range(D):
                        best[j] = S[i][j]
                    f_min = Fnew
                    #f_ahn = ahn_new

        return f_min

    def run(D, NP, nFES, Lower, Upper, Fun, Sol, eval_flag, evaluations, Q, v, Lb, Ub, Qmin, Qmax):
        """Run algorithm with initialized parameters.

        Return {decimal} - best
        """
        return move_bat(D=D, NP=NP, nFES=nFES, Lower=Lower, Upper=Upper, Fun=Fun, Sol=Sol, eval_flag=eval_flag, evaluations=evaluations, Q=Q, v=v, Lb=Lb, Ub=Ub, Qmin=Qmin, Qmax=Qmax)


    f_min = run(D=D, NP=NP, nFES=nFES, Lower=Lower, Upper=Upper, Fun=Fun, Sol=Sol, eval_flag=eval_flag, evaluations=evaluations, Q=Q, v=v, Lb=Lb, Ub=Ub, Qmin=Qmin, Qmax=Qmax)
    #Sol = init_bat(D=D, Lower=Lower, Upper=Upper, Q=Q, v=v, Lb=Lb, Ub=Ub)
    return f_min
