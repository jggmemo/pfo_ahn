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

    # def __init__(, D, NP, nFES, A, r, Qmin, Qmax, benchmark):
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

    # benchmark = Utility().get_benchmark(benchmark)
    # D = D  # dimension
    # NP = NP  # population size
    # nFES = nFES  # number of function evaluations
    # A = A  # loudness
    # r = r  # pulse rate
    # Qmin = Qmin  # frequency min
    # Qmax = Qmax  # frequency max
    # Lower = benchmark.Lower  # lower bound
    # Upper = benchmark.Upper  # upper bound

    f_min = 0.0  # minimum fitness

    Lb = [0] * D  # lower bound
    Ub = [0] * D  # upper bound
    Q = [0] * NP  # frequency

    v = [[0 for _i in range(D)]
         for _j in range(NP)]  # velocity
    Sol = [[0 for _i in range(D)] for _j in range(
        NP)]  # population of solutions
    Fitness = [0] * NP  # fitness
    best = [0] * D  # best solution
    # f_ahn = [] # best ahn structure
    evaluations = 0  # evaluations counter
    eval_flag = True  # evaluations flag
    C = CreateLinearCompound(n)

    def Fun(D, sol, x, y, C):
        # @nb.jit(nopython=True, parallel=True)
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
            # ahn = {'H': H, 'Pi': sol, 'n': n, 'C': C}
            fitness = np.sum(error)
            return fitness

        fitness = evaluate(D=D, sol=sol, x=x, y=y, C=C)
        return fitness

    @nb.jit(nopython=True, parallel=False)
    def best_bat(NP, D, Fitness, best, Sol):
        """Find the best bat."""
        i = 0
        j = 0
        for i in range(NP):
            if Fitness[i] < Fitness[j]:
                j = i
        for i in range(D):
            best[i] = Sol[j][i]
        f_min = Fitness[j]

    @nb.jit(nopython=True, parallel=False)
    def eval_true(evaluations, nFES, eval_flag):
        """Check evaluations."""
        if evaluations == nFES:
            eval_flag = False
        return eval_flag

    @nb.jit(nopython=True, parallel=False)
    def init_bat(D, Lower, Upper, Q, v, Sol, Fitness, evaluations):
        """Initialize population."""
        for i in range(D):
            Lb[i] = Lower[i]
            Ub[i] = Upper[i]

        for i in range(NP):
            Q[i] = 0
            for j in range(D):
                rnd = np.random.uniform(0, 1)
                v[i][j] = 0.0
                Sol[i][j] = Lb[j] + (Ub[j] - Lb[j]) * rnd
            Fitness[i] = Fun(D, Sol[i])
            evaluations = evaluations + 1
        best_bat()

    @nb.jit(nopython=True, parallel=False)
    def simplebounds(D, Lower, Upper, val):
        """Keep it within bounds."""
        for i in prange(D):
            if val[i] < Lower[i]:
                val[i] =lower
            if val[i] > upper:
                val[i] = upper
        return val

    def move_bat():
        """Move bats in search space."""
        S = [[0.0 for i in range(D)] for j in range(NP)]

        init_bat()

        while eval_flag is not False:
            for i in range(NP):
                # aqui deben ir los bounds

                # se valida eval_flag
                # Se calcula Fit
                # evaluaciones + 1
                # jerarquias
                # Cambio de posiciones
                rnd = np.random.uniform(0, 1)
                Q[i] = Qmin + (Qmin - Qmax) * rnd
                for j in range(D):
                    v[i][j] = v[i][j] + (Sol[i][j] -
                                         best[j]) * Q[i]
                    S[i][j] = Sol[i][j] + v[i][j]

                    S[i][j] = simplebounds(S[i][j], Lb[j], Ub[j])

                rnd = np.random.random()

                ### Select a solution among the best solutions
                ### Generate a local solution around the selected best solution
                if rnd > r:
                    for j in range(D):
                        S[i][j] = best[j] + 0.001 * np.random.gauss(0, 1)
                        S[i][j] = simplebounds(S[i][j], Lb[j],
                                               Ub[j])

                eval_true()

                if eval_flag is not True:
                    break
                ### Generate a new solution by flying randomly
                Fnew = Fun(D, S[i])
                evaluations = evaluations + 1
                rnd = np.random.random()

                if (Fnew <= Fitness[i]) and (rnd < A):
                    for j in range(D):
                        ###Accept a new solution
                        Sol[i][j] = S[i][j]
                    Fitness[i] = Fnew

                if Fnew <= f_min:
                    for j in range(D):
                        best[j] = S[i][j]
                    f_min = Fnew
                    # f_ahn = ahn_new

        return f_min

    def run():
        """Run algorithm with initialized parameters.

        Return {decimal} - best
        """
        return move_bat()
