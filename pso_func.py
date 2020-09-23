# encoding=utf8
import numba as nb
from numba import prange
from models.AHNFunctions import CreateLinearCompound
from models.AHNFunctions import DataInMolecules
from models.AHNFunctions import ComputeMoleculeParameters
import numpy as np



def ParticleSwarmAlgorithm(D, vMin, vMax, nFES, w, NP, C1, C2, Lower, Upper, x, y, n):
    r"""Implementation of Particle Swarm Optimization algorithm.

    **Algorithm:** Particle Swarm Optimization algorithm

    **Date:** 2018

    **Authors:** Lucija Brezočnik, Grega Vrbančič, and Iztok Fister Jr.

    **License:** MIT

    **Reference paper:**
        Kennedy, J. and Eberhart, R. "Particle Swarm Optimization".
        Proceedings of IEEE International Conference on Neural Networks.
        IV. pp. 1942--1948, 1995.
    """

    #def __init__(, D, vMin, vMax, benchmark, nFES=1000, w=-0.35, NP=60, C1=-0.72, C2=2.029):
    r"""**__init__(, NP, D, nFES, C1, C2, w, vMin, vMax, benchmark)**.

    Arguments:
        NP {integer} -- population size

        D {integer} -- dimension of problem

        nFES {integer} -- number of function evaluations

        C1 {decimal} -- cognitive component

        C2 {decimal} -- social component

        w {decimal} -- inertia weight

        vMin {decimal} -- minimal velocity

        vMax {decimal} -- maximal velocity

        benchmark {object} -- benchmark implementation object

    """

    #benchmark = Utility().get_benchmark(benchmark)
    #NP = NP  # population size; number of search agents
    #D = D  # dimension of the problem
    #C1 = C1  # cognitive component
    #C2 = C2  # social component
    #w = w  # inertia weight
    #vMin = vMin  # minimal velocity
    #vMax = vMax  # maximal velocity
    #Lower = benchmark.Lower  # lower bound
    #Upper = benchmark.Upper  # upper bound
    #nFES = nFES  # number of function evaluations
    eval_flag = True  # evaluations flag
    evaluations = 0  # evaluations counter
    #Fun = benchmark.function()

    Solution = np.zeros((NP, D))  # positions of search agents
    Velocity = np.zeros((NP, D))  # velocities of search agents

    pBestFitness = np.zeros(NP)  # personal best fitness
    pBestFitness.fill(float("inf"))
    pBestSolution = np.zeros((NP, D))  # personal best solution

    gBestFitness = float("inf")  # global best fitness
    gBestSolution = np.zeros(D)  # global best solution
    gBestAhn = []

    C = CreateLinearCompound(n)
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

    @nb.jit(nopython=True, parallel=False)
    def init(NP, D, Solution, Upper, Lower):
        """Initialize positions."""
        for i in range(NP):
            for j in range(D):
                Solution[i][j] = np.random.random() * \
                    (Upper[j] - Lower[j]) + Lower[j]
        return Solution

    @nb.jit(nopython=True, parallel=False)
    def eval_true(evaluations, nFES, eval_flag):
        """Check evaluations."""
        if evaluations == nFES:
            eval_flag = False
        return eval_flag

    @nb.jit(nopython=True, parallel=False)
    def bounds(D, position, Lower, Upper):
        """Keep it within bounds."""
        for i in range(D):
            if position[i] < Lower[i]:
                position[i] = Lower[i]
            if position[i] > Upper[i]:
                position[i] = Upper[i]
        return position

    @nb.jit(nopython=True, parallel=True)
    def actual(NP, Velocity, Solution, w, C1, C2, pBestSolution, gBestSolution):
        for i in range(NP):
            for j in range(D):
                Velocity[i][j] = (w * Velocity[i][j]) + \
                                 (C1 * np.random.random() * (pBestSolution[i][j] - Solution[i][j])) + \
                                 (C2 * np.random.random() * (gBestSolution[j] - Solution[i][j]))

                if Velocity[i][j] < vMin:
                    Velocity[i][j] = vMin
                if Velocity[i][j] > vMax:
                    Velocity[i][j] = vMax

                Solution[i][j] = Solution[i][j] + \
                                 Velocity[i][j]
        return Solution, Velocity

    def move_particles(D, NP, nFES, Lower, Upper, Fun, Solution, Velocity, eval_flag, evaluations, pBestFitness, pBestSolution, gBestFitness, gBestSolution, x, y, C):
        """Move particles in search space."""
        Solution = init(NP=NP, D=D, Solution=Solution, Upper=Upper, Lower=Lower)

        while eval_flag is not False:
            for i in range(NP):
                Solution[i] = bounds(D=D, Lower=Lower, Upper=Upper, position=Solution[i])

                eval_flag = eval_true(evaluations=evaluations, nFES=nFES, eval_flag=eval_flag)
                if eval_flag is not True:
                    break

                Fit = Fun(D=D, sol=Solution[i], x=x, y=y, C=C)
                evaluations = evaluations + 1

                if Fit < pBestFitness[i]:
                    pBestFitness[i] = Fit
                    pBestSolution[i] = Solution[i]

                if Fit < gBestFitness:
                    gBestFitness = Fit
                    gBestSolution = Solution[i]
                    #gBestAhn = ahn

            Solution, Velocity = actual(NP=NP, Velocity=Velocity, Solution=Solution, w=w, C1=C1, C2=C2, pBestSolution=pBestSolution, gBestSolution=gBestSolution)
            """
            for i in range(NP):
                for j in range(D):
                    Velocity[i][j] = (w * Velocity[i][j]) + \
                        (C1 * np.random.random() * (pBestSolution[i][j] - Solution[i][j])) + \
                        (C2 * np.random.random() * (gBestSolution[j] - Solution[i][j]))

                    if Velocity[i][j] < vMin:
                        Velocity[i][j] = vMin
                    if Velocity[i][j] > vMax:
                        Velocity[i][j] = vMax

                    Solution[i][j] = Solution[i][j] + \
                        Velocity[i][j]
            """
        return gBestFitness

    def run(D, NP, nFES, Lower, Upper, Fun, Solution, Velocity, eval_flag, evaluations, pBestFitness, pBestSolution, gBestFitness, gBestSolution, x, y, C):
        """Run."""
        return move_particles(D, NP, nFES, Lower, Upper, Fun, Solution, Velocity, eval_flag, evaluations, pBestFitness, pBestSolution, gBestFitness, gBestSolution, x, y, C)

    gBestFitness = run(D, NP, nFES, Lower, Upper, Fun, Solution, Velocity, eval_flag, evaluations, pBestFitness, pBestSolution, gBestFitness, gBestSolution, x, y, C)
    return gBestFitness