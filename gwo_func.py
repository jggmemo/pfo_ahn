import numba as nb
from numba import prange
from models.AHNFunctions import CreateLinearCompound
from models.AHNFunctions import DataInMolecules
from models.AHNFunctions import ComputeMoleculeParameters
import numpy as np


def GreyWolfOptimizer(D, NP, nFES, Lower, Upper, x, y, n):
    #ModelEval_type.define()
    """Implementation of Grey wolf optimizer.

    **Algorithm:** Grey wolf optimizer

    **Date:** 2018

    **Author:** Iztok Fister Jr.

    **License:** MIT

    **Reference paper:**
        Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis.
        "Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61.
        & Grey Wold Optimizer (GWO) source code version 1.0 (MATLAB) from MathWorks
    """

    #def init(D, NP, nFES, benchmark, Lower, Upper, Fun):
    """**__init__(self, D, NP, nFES, benchmark)**.

    Arguments:
        D {integer} -- dimension of problem

        NP {integer} -- population size

        nFES {integer} -- number of function evaluations

        benchmark {object} -- benchmark implementation object

    Raises:
        TypeError -- Raised when given benchmark function which does not exists.

    """
    #benchmark = benchmark
    #D = D  # dimension of the problem Centers
    #NP = NP  # population size; number of search agents
    #nFES = nFES  # number of function evaluations
    #Lower = Lower  # lower bound
    #Upper = Upper  # upper bound
    #Fun = benchmark.function()

    #Positions = np.array([[0 for _i in range(D)] for _j in range(NP)])
    Positions = np.zeros((NP,D))
    eval_flag = True  # evaluations flag
    evaluations = 0  # evaluations counter

    Alpha_pos = [0] * D  # init of alpha
    Alpha_score = float("inf")
    #Alpha_ahn = []

    Beta_pos = [0] * D  # init of beta
    Beta_score = float("inf")

    Delta_pos = [0] * D  # init of delta
    Delta_score = float("inf")
    #from models.AHN import ModelEval.function as Fun
    # function which returns evaluate function
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
    def initialization(D, NP, Lower, Upper, Positions):
        """Initialize positions."""
        for i in prange(0, NP):
            for j in prange(0, D):
                Positions[i, j] = np.random.rand() * (Upper[j] - Lower[j]) + Lower[j]
        return Positions

    @nb.jit(nopython=True, parallel=False)
    def eval_true(evaluations, nFES, eval_flag):
        """Check evaluations."""
        if evaluations == nFES:
            eval_flag = False
        return eval_flag

    @nb.jit(nopython=True, parallel=False)
    def bounds(D, Lower, Upper, position):
        """Keep it within bounds."""
        for i in prange(D):
            if position[i] < Lower[i]:
                position[i] = Lower[i]
            if position[i] > Upper[i]:
                position[i] = Upper[i]
        return position

    # pylint: disable=too-many-locals
    @nb.jit(nopython=True, parallel=True)
    def change_pos(NP, D, a, Alpha_pos, Positions, Beta_pos, Delta_pos):
        for i in range(NP):
            # Para cada agente aqui se mueven los Centros
            for j in range(D):
                r1 = np.random.rand()
                r2 = np.random.rand()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(
                    C1 * Alpha_pos[j] - Positions[i][j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = np.random.rand()
                r2 = np.random.rand()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * Beta_pos[j] - Positions[i][j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = np.random.rand()
                r2 = np.random.rand()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(
                    C3 * Delta_pos[j] - Positions[i][j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i][j] = (X1 + X2 + X3) / 3
        return Positions

    #@nb.jit(nopython=True, parallel=True)
    def move(D, NP, nFES, Lower, Upper, Fun, Positions, eval_flag, evaluations, Alpha_pos, Alpha_score, Beta_pos, Beta_score, Delta_pos, Delta_score, x, y, C):

        """Move wolves in search space."""

        Positions = initialization(D=D, NP=NP, Lower=Lower, Upper=Upper, Positions=Positions)
        #for i in prange(NP):
        #    for j in prange(D):
        #        Positions[i, j] = np.random.rand() * (Upper[j] - Lower[j]) + Lower[j]

        while eval_flag is not False:

            for i in range(NP):
                Positions[i] = bounds(D=D, Lower=Lower, Upper=Upper, position=Positions[i])
                #for z in prange(D):
                #    if Positions[i,z] < Lower[z]:
                #        Positions[i,z] = Lower[z]
                #    if Positions[i,z] > Upper[z]:
                #        Positions[i,z] = Upper[z]

                eval_flag = eval_true(evaluations, nFES, eval_flag)
                #if evaluations == nFES:
                #    eval_flag = False

                if eval_flag is not True:
                    break

                Fit = Fun(D=D, sol=Positions[i], x=x, y=y, C=C) #evaluacion
                #Fit = np.random.random()
                evaluations = evaluations + 1

                if Fit < Alpha_score:
                    Alpha_score = Fit
                    Alpha_pos = Positions[i]
                    #Alpha_ahn = ahn

                if ((Fit > Alpha_score) and (Fit < Beta_score)):
                    Beta_score = Fit
                    Beta_pos = Positions[i]

                if ((Fit > Alpha_score) and (Fit > Beta_score) and
                        (Fit < Delta_score)):
                    Delta_score = Fit
                    Delta_pos = Positions[i]

            a = 2 - evaluations * ((2) / nFES)
            Positions = change_pos(NP=NP, D=D, a=a, Alpha_pos=Alpha_pos, Beta_pos=Beta_pos, Delta_pos=Delta_pos, Positions=Positions)
        """
            for i in range(NP):

                #Para cada agente aqui semueven los Centros
                for j in range(D):

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(
                        C1 * Alpha_pos[j] - Positions[i][j])
                    X1 = Alpha_pos[j] - A1 * D_alpha

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * Beta_pos[j] - Positions[i][j])
                    X2 = Beta_pos[j] - A2 * D_beta

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(
                        C3 * Delta_pos[j] - Positions[i][j])
                    X3 = Delta_pos[j] - A3 * D_delta

                    Positions[i][j] = (X1 + X2 + X3) / 3
        """

        return Alpha_score

    def run(D, NP, nFES, Lower, Upper, Fun, Positions, eval_flag, evaluations,
            Alpha_pos, Alpha_score, Beta_pos, Beta_score, Delta_pos, Delta_score, x, y, C):
        """Run."""
        return move(D=D, NP=NP, nFES=nFES, Lower=Lower, Upper=Upper, Fun=Fun, Positions=Positions, eval_flag=eval_flag,
                    evaluations=evaluations, Alpha_pos=Alpha_pos, Alpha_score=Alpha_score, Beta_pos=Beta_pos,
                    Beta_score=Beta_score, Delta_pos=Delta_pos, Delta_score=Delta_score, x=x, y=y, C=C)

    Alpha_score = run(D=D, NP=NP, nFES=nFES, Lower=Lower, Upper=Upper, Fun=Fun, Positions=Positions, eval_flag=eval_flag,
                      evaluations=evaluations, Alpha_pos=Alpha_pos, Alpha_score=Alpha_score, Beta_pos=Beta_pos,
                      Beta_score=Beta_score, Delta_pos=Delta_pos, Delta_score=Delta_score, x=x, y=y, C=C)
    return Alpha_score
