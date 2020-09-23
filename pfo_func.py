import numba as nb
from numba import prange
from models.AHNFunctions import CreateLinearCompound
from models.AHNFunctions import DataInMolecules
from models.AHNFunctions import ComputeMoleculeParameters
import numpy as np

def PiranhaFeastOptimizer(D, NP, R, b, c, nFES, Lower, Upper, x, y, n):
    #Initialize parameters
    R_matrix = np.zeros((R,D))
    R_score = np.full((R), float("inf"))
    R_rad = np.full((R), 0.10)
    V = np.random.rand(NP,D)
    eval_flag = True  # evaluations flag
    evaluations = 0  # evaluations counter
    P = np.zeros((NP, D))
    P_score = np.zeros((NP, 1))



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

    #@nb.jit(nopython=True, parallel=False)
    def initialization(D, NP, Lower, Upper, P):
        #P = np.zeros((NP, D))
        #P_score = np.zeros((NP, 1))
        """Initialize positions."""
        for i in range(0, NP):
            for j in range(0, D):
                P[i, j] = np.random.rand() * (Upper[j] - Lower[j]) + Lower[j]
        return P

    #@nb.jit(nopython=True, parallel=False)
    def eval_true(evaluations, nFES, eval_flag):
        """Check evaluations."""
        if evaluations == nFES:
            eval_flag = False
        return eval_flag

    #@nb.jit(nopython=True, parallel=False)
    def bounds(D, Lower, Upper, position):
        """Keep it within bounds."""
        for i in range(D):
            if position[i] < Lower[i]:
                position[i] = Lower[i]
            if position[i] > Upper[i]:
                position[i] = Upper[i]
        return position

    """
    def attack(R, P, b, V):
        V_actual = V + np.random.rand() * b * (R - P)
        P_actual = P + V_actual
        return V_actual, P_actual

    def navigate(c, d, R):
        t=np.random.uniform(-1,1)
        M = d*np.exp(c*t)*np.cos(2*np.pi*t) + R
        return M
    """
    @nb.jit(nopython=True, parallel=True)
    def actual_pos(NP, P, V, R_matrix, R_rad, c, Fit, R_score):
        for i in prange(NP):
            #dist_matrix = Fit[i] - R_score #np.asarray([Fit - R_score[i] for i in range(len(R_matrix))])
            dist_matrix = np.asarray([np.linalg.norm(P[i] - R_matrix[j]) for j in prange(len(R_matrix))])  # Distancia entre pirañas y presas
            R_ind = np.argmin(dist_matrix)  # Identifica la presa mas cercana a la piraña i
            if (dist_matrix[R_ind] < R_rad[R_ind]) and (dist_matrix[R_ind] != 0):
                V[i] = V[i] + np.random.rand()*b*(R_matrix[R_ind] - P[i])
                P[i] = P[i] + V[i]
                #V[i], P[i] = attack(R=R_matrix[R_ind], P=P[i], b=b, V=V[i])
                R_rad[R_ind] = R_rad[R_ind] * (0.90 ** (R_ind + 1))
            else:
                acum = np.sum(dist_matrix)
                Prob_nj = np.divide(acum, dist_matrix)#np.asarray([(acum / d) for d in dist_matrix])
                acum = np.sum(Prob_nj)
                Prob_nj = np.divide(Prob_nj, acum)#Prob_nj/acum
                #Prob_nj = [w / acum for w in Prob_nj]
                #selectR = np.random.choice(a=list(range(len(R_matrix))), p=Prob_nj)
                selectR = np.searchsorted(np.cumsum(Prob_nj), np.random.rand(1))[0]
                t = np.random.uniform(-1,1)
                P[i] = dist_matrix[selectR] * np.exp(c*t)*np.cos(2*np.pi*t) + R_matrix[selectR]
                #P[i] = navigate(c=c, d=dist_matrix[selectR], R=R_matrix[selectR])

        return P, V
    """
    #@nb.jit(nopython=True, parallel=True)
    def actual_pos2(NP, P, V, R_matrix, R_rad, c, Fit, R_score):
        for i in range(NP):
            R_ind = 0
            for j in prange(len(R_score)):
                if (Fit[i] < np.multiply(R_score[j],1+R_rad[j])):
                    R_ind = j
                    break

            #dist_matrix = Fit[i] - R_score #np.asarray([Fit - R_score[i] for i in range(len(R_matrix))])
            #dist_matrix = np.asarray([np.linalg.norm(P[i] - R_matrix[j]) for j in range(len(R_matrix))])  # Distancia entre pirañas y presas
            #R_ind = np.argmax(dist_matrix)  # Identifica la presa mas cercana a la piraña i
            if R_ind != 0:
                V[i] = V[i] + np.random.rand()*b*(R_matrix[R_ind] - P[i])
                P[i] = P[i] + V[i]
                #V[i], P[i] = attack(R=R_matrix[R_ind], P=P[i], b=b, V=V[i])
                R_rad[R_ind] = (R_rad[R_ind]) * (0.90 ** (R_ind + 1))
            else:
                dist_matrix = np.asarray([np.linalg.norm(P[i] - R_matrix[j]) for j in range(len(R_matrix))])
                acum = np.sum(dist_matrix)
                Prob_nj = np.divide(acum, dist_matrix)#np.asarray([(acum / d) for d in dist_matrix])
                acum = np.sum(Prob_nj)
                Prob_nj = np.divide(Prob_nj, acum)#Prob_nj/acum
                #Prob_nj = [w / acum for w in Prob_nj]
                #selectR = np.random.choice(a=list(range(len(R_matrix))), p=Prob_nj)
                selectR = np.searchsorted(np.cumsum(Prob_nj), np.random.rand(1))[0]
                t = np.random.uniform(-1,1)
                P[i] = dist_matrix[selectR] * np.exp(c*t)*np.cos(2*np.pi*t) + R_matrix[selectR]
                #P[i] = navigate(c=c, d=dist_matrix[selectR], R=R_matrix[selectR])

        return P, V
    """
    def move(R_matrix, R_score, R_rad, V, eval_flag, evaluations, c, P, P_score):
        P = initialization(D=D, NP=NP, Lower=Lower, Upper=Upper, P=P)   #Inicialización
        #P_score[i] = Fun(D=D, sol=P[i], x=x, y=y, C=C)
        while eval_flag is not False:
            for i in range(NP):
                P[i] = bounds(D=D, Lower=Lower, Upper=Upper, position=P[i]) #Valida limites
                eval_flag = eval_true(evaluations=evaluations, nFES=nFES, eval_flag=eval_flag)
                if eval_flag is not True:
                    break
                Fit = Fun(D=D, sol=P[i], x=x, y=y, C=C) #Fitness de la solucion
                P_score[i] = Fit
                evaluations = evaluations + 1
                ####    Identifica presas
                for z in range(len(R_matrix)):
                    if Fit <= R_score[z]:
                        R_score[z] = Fit
                        R_matrix[z] = P[i]
                        R_rad[z] = 0.10
                        break
                #### Si es la primera evaluacion, ordenar R
                if evaluations == (len(R_matrix)):
                    ind = np.argsort(R_score)
                    R_score = R_score[ind]
                    R_matrix = R_matrix[ind]
                    R_rad = R_rad[ind]
                """
                dist_matrix = np.asanyarray([np.linalg.norm(P[i] - R_matrix[j]) for j in range(len(R_matrix))])   #Distancia entre pirañas y presas
                R_ind = np.argmin(dist_matrix) # Identifica la presa mas cercana a la piraña i
                if dist_matrix[R_ind] < R_rad[R_ind]:
                    V[i], P[i] = attack(R=R_matrix[R_ind], P=P[i], b=b, V=V[i])
                    R_rad[R_ind] = R_rad[R_ind]*0.95**(R_ind+1)

                else:
                    acum = sum(dist_matrix)
                    Prob_nj = np.asanyarray([(acum/d) for d in dist_matrix])
                    acum = sum(Prob_nj)
                    Prob_nj = [w / acum for w in Prob_nj]
                    selectR = np.random.choice(a=list(range(len(R_matrix))), p=Prob_nj)
                    P[i] = navigate(c=c, d=dist_matrix[selectR], R=R_matrix[selectR])
                """
            P, V = actual_pos(NP=NP, P=P, V=V, R_matrix=R_matrix, R_rad=R_rad, c=c, Fit=P_score, R_score=R_score)
        Best_pray = R_score[0]

        return Best_pray

    def run(R_matrix, R_score, R_rad, V, eval_flag, evaluations, c, P, P_score):
        return move(R_matrix, R_score, R_rad, V, eval_flag, evaluations, c, P, P_score)

    Best_pray = run(R_matrix, R_score, R_rad, V, eval_flag, evaluations, c, P, P_score)

    return Best_pray

