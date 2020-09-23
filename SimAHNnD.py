import numpy as np
from SimDataInMolecules import SimDataInMolecules
from models.AHN import CH_X



def SimAHNnD(ahn, X, Y0):

    if ahn['H'].__len__() != ahn['n']:
        newH = [[] for i in range(ahn['n'])]
        init = 0
        count = 0
        for j in newH:
            j.append((ahn['H'][init:(X.shape[1]*(int(ahn['C']['Omega'][count] + 1 + (init/X.shape[1]))))]))
            init += X.shape[1]*(int(ahn['C']['Omega'][count] + 1))
            count += 1
        ahn['H'] = [np.stack(h) for h in newH]
        count = 0
        for item in ahn['H']:
            ahn['H'][count] = item.reshape((item.shape[1], item.shape[0]))
            count += 1
        ahn['Pi'] = np.stack(ahn['Pi']).reshape((ahn['n'], X.shape[1]))

    #Initial Statements
    Yapprox = np.array([0])
    indexes = np.array([0])
    #Unflatten AHN-structure
    Hsim = ahn['H']
    posMoleculesSim = ahn['Pi']
    n = ahn['n']
    C = ahn['C']

    #Distribute data over molecules
    SigmaSplit = SimDataInMolecules(X=X, posMolecules=ahn['Pi'])

    #Evaluate AHN-model
    for i in list(range(0,n)):
        XiSim = SigmaSplit['X'][i]
        if np.any(XiSim):
            indexesi = SigmaSplit['index'][i]
            ki = C['Omega'][i][0]
            Phi = CH_X(XiSim, int(ki))
            Ym = np.matmul(Phi, Hsim[i])
            #YmI = np.matmul(Phi, Hsim[i])
            Yapprox=np.vstack((Yapprox,Ym))
            indexes=np.hstack((indexes,indexesi))


    Yapprox = Yapprox[1:]
    indexes = indexes[1:]

    I = indexes.argsort(axis=0)
    Y = Yapprox[I]

    return Y, SigmaSplit, ahn
