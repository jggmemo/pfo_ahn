import numpy as np

def SimDataInMolecules(X, posMolecules):

    numSamples = X.shape[0]
    n=np.array(posMolecules).shape[0]
    #n = posMolecules.shape[0]
    SigmaSplit = {'X': [], 'index': []}
    dist = np.zeros((numSamples,n))

    for i in list(range(0,n)):
        SigmaSplit['X'].append([])
        SigmaSplit['index'].append([])

        for j in list(range(0,numSamples)):
            dist[j,i] = np.sqrt(np.sum((X[j][:] - posMolecules[i][:])**2))

    moleculeNumber = np.argsort(dist, axis=1)

    for j in list(range(0, numSamples)):
        SigmaSplit['X'][moleculeNumber[j][0]].append(X[j])
        SigmaSplit['index'][moleculeNumber[j][0]].append(j)

    for g in list(range(0,n)):
        SigmaSplit['X'][g] = np.asarray(SigmaSplit['X'][g])
        SigmaSplit['index'][g] = np.asarray(SigmaSplit['index'][g])


    return SigmaSplit