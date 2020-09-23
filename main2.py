from models.AHN import train as AHN
import numpy as np
from tqdm import tqdm
from time import time
from pandas import DataFrame
from statistics import mean
from statistics import stdev

run_data = []

for reg in [1000, 10000, 100000]: #excecutes 1000, 10000, 100000 registries per iteration
    print("Solving with {0} registries".format(reg))
    x = (2 * np.random.rand(reg, 1)) - 1
    x = np.msort(x)
    y = ((x < 0.1).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.arctan(x * np.pi)) + (
                ((x >= 0.1).astype(int) * (x < 0.6).astype(int)) * (
                    0.05 * np.random.rand(reg, 1) + np.sin(x * np.pi))) + (
                    ((x >= 0.6).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.cos(x * np.pi)))

    #x = (2 * np.random.rand(reg, 1)) - 1
    #x = np.msort(x)
    #y = ((x < 0.1).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.arctan(x * np.pi)) + (((x >= 0.1).astype(int) * (x < 0.6).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.sin(x * np.pi))) + (((x >= 0.6).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.cos(x * np.pi)))
    #t = np.arange(0, (reg/100), 0.01)
    #x = np.ones((t.__len__(), 2))
    #x[:, 0] = np.cos(t[:])
    #x[:, 1] = t[:]
    #y = np.sin(t[:])
    #y = y.reshape(t.__len__(), 1)

    for molNum in tqdm([3, 10, 20]):
        fail_count = 0
        print("Training {0} molecules".format(molNum))
        settings = {
            'molecules': molNum,
            'learningRate': 0.05,
            'iterations': 100,
            'learnAlg': 'PFO'
        }
        rms_results = []
        time_results = []
        for rep_exp in tqdm(range(2)):
            start = time()
            error = AHN(x, y, n=molNum, model=settings['learnAlg'], iterations=settings['iterations'])
            end = time()
            exec_time = float('%.3f'%(end - start))
            print('-----------------------------------------------')
            print('AHN-model trained with {0} molecules, {3} training iterations and {1} registries in {2} seconds'.format(molNum, reg, exec_time, settings['iterations']))
                #ViewResults(ysim, y, mode='3D', x=x)
            rms_results.append(error)
            time_results.append(exec_time)

        rms_mean = mean(rms_results)
        rms_stDev = stdev(rms_results)
        time_mean = mean(time_results)
        time_stDev = stdev(time_results)

        run_data.append({
            'Reg_Number': reg,
            'Molecules': molNum,
            'RMSE_Mean': rms_mean,
            'RMSE_stDev': rms_stDev,
            'ET_Mean': time_mean,
            'ET_stDev': time_stDev
        })
        #with open('results_{0}.csv'.format(settings['learnAlg'])) as file:
            #run_data.to_csv(file, mode='a', header=file.tell()==0, columns=['Reg_Number', 'Molecules', 'RMSE_Mean', 'RMSE_stDev', 'ET_Mean', 'ET_stDev'])
        #run_data.to_csv('results_{0}.csv'.format(settings['learnAlg']), mode='a', columns=['Reg_Number', 'Molecules', 'RMSE_Mean', 'RMSE_stDev', 'ET_Mean', 'ET_stDev'])
        print('first')

finalResults = DataFrame.from_dict(run_data)
finalResults.to_csv('results_{0}_2Dim.csv'.format(settings['learnAlg']), index=False)
print('finished')