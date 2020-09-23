from models.AHN import train as AHN
import numpy as np
from predict import predict
from ViewResults import ViewResults
from tqdm import tqdm
from time import time
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from statistics import mean
from statistics import stdev

"""""
reg = 1000
x = (2 * np.random.rand(reg, 1)) - 1
x = np.msort(x)
y = ((x < 0.1).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.arctan(x * np.pi)) + (
        ((x >= 0.1).astype(int) * (x < 0.6).astype(int)) * (
        0.05 * np.random.rand(reg, 1) + np.sin(x * np.pi))) + (
            ((x >= 0.6).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.cos(x * np.pi)))

t=np.arange(0,15, 0.01)
x=np.ones((t.__len__(), 2))
x[:,0]=np.cos(t[:])
x[:,1]=t[:]
y=np.sin(t[:])
y = y.reshape(t.__len__(),1)

error, ahn = AHN(x, y, n=3, model='PSO')
ysim, _, _, _ = predict(ahn,x, y)
ViewResults(ysim, y)

print(error)
print(ahn)
"""""
run_data = []

for reg in [1000, 10000, 100000]: #excecutes 1000, 10000, 100000 registries per iteration
    print("Solving with {0} registries".format(reg))
    #x = (2 * np.random.rand(reg, 1)) - 1
    #x = np.msort(x)
    #y = ((x < 0.1).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.arctan(x * np.pi)) + (
    #            ((x >= 0.1).astype(int) * (x < 0.6).astype(int)) * (
    #                0.05 * np.random.rand(reg, 1) + np.sin(x * np.pi))) + (
    #                ((x >= 0.6).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.cos(x * np.pi)))
    x = (2 * np.random.rand(reg, 1)) - 1
    x = np.msort(x)
    y = ((x < 0.1).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.arctan(x * np.pi)) + (((x >= 0.1).astype(int) * (x < 0.6).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.sin(x * np.pi))) + (((x >= 0.6).astype(int)) * (0.05 * np.random.rand(reg, 1) + np.cos(x * np.pi)))

    t = np.arange(0, (reg/100), 0.01)
    x = np.ones((t.__len__(), 2))
    x[:, 0] = np.cos(t[:])
    x[:, 1] = t[:]
    y = np.sin(t[:])
    y = y.reshape(t.__len__(), 1)

    for molNum in tqdm([10]):
        fail_count = 0
        print("Training {0} molecules".format(molNum))
        settings = {
            'molecules': molNum,
            'learningRate': 0.05,
            'iterations': 100,
            'learnAlg': 'GWO'
        }
        rms_results = []
        time_results = []
        for rep_exp in tqdm(range(10)):
            start = time()
            error, ahn = AHN(x, y, n=molNum, model=settings['learnAlg'], iterations=settings['iterations'])
            end = time()
            exec_time = float('%.3f'%(end - start))
            print('-----------------------------------------------')
            print('AHN-model trained with {0} molecules, {3} training iterations and {1} registries in {2} seconds'.format(molNum, reg, exec_time, settings['iterations']))

            valid = []
            for h in ahn['H']:
                if len(h) != 0:
                    valid.append(True)
                else:
                    valid.append(False)

            if all(valid) == True:
                print('Predicting...')
                print('----------------------------------------------------------------------------------------------')
                ysim, _, _, = predict(ahn, x, y)
                #ViewResults(ysim, y, mode='3D', x=x)
                rms = sqrt(mean_squared_error(y, ysim))
                rms_results.append(rms)
                time_results.append(exec_time)
            else:
                fail_count = fail_count + 1
                print('failed prediction')
            if fail_count > 8:
                succes = 0
                while succes <= 1:
                    start = time()
                    error, ahn = AHN(x, y, n=molNum, model=settings['learnAlg'], iterations=settings['iterations'])
                    end = time()
                    exec_time = float('%.3f' % (end - start))
                    print('-----------------------------------------------')
                    print(
                        'RETRAINING: AHN-model trained with {0} molecules, {3} training iterations and {1} registries in {2} seconds'.format(
                            molNum, reg, exec_time, settings['iterations']))
                    valid = []
                    for h in ahn['H']:
                        if len(h) != 0:
                            valid.append(True)
                        else:
                            valid.append(False)

                    if all(valid) == True:
                        print('Predicting...')
                        print(
                            '----------------------------------------------------------------------------------------------')
                        ysim, _, _, = predict(ahn, x, y)
                        # ViewResults(ysim, y)
                        rms = sqrt(mean_squared_error(y, ysim))
                        rms_results.append(rms)
                        time_results.append(exec_time)
                        succes = succes + 1
                    else:
                        print('failed prediction')

        #ViewResults(ysim, y, mode='3D', x=x)
            #ViewResults(Yestimates, y)

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
finalResults.to_csv('results_{0}.csv'.format(settings['learnAlg']), index=False)
print('finish')