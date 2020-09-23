from models.AHN import train as AHN
import numpy as np
from tqdm import tqdm
from time import time
from pandas import DataFrame, read_csv
from statistics import mean
from statistics import stdev

run_data = []

df_feat = read_csv('har.csv')
df_tags = read_csv('har_tags.csv')

x = df_feat.values[0:100]
y = df_tags.values[0:100]

for molNum in tqdm([3]):
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
    for rep_exp in tqdm(range(5)):
        start = time()
        error = AHN(x, y, n=molNum, model=settings['learnAlg'], iterations=settings['iterations'])
        end = time()
        exec_time = float('%.3f'%(end - start))
        print('-----------------------------------------------')
        print('AHN-model trained with {0} molecules, {3} training iterations in {2} seconds'.format(molNum, 1 , exec_time, settings['iterations']))

        rms_results.append(error)
        time_results.append(exec_time)

    rms_mean = mean(rms_results)
    rms_stDev = stdev(rms_results)
    time_mean = mean(time_results)
    time_stDev = stdev(time_results)

    run_data.append({
        'Molecules': molNum,
        'RMSE_Mean': rms_mean,
        'RMSE_stDev': rms_stDev,
        'ET_Mean': time_mean,
        'ET_stDev': time_stDev
    })
    print('first')

finalResults = DataFrame.from_dict(run_data)
finalResults.to_csv('results_{0}_HAR_test.csv'.format(settings['learnAlg']), index=False)
print('finished')