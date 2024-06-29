import pandas as pd
import numpy as np
import torch
import time
from tabulate import tabulate

start = time.time()
def lagged_mean(prices,q, lag ):
    # takes mean of (vector entry minus its shifted entry)^q
    return np.mean( pow(np.abs(prices- prices.shift(lag)), q) )
def findH(volSeries):
    # this function estimates H using the least squares approach
    K_q=[]
    qVec=[.5, 1., 1.5, 2., 3.]
    logVec=np.log(range(1,31))
    for q in qVec:
        meanVec=[]
        for n in range(1,31):
            vn=lagged_mean(volSeries,q,n)
            meanVec.append(vn)
        meanlogVec=np.log(meanVec)
        res=np.polyfit(logVec,meanlogVec,1)
        K_q.append(res[0])
    res=np.polyfit(qVec,K_q,1)
    return(res[0])

# models loaded
sigformer = torch.load('data/results/trained_models/sigformer.pth')
sigformer.eval()
deepsignet = torch.load('data/results/trained_models/deepsignet.pth')
deepsignet.eval()
cnn = torch.load('data/results/trained_models/cnn.pth')
cnn.eval()
transformer = torch.load('data/results/trained_models/transformer.pth')
transformer.eval()

models = {'sigformer': sigformer, 'deepsignet': deepsignet,
          'cnn': cnn, 'transformer': transformer}

# data loaded
market_data=pd.read_csv('data/market_data/oxfordmanrealizedvolatilityindices.csv')
market_data=market_data.drop(columns=['close_time','open_price'])
Symbol=np.unique(market_data.Symbol)
methodList=market_data.columns[2:]

# Calibration
for s in Symbol:
    data=market_data[market_data.Symbol==s].iloc[0:200]
    for meth in methodList:
        sample=data[meth]
        diffVec = {'sigformer': [], 'deepsignet': [], 'cnn': [], 'transformer': []}
        for t in [0,10,20,30,40,50,60,70,80,90,100]:
            volSeries=sample.iloc[t:(100+t)]
            logvolSeries=np.log(volSeries)
            'least-square prediction'
            ls_pred=findH(logvolSeries)
            'trained models prediction'
            volSeries = torch.tensor(volSeries.values, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            for model_name, model in models.items():
                model_pred=model(volSeries)
                model_pred=np.float64(model_pred.item())
                diffVec[model_name].append(np.square(ls_pred - model_pred))

table_data = []
for model_name in models.keys():
    rmse = np.sqrt(np.mean(diffVec[model_name]))
    std_dev = np.std(diffVec[model_name])
    table_data.append([model_name, rmse, std_dev])

print(tabulate(table_data, headers=['Model', 'RMSE', 'Std Dev'],
               tablefmt='grid', floatfmt=['', '.2e', '.2e']))

end = time.time()
print('---------------总耗时 {:.2f}s---------------'.format(end-start))