import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
import antropy as ant
import statsmodels.stats.api as sms
import torch
import NNmodels

__name__ = "__calibration__"
if __name__ == '__calibration__':
    def RS_hurst(data, min_window=10):
        data = np.array(data); N = len(data)
        window_sizes = []; RS_values = []
        for window_size in range(min_window, N//2+1):
            window_sizes.append(window_size)
            sub_rs = []
            for start in range(0, N, window_size):
                end = start+window_size
                if end>N:
                    break
                segment = data[start:end]
                mean_segment = np.mean(segment)
                Z = np.cumsum(segment-mean_segment)
                R = np.max(Z)-np.min(Z)
                S = np.std(segment)
                if S > 0:
                    sub_rs.append(R/S)
            RS_values.append(np.mean(sub_rs))
        log_window_sizes = np.log(window_sizes)
        log_RS_values = np.log(RS_values)
        H, _ = np.polyfit(log_window_sizes, log_RS_values, 1)
        return H
    start = time.time()
    '''参数输入'''
    models = 'fBm'
    grid_points = 100
    device = torch.device('cpu')
    '''载入模型'''
    lstm = NNmodels.lstm_length(grid_points=grid_points)
    sigma = NNmodels.sigma_length(augment_include_original=True, augment_include_time=True, T=1, stride=int(grid_points/2))
    deepsignet = NNmodels.deepsignet_length(augment_include_original=True, augment_include_time=True, T=1)
    cnn = NNmodels.cnn_length()
    transformer = NNmodels.transformer_length(augment_include_original=True, augment_include_time=True, T=1, grid_points=grid_points)
    lstm.load_state_dict(torch.load(f'data/results/numerical_example7/trained_models/{models}_lstm_length{grid_points}.pth', map_location=torch.device(device))); lstm.eval()
    sigma.load_state_dict(torch.load(f'data/results/numerical_example7/trained_models/{models}_sigma_length{grid_points}.pth', map_location=torch.device(device))); sigma.eval()
    deepsignet.load_state_dict(torch.load(f'data/results/numerical_example7/trained_models/{models}_deepsignet_length{grid_points}.pth', map_location=torch.device(device))); deepsignet.eval()
    cnn.load_state_dict(torch.load(f'data/results/numerical_example7/trained_models/{models}_cnn_length{grid_points}.pth', map_location=torch.device(device))); cnn.eval()
    transformer.load_state_dict(torch.load(f'data/results/numerical_example7/trained_models/{models}_transformer_length{grid_points}.pth', map_location=torch.device(device))); transformer.eval()
    models = {'lstm': lstm, 'sigma': sigma, 'deepsignet': deepsignet,
              'cnn': cnn, 'transformer': transformer}
    '''导入47号电池数据'''
    battery_data=pd.read_csv('data/market_data/Battery_Data_Cleaned.csv')
    battery_data = battery_data[battery_data.battery_id==5]
    battery_capacity = battery_data.Capacity
    battery_capacity = battery_capacity[battery_capacity != 0] #过滤掉battery_capacity中值为0的异常值
    battery_capacity = (battery_capacity/np.max(battery_capacity))[1:] #归一化
    '''绘制电池寿命图像'''
    plt.figure(figsize=(10, 6))
    plt.plot(battery_capacity.values, linewidth=3)
    plt.xlabel("Cycles", fontsize=16, fontweight='bold')
    plt.ylabel("Capacity Proportion, Percentage", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(ticks=[0.8, 0.85, 0.9, 0.95, 1.0], labels=['80', '85', '90', '95', '100'], fontsize=15)
    plt.savefig('data/results/numerical_example7/plots/battery_degradation.eps')
    '''估计Hurst指数'''
    RS_H = []; Higuchi_H = []
    model_predictions = {model_name: [] for model_name in models.keys()}
    for t in range(0,451,10):#[0,10,..,450]一共46次循环，代表[50,60,...,500]轮电池效率
        battery_capacity_series = battery_capacity.iloc[t:(100+t)]
        rs_h = RS_hurst(battery_capacity_series)
        higuchi_h = 2-ant.higuchi_fd(battery_capacity_series, kmax=10) # Hurst = 2-FHD
        RS_H.append(rs_h); Higuchi_H.append(higuchi_h)
        for model_name, model in models.items():
            model_pred = model(torch.tensor(battery_capacity_series.values, dtype=torch.float).unsqueeze(0).unsqueeze(0))
            model_pred = np.float64(model_pred.item())
            model_predictions[model_name].append(model_pred)
    '''绘制Hurst指数估计图表'''
    table_data = []
    table_data.append(['R/S Method', np.mean(RS_H), sms.DescrStatsW(RS_H).tconfint_mean(alpha=1-0.95)])
    table_data.append(['Higuchi Method', np.mean(Higuchi_H), sms.DescrStatsW(Higuchi_H).tconfint_mean(alpha=1-0.95)])
    for model_name in models.keys():
        table_data.append([model_name, np.mean(model_predictions[model_name]),
                           sms.DescrStatsW(model_predictions[model_name]).tconfint_mean(alpha=1-0.95)])
    print(tabulate(table_data, headers=['Model', 'H-estimate', '95% confidence intervals'],
                   tablefmt='grid', floatfmt=['', '.3e', '.3e']))
    '''绘制Hurst指数估计图像'''
    markers = ['v','^','*','+','|']
    linestyles = ['dashdot', 'dotted', (0,(1,10)), (0,(5,10)), (0,(3,1,1,1))]
    plt.figure(figsize=(20, 10))
    plt.plot(RS_H, label='R/S Method', color='black', linestyle='solid', marker='.', markersize=10, linewidth=3)
    plt.plot(Higuchi_H, label='Higuchi Method', color='grey', linestyle='dashed', marker='o', markersize=10, linewidth=3)
    for i, (model_name, predictions) in enumerate(model_predictions.items()):
        plt.plot(predictions, label=f'{model_name}', marker=markers[i], linestyle=linestyles[i], markersize=10, linewidth=3)
    plt.xlabel("Cycles", fontsize=20, fontweight='bold')
    plt.ylabel('Hurst-estimate', fontsize=20, fontweight='bold')
    plt.xticks(ticks=range(0,46,5), labels=range(50,501,50), fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=4, prop={'size': 20, 'weight': 'bold'})
    plt.savefig('data/results/numerical_example7/plots/battery_hurst_estimate.eps')






