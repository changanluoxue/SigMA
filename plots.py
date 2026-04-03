import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from matplotlib.lines import lineStyles
from tabulate import tabulate
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
import NNmodels

__name__ = "__example2__" # "__example2__", "__example4__", "__example5__"

if __name__ == '__example2__':
    '''Input error data'''
    fBm_100 = np.array([[9.52e-2, 1.72e-2, 1.34e-2],
                        [1.18e-1, 1.15e-2, 1.06e-2],
                        [2.85e-1, 1.32e-1, 1.09e-1]])
    fOU_100 = np.array([[1.34e-1, 3.34e-2, 3.41e-2],
                        [1.34e-1, 2.96e-2, 2.44e-2],
                        [1.36e-1, 1.35e-1, 1.35e-1]])
    rHeston_100 = np.array([[5.89e-2, 5.81e-2, 5.78e-2],
                            [5.89e-2, 5.44e-2, 5.48e-2],
                            [1.15e-1, 5.89e-2, 5.89e-2]])
    fBm_500 = np.array([[9.32e-2, 8.41e-3, 4.53e-2],
                        [1.26e-1, 8.89e-3, 8.24e-2],
                        [2.74e-1, 1.33e-1, 1.08e-1]])
    fOU_500 = np.array([[1.37e-1, 2.18e-2, 2.93e-2],
                        [1.37e-1, 1.59e-2, 1.26e-2],
                        [1.39e-1, 1.39e-1, 1.39e-1]])
    rHeston_500 = np.array([[5.92e-2, 5.03e-2, 3.44e-2],
                            [5.92e-2, 1.02e-2, 1.01e-2],
                            [7.59e-2, 5.92e-2, 5.92e-2]])
    nn_models = ['DeepSigNet', 'SigMA', 'SigSA']
    truncation_order = ['1', '3', '5']
    v_model_dict = {'fBm_100': fBm_100, 'fOU_100': fOU_100, 'rHeston_100': rHeston_100,
                    'fBm_500': fBm_500, 'fOU_500': fOU_500, 'rHeston_500': rHeston_500}
    '''Select stochastic process'''
    v_model = 'fBm_100'  # 'fBm_100', 'fOU_100', 'rHeston_100', 'fBm_500', 'fOU_500', 'rHeston_500'
    '''Plot error line chart'''
    plt.figure(figsize=(10, 10))
    colors = ['blue', 'orange', 'red']
    markers = ['o', 's', '^']
    lines = ['--', '-', ':']
    for i in range(len(nn_models)):
        plt.plot(truncation_order, v_model_dict[v_model][i], label=nn_models[i], color=colors[i],
                 marker=markers[i], linewidth=3.5, linestyle=lines[i], markersize=10)
    plt.xticks(truncation_order, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Truncation order', fontsize=20, fontweight='bold')
    plt.ylabel('Test RMSE', fontsize=20, fontweight='bold')
    legend = plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 20})
    'Highlight SigMA'
    for text in legend.get_texts():
        if text.get_text() == 'SigMA':
            text.set_fontweight('bold')
        else:
            text.set_fontweight('normal')
    plt.savefig(f'data/results/numerical_example2/plots/{v_model}_example2.eps')
    plt.show()

if __name__ == "__example4__":
    '''Import error data'''
    fBm = np.array([[1.90e-2, 4.68e-2, 7.75e-2, 8.94e-2],
                    [1.05e-1, 1.07e-1, 1.12e-1, 1.10e-1],
                    [7.70e-2, 5.35e-2, 3.83e-2, 3.89e-2],
                    [1.60e-2, 9.02e-3, 6.84e-3, 3.13e-2],
                    [1.66e-2, 8.51e-3, 8.16e-3, 1.09e-2]])
    fOU = np.array([[9.22e-2, 8.93e-2, 9.53e-2, 9.24e-2],
                    [4.30e-2, 4.05e-2, 4.01e-2, 3.85e-2],
                    [9.22e-2, 4.69e-2, 3.19e-2, 3.46e-2],
                    [2.82e-2, 1.61e-2, 2.03e-2, 2.28e-2],
                    [3.51e-2, 3.01e-2, 2.96e-2, 2.51e-2]])
    rHeston = np.array([[5.72e-2, 5.24e-2, 4.37e-2, 4.25e-2],
                        [8.05e-2, 7.65e-2, 6.60e-2, 5.79e-2],
                        [7.25e-2, 6.29e-2, 6.16e-2, 6.11e-2],
                        [4.78e-2, 1.01e-2, 7.23e-3, 7.06e-3],
                        [5.77e-2, 5.13e-2, 1.97e-2, 9.05e-3]])
    nn_models = ['Transformer', 'CNN', 'LSTM', 'SigMA', 'DeepSigNet']
    input_lengths = ['100', '500', '1000', '1500']
    v_model_dict = {'fBm': fBm, 'fOU': fOU, 'rHeston': rHeston}
    '''Select stochastic process'''
    v_model = 'rHeston' # 'fBm', 'fOU', 'rHeston'
    '''Plot error line chart'''
    plt.figure(figsize=(10, 10))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    lines = ['--', ':', (0, (3, 2, 1, 2, 1, 2)), '-', '-.']
    for i in range(len(nn_models)):
        plt.plot(input_lengths, v_model_dict[v_model][i], label=nn_models[i], color=colors[i],
                 marker=markers[i], linewidth=3.5, linestyle=lines[i], markersize=10)
    plt.xticks(input_lengths, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Input Length', fontsize=20, fontweight='bold')
    plt.ylabel('Test RMSE', fontsize=20, fontweight='bold')
    legend = plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 20})
    'Highlight SigMA'
    for text in legend.get_texts():
        if text.get_text() == 'SigMA':
            text.set_fontweight('bold')
        else:
            text.set_fontweight('normal')
    plt.savefig(f'data/results/numerical_example4/plots/{v_model}_example4.eps')
    plt.show()

if __name__ == '__example5__':
    '''Select stochastic process'''
    v_model = 'rHeston'  # 'fOU', 'rHeston'
    '''Input error data'''
    errors_average_rse = {'Transformer': [], 'SigMA': [], 'Deepsignet': [], 'CNN': [], 'LSTM': []}
    errors_average_rmse = {'Transformer': [], 'SigMA': [], 'Deepsignet': [], 'CNN': [], 'LSTM': []}
    for round in range(10):
        data = np.load(f'data/results/numerical_example5/errors/{v_model}/errors_se_round{round}.npz')
        for nn_model_name in errors_average_rse:
            error_average_rse = np.mean(np.sqrt(data[nn_model_name].squeeze(0)), axis=1)
            error_average_rmse = np.mean(np.sqrt(np.mean(data[nn_model_name].squeeze(0), axis=0)))
            errors_average_rse[nn_model_name].append(error_average_rse)
            errors_average_rmse[nn_model_name].append(error_average_rmse)

    'Print the 10-round mean of Average RMSE'
    print(f'{v_model}_Average_RMSE_average:--------------')
    for nn_model_name in errors_average_rmse:
        print(f"{nn_model_name:15}: {np.mean(errors_average_rmse[nn_model_name]):7.3f}")

    'Print and plot the distribution of 10-round Average RSE'
    print(f'{v_model}_Average_RSE:--------------')
    colors = {'Transformer': 'blue', 'SigMA': 'orange', 'Deepsignet': 'purple', 'CNN': 'red', 'LSTM': 'green'}
    lines = {'Transformer': '--', 'SigMA': '-', 'Deepsignet': '-.', 'CNN': ':', 'LSTM': (0, (3, 2, 1, 2, 1, 2))}
    plt.figure(figsize=(20, 8))
    for nn_model_name in errors_average_rse:
        rse = np.concatenate(errors_average_rse[nn_model_name])
        q1 = np.percentile(rse, 25)
        q3 = np.percentile(rse, 75)
        maxi = np.max(rse)
        print(f'{nn_model_name:15} max:{maxi:7.3f} q3:{q3:7.3f} q1:{q1:7.3f}')
        sns.kdeplot(rse, label=nn_model_name, bw_adjust=5, linewidth=3.5, color=colors[nn_model_name], linestyle=lines[nn_model_name])
    plt.xlim(0, 2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Average RSE', fontsize=20, fontweight='bold')
    plt.ylabel('Probability density', fontsize=20, fontweight='bold')
    legend = plt.legend(loc='best', prop={'size': 20})
    'Highlight SigMA'
    for text in legend.get_texts():
        if text.get_text() == 'SigMA':
            text.set_fontweight('bold')
        else:
            text.set_fontweight('normal')
    plt.tight_layout()
    plt.savefig(f'data/results/numerical_example5/plots/{v_model}_example5.eps')
    plt.show()
