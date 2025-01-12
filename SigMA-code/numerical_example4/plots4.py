import matplotlib.pyplot as plt
import numpy as np

methods = ['Transformer', 'SigMA', 'DeepSigNet', 'CNN']
markers = ['v','o','.','^']
linestyles = ['--', '-.', '-', ':']
models = ['fBm', 'rBergomi', 'rHeston']
input_length = [100, 500, 1000, 1500]
RMSE_Huniform = [np.array([[7.37e-3, 7.68e-2, 8.63e-2],
                           [6.79e-3, 3.65e-2, 6.09e-2]]),

                 np.array([[1.02e-2, 4.92e-2, 7.56e-2],
                           [8.80e-3, 2.81e-2, 2.75e-2],
                           [1.35e-2, 2.93e-2, 1.57e-2],
                           [1.10e-2, 2.80e-2, 9.40e-3]]),

                 np.array([[1.14e-2, 4.73e-2, 1.09e-1],
                           [1.89e-2, 2.63e-2, 7.30e-2],
                           [1.97e-2, 2.38e-2, 3.42e-2],
                           [2.74e-2, 2.76e-2, 2.82e-2]]),

                 np.array([[7.12e-2, 4.22e-2, 8.48e-2],
                           [8.11e-2, 3.84e-2, 7.18e-2],
                           [8.44e-2, 4.11e-2, 5.52e-2],
                           [8.17e-2, 3.57e-2, 5.09e-2]])]
RMSE_Hbeta  =   [np.array([[1.22e-2, 3.41e-2, 5.71e-2],
                           [6.81e-3, 2.21e-2, 5.64e-2]]),

                 np.array([[9.52e-3, 3.64e-2, 5.53e-2],
                           [9.15e-3, 2.67e-2, 1.04e-2],
                           [7.46e-3, 1.70e-2, 8.25e-3],
                           [1.07e-2, 1.90e-2, 7.12e-3]]),

                 np.array([[1.19e-2, 3.55e-2, 5.98e-2],
                           [1.26e-2, 2.13e-2, 4.95e-2],
                           [1.54e-2, 1.47e-2, 1.46e-2],
                           [1.68e-2, 1.97e-2, 1.28e-2]]),

                 np.array([[1.26e-1, 4.95e-2, 7.94e-2],
                           [7.81e-2, 5.01e-2, 6.99e-2],
                           [8.04e-2, 4.48e-2, 7.43e-2],
                           [8.78e-2, 4.12e-2, 5.87e-2]])]

colors = np.array([[0.5       , 0.5       , 0.5       , 1.        ],
                   [0.        , 0.64509804, 1.        , 1.        ],
                   [0.9       , 0.7       , 0.        , 1.        ],
                   [1.        , 0.18954248, 0.        , 1.        ],
                   [0.        , 0.06470588, 1.        , 1.        ],
                   [0.05882352, 0.51764705, 0.17647058, 1.        ],
                   [0.28627450, 0.18823529, 0.06666666, 1.        ]])

for i, model in enumerate(models):
    plt.figure(figsize=(10, 8))
    for j, method in enumerate(methods):
        if method == 'Transformer':
            plt.plot([100,500], RMSE_Huniform[j][:, i], label=method, color=colors[j], linestyle=linestyles[j], marker=markers[j])
        else:
            plt.plot(input_length, RMSE_Huniform[j][:, i], label=method, color=colors[j], linestyle=linestyles[j], marker=markers[j])
    plt.xticks(input_length, fontsize=18)
    plt.yticks(fontsize = 18)
    plt.xlabel('Input Length', fontsize=18, fontweight='bold')
    plt.ylabel('RMSE', fontsize=18, fontweight='bold')
    plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 18, 'weight': 'bold'})
    plt.grid(True, alpha=0.2)
    plt.savefig(f'data/results/example4_Huniform_{model}.png')



for i, model in enumerate(models):
    plt.figure(figsize=(10, 8))
    for j, method in enumerate(methods):
        if method == 'Transformer':
            plt.plot([100, 500], RMSE_Hbeta[j][:, i], label=method, color=colors[j], linestyle=linestyles[j], marker=markers[j])
        else:
            plt.plot(input_length, RMSE_Hbeta[j][:, i], label=method, color=colors[j], linestyle=linestyles[j], marker=markers[j])
    plt.xticks(input_length, fontsize=18)
    plt.yticks(fontsize = 18)
    plt.xlabel('Input Length', fontsize=18, fontweight='bold')
    plt.ylabel('RMSE', fontsize=18, fontweight='bold')
    plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 18, 'weight': 'bold'})
    plt.grid(True, alpha=0.2)
    plt.savefig(f'data/results/example4_Hbeta_{model}.png')


