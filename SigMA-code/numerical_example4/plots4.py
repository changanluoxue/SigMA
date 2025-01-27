import matplotlib.pyplot as plt
import numpy as np

methods = ['Transformer', 'SigMA', 'DeepSigNet', 'CNN', 'LSTM']
markers = ['v','o','.','^','*']
linestyles = ['dashed', 'dashdot', 'solid', 'dotted', (0,(1,10))]
models = ['fBm', 'rBergomi', 'rHeston']
input_length = [100, 500, 1000, 1500]
RMSE_Huniform = [
    np.array([[2.16e-2, 1.66e-1, 3.04e-2],
              [5.15e-3, 2.23e-1, 3.64e-2]]),

    np.array([[1.36e-2, 7.93e-2, 3.02e-2],
              [8.09e-3, 9.73e-2, 3.05e-2],
              [8.55e-3, 1.02e-1, 2.99e-2],
              [3.07e-2, 1.05e-1, 2.58e-2]]),

    np.array([[1.70e-2, 9.99e-2, 3.14e-2],
              [1.02e-2, 1.06e-1, 3.46e-2],
              [7.97e-3, 1.06e-1, 3.09e-2],
              [1.20e-2, 1.05e-1, 3.28e-2]]),

    np.array([[9.63e-2, 7.53e-2, 7.42e-2],
              [1.07e-1, 7.29e-2, 7.63e-2],
              [1.20e-1, 8.55e-2, 5.52e-2],
              [1.17e-1, 8.05e-2, 7.03e-2]]),

    np.array([[7.22e-2, 6.48e-2, 6.10e-2],
              [3.84e-2, 4.03e-2, 4.68e-2],
              [4.10e-2, 4.69e-2, 8.22e-2],
              [3.38e-2, 4.00e-2, 5.74e-2]])
]
RMSE_Hbeta = [
    np.array([[9.95e-3, 3.89e-2, 5.69e-2],
              [8.85e-3, 2.46e-2, 5.31e-2]]),

    np.array([[1.07e-2, 3.52e-2, 5.61e-2],
              [1.16e-2, 2.38e-2, 1.06e-2],
              [1.02e-2, 2.19e-2, 8.47e-3],
              [1.32e-2, 2.13e-2, 6.51e-3]]),

    np.array([[1.19e-2, 3.76e-2, 5.99e-2],
              [1.58e-2, 2.11e-2, 5.31e-2],
              [1.83e-2, 1.77e-2, 1.24e-2],
              [1.52e-2, 1.52e-2, 9.54e-3]]),

    np.array([[1.28e-1, 5.06e-2, 8.08e-2],
              [7.49e-2, 4.75e-2, 7.03e-2],
              [8.25e-2, 4.68e-2, 4.76e-2],
              [9.26e-2, 4.66e-2, 5.41e-2]]),

    np.array([[4.70e-2, 5.36e-2, 7.29e-2],
              [3.26e-2, 3.14e-2, 6.49e-2],
              [2.61e-2, 3.06e-2, 6.34e-2],
              [2.85e-2, 2.15e-2, 6.34e-2]])
]

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


