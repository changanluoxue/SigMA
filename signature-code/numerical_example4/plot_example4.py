import matplotlib.pyplot as plt
import numpy as np

methods = ['Transformer', 'SigFormer', 'DeepSigNet', 'CNN']
models = ['fBm', 'rBergomi', 'rHeston']
input_length = [100, 200, 300, 400, 500]
RMSE_Huniform = np.array([
                 [8.05e-3, 5.79e-2, 8.82e-2],
                 [6.07e-3, 4.86e-2, 8.00e-2],
                 [6.78e-3, 5.04e-2, 9.55e-2],
                 [7.53e-3, 3.76e-2, 8.67e-2],
                 [1.11e-2, 4.61e-2, 8.17e-2],
                 [1.36e-2, 5.38e-2, 9.06e-2],
                 [1.54e-2, 4.95e-2, 6.99e-2],
                 [1.56e-2, 4.28e-2, 5.49e-2],
                 [1.90e-2, 3.79e-2, 5.47e-2],
                 [1.43e-2, 3.88e-2, 4.36e-2],
                 [1.69e-2, 6.22e-2, 1.31e-1],
                 [1.69e-2, 5.38e-2, 1.25e-1],
                 [1.62e-2, 4.29e-2, 1.27e-1],
                 [2.09e-2, 4.50e-2, 1.22e-1],
                 [1.81e-2, 4.31e-2, 1.20e-1],
                 [7.24e-2, 5.11e-2, 9.56e-2],
                 [8.42e-2, 4.79e-2, 8.73e-2],
                 [8.66e-2, 5.51e-2, 9.82e-2],
                 [7.76e-2, 4.84e-2, 9.21e-2],
                 [8.51e-2, 4.31e-2, 8.86e-2]]).reshape(4, 5, 3)
RMSE_Hbeta = np.array([
                 [8.97e-3, 5.02e-2, 6.04e-2],
                 [9.43e-3, 4.02e-2, 5.73e-2],
                 [1.15e-2, 3.25e-2, 5.78e-2],
                 [7.96e-3, 3.21e-2, 5.82e-2],
                 [8.05e-3, 2.60e-2, 5.86e-2],
                 [1.28e-2, 3.92e-2, 5.81e-2],
                 [1.78e-2, 3.36e-2, 5.63e-2],
                 [1.93e-2, 2.87e-2, 3.35e-2],
                 [1.48e-2, 3.13e-2, 2.91e-2],
                 [1.75e-2, 2.79e-2, 3.13e-2],
                 [1.64e-2, 5.18e-2, 6.03e-2],
                 [1.49e-2, 4.42e-2, 5.82e-2],
                 [1.66e-2, 4.23e-2, 5.76e-2],
                 [1.73e-2, 3.70e-2, 5.89e-2],
                 [2.14e-2, 3.56e-2, 5.80e-2],
                 [1.30e-1, 5.69e-2, 8.45e-2],
                 [1.10e-1, 6.15e-2, 8.47e-2],
                 [8.80e-2, 5.59e-2, 8.44e-2],
                 [8.76e-2, 5.77e-2, 7.93e-2],
                 [8.37e-2, 5.84e-2, 8.09e-2]]).reshape(4, 5, 3)

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
        plt.plot(input_length, RMSE_Huniform[j, :, i], label=method, color=colors[j], linestyle='-', marker='o')
    plt.xticks(input_length, fontsize=16)
    plt.yticks(fontsize = 16)
    plt.xlabel('Input Length', fontsize=16)
    plt.ylabel('RMSE', fontsize=16)
    plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 16})
    plt.grid(True, alpha=0.2)
    plt.savefig(f'data/results/example4_Huniform_{model}.png')



for i, model in enumerate(models):
    plt.figure(figsize=(10, 8))
    for j, method in enumerate(methods):
        plt.plot(input_length, RMSE_Hbeta[j, :, i], label=method, color=colors[j], linestyle='-', marker='o')
    plt.xticks(input_length, fontsize=16)
    plt.yticks(fontsize = 16)
    plt.xlabel('Input Length', fontsize=16)
    plt.ylabel('RMSE', fontsize=16)
    plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 16})
    plt.grid(True, alpha=0.2)
    plt.savefig(f'data/results/example4_Hbeta_{model}.png')


