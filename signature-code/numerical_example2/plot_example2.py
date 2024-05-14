import matplotlib.pyplot as plt
import numpy as np

methods = ['DeepSigNet', 'SigFormer', 'Simple SigFormer']
models = ['fBm', 'rBergomi', 'rHeston']
truncation_order = [1, 2, 3, 4, 5]
RMSE = np.array([
    # Truncation order 1
    [4.28e-2, 5.99e-2, 6.08e-2],  # DeepSigNet
    [4.18e-2, 5.99e-2, 6.08e-2],  # SigFormer
    [5.91e-2, 5.99e-2, 6.08e-2],  # Simple SigFormer
    # Truncation order 2
    [8.93e-3, 5.60e-2, 6.06e-2],  # DeepSigNet
    [9.16e-3, 5.17e-2, 6.03e-2],  # SigFormer
    [5.55e-2, 5.95e-2, 6.08e-2],  # Simple SigFormer
    # Truncation order 3
    [1.50e-2, 5.37e-2, 6.04e-2],  # DeepSigNet
    [1.14e-2, 4.14e-2, 5.85e-2],  # SigFormer
    [3.86e-2, 5.95e-2, 6.08e-2],  # Simple SigFormer
    # Truncation order 4
    [3.57e-2, 5.30e-2, 6.03e-2],  # DeepSigNet
    [2.37e-2, 4.06e-2, 5.13e-2],  # SigFormer
    [3.13e-2, 5.95e-2, 6.08e-2],  # Simple SigFormer
    # Truncation order 5
    [2.76e-2, 5.31e-2, 6.04e-2],  # DeepSigNet
    [5.28e-2, 4.06e-2, 4.94e-2],  # SigFormer
    [2.77e-2, 5.95e-2, 6.08e-2]   # Simple SigFormer
]).reshape(5, 3, 3)

colors = np.array([[0.5       , 0.5       , 0.5       , 1.        ],
                   [0.9       , 0.7       , 0.        , 1.        ],
                   [1.        , 0.18954248, 0.        , 1.        ],
                   [0.        , 0.06470588, 1.        , 1.        ],
                   [0.05882352, 0.51764705, 0.17647058, 1.        ],
                   [0.28627450, 0.18823529, 0.06666666, 1.        ]])

for i, model in enumerate(models):
    plt.figure(figsize=(10, 8))
    for j, method in enumerate(methods):
        plt.plot(truncation_order, RMSE[:, j, i], label=method, color=colors[j], linestyle='-', marker='o')
    plt.xticks(truncation_order, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Truncation order', fontsize=16)
    plt.ylabel('RMSE', fontsize=16)
    plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 16})
    plt.grid(True, alpha=0.2)
    plt.savefig(f'data/results/example2_{model}.png')