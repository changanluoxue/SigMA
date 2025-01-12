import matplotlib.pyplot as plt
import numpy as np

methods = ['DeepSigNet', 'SigMA', 'SigSA']
models = ['fBm', 'rBergomi', 'rHeston']
markers = ['v','o','^']
linestyles = ['--', '-.', '-']
truncation_order = [1, 3, 5]

# 两个RMSE矩阵依次对应输入长度为100和500的情况
# RMSE = np.array([
#     # Truncation order 1
#     [4.26e-2, 5.95e-2, 6.08e-2],  # DeepSigNet
#     [3.93e-2, 5.95e-2, 6.08e-2],  # SigMA
#     [5.92e-2, 5.95e-2, 6.08e-2],  # SigSA
#     # Truncation order 3
#     [1.20e-2, 3.99e-2, 6.00e-2],  # DeepSigNet
#     [1.08e-2, 3.60e-2, 4.71e-2],  # SigMA
#     [3.75e-2, 5.95e-2, 6.08e-2],  # SigSA
#     # Truncation order 5
#     [3.40e-2, 4.09e-2, 5.97e-2],  # DeepSigNet
#     [5.05e-2, 3.51e-2, 4.81e-2],  # SigMA
#     [2.44e-2, 5.95e-2, 6.08e-2]   # SigSA
# ]).reshape(3, 3, 3)

RMSE = np.array([
    # Truncation order 1
    [3.85e-2, 5.90e-2, 5.96e-2],  # DeepSigNet
    [4.09e-2, 5.90e-2, 5.96e-2],  # SigMA
    [6.14e-2, 5.90e-2, 5.96e-2],  # SigSA
    # Truncation order 3
    [1.56e-2, 1.76e-2, 4.38e-2],  # DeepSigNet
    [7.67e-3, 2.38e-2, 1.02e-2],  # SigMA
    [3.66e-2, 5.90e-2, 5.96e-2],  # SigSA
    # Truncation order 5
    [6.13e-2, 2.37e-2, 4.89e-2],  # DeepSigNet
    [1.03e-1, 2.17e-2, 9.79e-3],  # SigMA
    [2.54e-2, 5.90e-2, 5.96e-2]   # SigSA
]).reshape(3, 3, 3)

colors = np.array([[0.5       , 0.5       , 0.5       , 1.        ],
                   [0.9       , 0.7       , 0.        , 1.        ],
                   [1.        , 0.18954248, 0.        , 1.        ],
                   [0.        , 0.06470588, 1.        , 1.        ],
                   [0.05882352, 0.51764705, 0.17647058, 1.        ],
                   [0.28627450, 0.18823529, 0.06666666, 1.        ]])

for i, model in enumerate(models):
    plt.figure(figsize=(10, 8))
    for j, method in enumerate(methods):
        plt.plot(truncation_order, RMSE[:, j, i], label=method, color=colors[j], linestyle=linestyles[j], marker=markers[j])
    plt.xticks(truncation_order, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Truncation order', fontsize=18, fontweight='bold')
    plt.ylabel('RMSE', fontsize=18, fontweight='bold')
    plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 18, 'weight': 'bold'})
    plt.grid(True, alpha=0.2)
    plt.savefig(f'data/results/example2_Hbeta_{model}_grid=500.png')