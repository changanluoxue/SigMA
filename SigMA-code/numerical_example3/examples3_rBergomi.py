import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
import models

start = time.time()
'''参数输入'''
time_augmented = False
max_epochs = 150
lr = 0.0001
optimizer_fn=optim.Adam
'''自定义损失函数'''
def loss_fn(y_pre,y):
    return np.sqrt(F.mse_loss(y_pre,y))
'''导入数据集'''
# X_train = np.load('data/simulated_data/rBergomi/Xtrain_grid=1500_Huniform_eta=1_v0=0.01.npy')
# Y_train = np.load('data/simulated_data/rBergomi/Ytrain_grid=1500_Huniform_eta=1_v0=0.01.npy')
# X_eval = np.load('data/simulated_data/rBergomi/Xeval_grid=1500_Huniform_eta=1_v0=0.01.npy')
# Y_eval = np.load('data/simulated_data/rBergomi/Yeval_grid=1500_Huniform_eta=1_v0=0.01.npy')

X_train = np.load('data/simulated_data/rBergomi/Xtrain_grid=1500_Hbeta_eta=1_v0=0.01.npy')
Y_train = np.load('data/simulated_data/rBergomi/Ytrain_grid=1500_Hbeta_eta=1_v0=0.01.npy')
X_eval = np.load('data/simulated_data/rBergomi/Xeval_grid=1500_Hbeta_eta=1_v0=0.01.npy')
Y_eval = np.load('data/simulated_data/rBergomi/Yeval_grid=1500_Hbeta_eta=1_v0=0.01.npy')

X_train = np.expand_dims(X_train, 1); X_eval = np.expand_dims(X_eval, 1)
train_dataloader, eval_dataloader, example_batch_x, example_batch_y = utils.generate_torch_batched_data(X_train, Y_train,
                                                                                                        X_eval, Y_eval,
                                                                                                        train_batch_size=64,
                                                                                                        test_batch_size=64)
'''定义训练器'''
model_trainer = utils.create_model_supervised_trainer(max_epochs=max_epochs, optimizer_fn=optimizer_fn,
                                                      loss_fn=loss_fn, train_dataloader=train_dataloader,
                                                      eval_dataloader=eval_dataloader, example_batch_x=example_batch_x, lr=lr)
'''训练模型'''
history={}

print('******开始训练SigMA_1******')
sigma_1 = models.sigma_1(augment_include_time=True, T=1)
model_trainer(sigma_1, r'$\rm SigMA\,without\,CNN$', history)
torch.save(sigma_1, 'data/results/input_length_1500/rBergomi/trained_models/SigMA_1.pth')
print('******SigMA_1训练完成******')

print('******开始训练SigMA_2******')
sigma_2 = models.sigma_2(augment_include_original=True, augment_include_time=True, T=1)
model_trainer(sigma_2, r'$\rm SigMA\,without\,MLP$', history)
torch.save(sigma_2, 'data/results/input_length_1500/rBergomi/trained_models/SigMA_2.pth')
print('******SigMA_2训练完成******')

print('******开始训练SigMA_3******')
sigma_3 = models.sigma_3(augment_include_time=True, T=1)
model_trainer(sigma_3, r'$\rm SigMA\,without\,CNN\,or\,MLP$', history)
torch.save(sigma_3, 'data/results/input_length_1500/rBergomi/trained_models/SigMA_3.pth')
print('******SigMA_3训练完成******')

print('******开始训练SigMA******')
sigma = models.sigma(augment_include_original=True, augment_include_time=True, T=1)
model_trainer(sigma, 'SigMA', history)
torch.save(sigma, 'data/results/input_length_1500/rBergomi/trained_models/SigMA.pth')
print('******SigMA训练完成******')

'''输出训练结果'''
params = {}
for i, j in zip((r'$\rm SigMA\,without\,CNN$', r'$\rm SigMA\,without\,MLP$', r'$\rm SigMA\,without\,CNN\,or\,MLP$', 'SigMA'),
                (sigma_1, sigma_2, sigma_3, sigma,)):
    params[i] = utils.count_parameters(j)
table_data = []
for key in history:
    table_data.append([key, history[key]['eval_mse'][-1], history[key]['eval_loss'][-1], params[key]])
# 打印表格
print(tabulate(table_data, headers=['Model', 'Eval MSE', 'Eval Loss', '# Parameters'],
               tablefmt='grid', floatfmt=['', '.2e', '.2e', '']))
'''绘制图形'''
colors = np.array([[0.5       , 0.5       , 0.5       , 1.        ],
                   [0.        , 0.64509804, 1.        , 1.        ],
                   [0.9       , 0.7       , 0.        , 1.        ],
                   [1.        , 0.18954248, 0.        , 1.        ],
                   [0.        , 0.06470588, 1.        , 1.        ],
                   [0.05882352, 0.51764705, 0.17647058, 1.        ],
                   [0.28627450, 0.18823529, 0.06666666, 1.        ]])
df_eval_logging = pd.DataFrame()
for k in (r'$\rm SigMA\,without\,CNN$', r'$\rm SigMA\,without\,MLP$', r'$\rm SigMA\,without\,CNN\,or\,MLP$', 'SigMA'):
    df_eval_logging[k] = history[k]['eval_mse']
fig, axes = plt.subplots(figsize=(12, 12))
df_eval_logging.rolling(5).mean().plot(grid=False, ax=axes, color=colors, lw=1.5, alpha=0.8)
plt.yscale('log')
axes.set_xlabel('Epoch')
axes.set_ylabel('Test MSE')
plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=2, prop={'size': 16})

# plt.savefig('data/results/input_length_1500/rBergomi/rBergomi_grid=1500_Huniform_eta=1_v0=0.01_round3.eps', bbox_inches='tight')
plt.savefig('data/results/input_length_1500/rBergomi/rBergomi_grid=1500_Hbeta_eta=1_v0=0.01_round3.eps', bbox_inches='tight')

end = time.time()
print('---------------总耗时 {:.2f}s---------------'.format(end-start))



