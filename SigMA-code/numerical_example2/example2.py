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

input_length = 100
round = 1
truncation = 5
H_distribution = 'Huniform' #'Huniform', 'Hbeta'
volatility_model = 'fBm' #'fBm', 'rBergomi', 'rHeston'

'''导入数据集'''
X_train = np.load(f'data/simulated_data/{volatility_model}/Xtrain_grid={input_length}_{H_distribution}.npy')
Y_train = np.load(f'data/simulated_data/{volatility_model}/Ytrain_grid={input_length}_{H_distribution}.npy')
X_eval = np.load(f'data/simulated_data/{volatility_model}/Xeval_grid={input_length}_{H_distribution}.npy')
Y_eval = np.load(f'data/simulated_data/{volatility_model}/Yeval_grid={input_length}_{H_distribution}.npy')

X_train = np.expand_dims(X_train, 1); X_eval = np.expand_dims(X_eval, 1)
train_dataloader, eval_dataloader, example_batch_x, example_batch_y \
    = utils.generate_torch_batched_data(X_train, Y_train, X_eval, Y_eval, train_batch_size=64, test_batch_size=64)
'''定义训练器'''
model_trainer = utils.create_model_supervised_trainer(max_epochs=max_epochs, optimizer_fn=optimizer_fn,
                                                      loss_fn=loss_fn, train_dataloader=train_dataloader,
                                                      eval_dataloader=eval_dataloader, example_batch_x=example_batch_x, lr=lr)
'''训练模型'''
history={}
print('******开始训练DeepSigNet******')
deepsignet = models.deepsignet(augment_include_original=True, augment_include_time=True, T=1)
model_trainer(deepsignet, 'DeepSigNet', history)
torch.save(deepsignet, f'data/results/Trun_{truncation}/input_length_{input_length}/{volatility_model}/trained_models/{H_distribution}/round{round}/DeepSigNet.pth')
print('******DeepSigNet训练完成******')

print('******开始训练SigMA******')
sigma = models.sigma(augment_include_original=True, augment_include_time=True, T=1)
model_trainer(sigma, 'SigMA', history)
torch.save(sigma, f'data/results/Trun_{truncation}/input_length_{input_length}/{volatility_model}/trained_models/{H_distribution}/round{round}/SigMA.pth')
print('******SigMA训练完成******')

print('******开始训练SigSA******')
sigsa = models.sigsa(augment_include_time=True, T=1)
model_trainer(sigsa, 'SigSA', history)
torch.save(sigsa, f'data/results/Trun_{truncation}/input_length_{input_length}/{volatility_model}/trained_models/{H_distribution}/round{round}/SigSA.pth')
print('******SigSA训练完成******')

'''输出训练结果'''
params = {}
for i, j in zip(('DeepSigNet', 'SigMA', 'SigSA'),
                (deepsignet, sigma, sigsa)):
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
for k in ('DeepSigNet', 'SigMA', 'SigSA'):
    df_eval_logging[k] = history[k]['eval_mse']
fig, axes = plt.subplots(figsize=(10, 8))
df_eval_logging.rolling(5).mean().plot(grid=False, ax=axes, color=colors, lw=1.5, alpha=0.8)
plt.yscale('log')
axes.set_xlabel('Epoch', fontsize=16, fontweight='bold')
axes.set_ylabel('Test MSE', fontsize=16, fontweight='bold')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=3, prop={'size': 16, 'weight': 'bold'})

plt.savefig(f'data/results/Trun_{truncation}/input_length_{input_length}/{volatility_model}/{volatility_model}_grid={input_length}_{H_distribution}_Trun{truncation}_round{round}.eps', bbox_inches='tight')

end = time.time()
print('---------------总耗时 {:.2f}s---------------'.format(end-start))



