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
# X_train = np.load('data/simulated_data/fBm/Xtrain_grid=1500_Huniform.npy')
# Y_train = np.load('data/simulated_data/fBm/Ytrain_grid=1500_Huniform.npy')
# X_eval = np.load('data/simulated_data/fBm/Xeval_grid=1500_Huniform.npy')
# Y_eval = np.load('data/simulated_data/fBm/Yeval_grid=1500_Huniform.npy')

X_train = np.load('data/simulated_data/fBm/Xtrain_grid=1500_Hbeta.npy')
Y_train = np.load('data/simulated_data/fBm/Ytrain_grid=1500_Hbeta.npy')
X_eval = np.load('data/simulated_data/fBm/Xeval_grid=1500_Hbeta.npy')
Y_eval = np.load('data/simulated_data/fBm/Yeval_grid=1500_Hbeta.npy')

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

print('******开始训练SigFormer_1******')
sigformer_1 = models.sigformer_1(augment_include_time=True, T=1)
model_trainer(sigformer_1, r'$\rm SigFormer\,without\,CNN$', history)
torch.save(sigformer_1, 'data/results/input_length_1500/fBm/trained_models/SigFormer_1.pth')
print('******SigFormer_1训练完成******')

print('******开始训练SigFormer_2******')
sigformer_2 = models.sigformer_2(augment_include_original=True, augment_include_time=True, T=1)
model_trainer(sigformer_2, r'$\rm SigFormer\,without\,MLP$', history)
torch.save(sigformer_2, 'data/results/input_length_1500/fBm/trained_models/SigFormer_2.pth')
print('******SigFormer_2训练完成******')

print('******开始训练SigFormer_3******')
sigformer_3 = models.sigformer_3(augment_include_time=True, T=1)
model_trainer(sigformer_3, r'$\rm SigFormer\,without\,CNN\,or\,MLP$', history)
torch.save(sigformer_3, 'data/results/input_length_1500/fBm/trained_models/SigFormer_3.pth')
print('******SigFormer_3训练完成******')

print('******开始训练SigFormer******')
sigformer = models.sigformer(augment_include_original=True, augment_include_time=True, T=1)
model_trainer(sigformer, 'SigFormer', history)
torch.save(sigformer, 'data/results/input_length_1500/fBm/trained_models/SigFormer.pth')
print('******SigFormer训练完成******')

'''输出训练结果'''
params = {}
for i, j in zip((r'$\rm SigFormer\,without\,CNN$', r'$\rm SigFormer\,without\,MLP$', r'$\rm SigFormer\,without\,CNN\,or\,MLP$', 'SigFormer'),
                (sigformer_1, sigformer_2, sigformer_3, sigformer)):
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
for k in (r'$\rm SigFormer\,without\,CNN$', r'$\rm SigFormer\,without\,MLP$', r'$\rm SigFormer\,without\,CNN\,or\,MLP$', 'SigFormer'):
    df_eval_logging[k] = history[k]['eval_mse']
fig, axes = plt.subplots(figsize=(12, 12))
df_eval_logging.rolling(5).mean().plot(grid=False, ax=axes, color=colors, lw=1.5, alpha=0.8)
plt.yscale('log')
axes.set_xlabel('Epoch')
axes.set_ylabel('Test MSE')
plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=2, prop={'size': 16})

# plt.savefig('data/results/input_length_1500/fBm/fBm_grid=1500_Huniform_round3.eps', bbox_inches='tight')
plt.savefig('data/results/input_length_1500/fBm/fBm_grid=1500_Hbeta_round3.eps', bbox_inches='tight')

end = time.time()
print('---------------总耗时 {:.2f}s---------------'.format(end-start))



