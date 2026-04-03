import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import pickle
from tabulate import tabulate
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
import NNmodels

'''Custom loss function'''
def rmse_loss_fn(y_pre, y):
    return torch.sqrt(F.mse_loss(y_pre, y))

__name__ = "__results__" # "__training__", "__results__"

if __name__ == "__training__":
    start = time.time()
    '''Parameter input'''
    models = 'fBm'  # 'fBm', 'fOU', 'rHeston'
    grid_points = 100  # 100, 500, 1000, 1500
    max_epochs = 150
    lr = 0.0001; optimizer_fn = optim.Adam
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    '''Import and process dataset'''
    X_train = np.load(f'data/simulated_data/{models}/Xtrain_single_grid={grid_points}.npy')
    Y_train = np.load(f'data/simulated_data/{models}/Ytrain_single_grid={grid_points}.npy')
    X_eval = np.load(f'data/simulated_data/{models}/Xeval_single_grid={grid_points}.npy')
    Y_eval = np.load(f'data/simulated_data/{models}/Yeval_single_grid={grid_points}.npy')
    X_train = np.expand_dims(X_train, 1); X_eval = np.expand_dims(X_eval, 1)
    train_dataloader, eval_dataloader, example_batch_x, example_batch_y = utils.generate_torch_batched_data(X_train, Y_train, X_eval, Y_eval,
                                                                                                            train_batch_size=64, test_batch_size=64, device=device)

    '''Define trainer'''
    model_trainer = utils.create_model_supervised_trainer(max_epochs=max_epochs, optimizer_fn=optimizer_fn,
                                                          loss_fn=rmse_loss_fn, train_dataloader=train_dataloader,
                                                          eval_dataloader=eval_dataloader, example_batch_x=example_batch_x, lr=lr, device=device)

    '''Train models'''
    if grid_points == 100:
        name = [r'$\rm SigMA_{stride=1}$', r'$\rm SigMA_{stride=10}$', r'$\rm SigMA_{stride=50}$']
        for round in range(3):
            history={}
            print('******Starting SigMA_stride1 training******')
            sigma_stride1 = NNmodels.sigma_stride(augment_include_original=True, augment_include_time=True, T=1, grid_points=grid_points, stride=1).to(device)
            model_trainer(sigma_stride1, name[0], history)
            torch.save(sigma_stride1, f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stride1_round{round}.pth')
            print('******SigMA_stride1 training complete******')

            print('******Starting SigMA_stride10 training******')
            sigma_stride10 = NNmodels.sigma_stride(augment_include_original=True, augment_include_time=True, T=1, grid_points=grid_points, stride=10).to(device)
            model_trainer(sigma_stride10, name[1], history)
            torch.save(sigma_stride10, f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stride10_round{round}.pth')
            print('******SigMA_stride10 training complete******')

            print('******Starting SigMA_stride50 training******')
            sigma_stride50 = NNmodels.sigma_stride(augment_include_original=True, augment_include_time=True, T=1, grid_points=grid_points, stride=50).to(device)
            model_trainer(sigma_stride50, name[2], history)
            torch.save(sigma_stride50, f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stride50_round{round}.pth')
            print('******SigMA_stride50 training complete******')

            with open(f'data/results/numerical_example1/history/{models}/history_length{grid_points}_round{round}.pkl', 'wb') as f:
                pickle.dump(history, f)

    if grid_points != 100:
        name = [r'$\rm SigMA_{stride=50}$', fr'$\rm SigMA_{{stride={int(grid_points/2)}}}$']
        for round in range(3):
            history = {}
            print('******Starting SigMA_stride50 training******')
            sigma_stride50 = NNmodels.sigma_stride(augment_include_original=True, augment_include_time=True, T=1, grid_points=grid_points, stride=50).to(device)
            model_trainer(sigma_stride50, name[0], history)
            torch.save(sigma_stride50, f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stride50_round{round}.pth')
            print('******SigMA_stride50 training complete******')

            print('******Starting SigMA_stridehalf training******')
            sigma_stridehalf = NNmodels.sigma_stride(augment_include_original=True, augment_include_time=True, T=1, grid_points=grid_points, stride=int(grid_points/2)).to(device)
            model_trainer(sigma_stridehalf, name[1], history)
            torch.save(sigma_stridehalf, f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stridehalf_round{round}.pth')
            print('******SigMA_stridehalf training complete******')

            with open(f'data/results/numerical_example1/history/{models}/history_length{grid_points}_round{round}.pkl', 'wb') as f:
                pickle.dump(history, f)

    end = time.time()
    print('---------------Total time elapsed {:.2f}s---------------'.format(end-start))

if __name__ == '__results__':
    '''Parameter input'''
    models = 'rHeston'  # 'fBm', 'fOU' , 'rHeston'
    grid_points = 1500  # 100, 500, 1000, 1500
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    '''Import datasetOutput training results'''
    history = []
    for round in range(3):
        with open(f'data/results/numerical_example1/history/{models}/history_length{grid_points}_round{round}.pkl', 'rb') as f:
            history.append(pickle.load(f))
    '''Output training results'''
    'Print summary table'
    print(f"{models}_length{grid_points}:")
    params = {}
    if grid_points == 100:
        sigma_stride1 = torch.load(f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stride1_round0.pth', weights_only=False, map_location=device)
        sigma_stride10 = torch.load(f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stride10_round0.pth', weights_only=False, map_location=device)
        sigma_stride50 = torch.load(f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stride50_round0.pth', weights_only=False, map_location=device)
        for i, j in zip((r'$\rm SigMA_{stride=1}$', r'$\rm SigMA_{stride=10}$', r'$\rm SigMA_{stride=50}$'),
                        (sigma_stride1, sigma_stride10, sigma_stride50)):
            params[i] = utils.count_parameters(j)
        'Per-round tables'
        for round in range(3):
            table_data = []
            for key in history[round]:
                table_data.append([key, history[round][key]['eval_mse'][-1], history[round][key]['eval_loss'][-1], params[key]])
            print(f"Round{round}:")
            print(tabulate(table_data, headers=['Model', 'Eval MSE', 'Eval RMSE', '# Parameters'],
                           tablefmt='grid', floatfmt=['', '.3e', '.3e', '']))
        'Averaged'
        Amse, Armse, C = defaultdict(float), defaultdict(float), defaultdict(int)
        for round in range(3):
            for k, v in history[round].items():
                Amse[k] += v['eval_mse'][-1]; Armse[k] += v['eval_loss'][-1]; C[k] += 1
        Arows = [[k, Amse[k]/C[k], Armse[k]/C[k], params[k]] for k in C]
        print("Average:")
        print(tabulate(Arows, headers=['Model', 'Eval MSE', 'Eval RMSE', '# Parameters'],
                       tablefmt='grid', floatfmt=['','.2e','.2e','']))
    if grid_points != 100:
        sigma_stride50 = torch.load(f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stride50_round0.pth', weights_only=False, map_location=device)
        sigma_stridehalf = torch.load(f'data/results/numerical_example1/trained_models/{models}/sigma_length{grid_points}_stridehalf_round0.pth', weights_only=False, map_location=device)
        for i, j in zip((r'$\rm SigMA_{stride=50}$', fr'$\rm SigMA_{{stride={int(grid_points/2)}}}$'),
                        (sigma_stride50, sigma_stridehalf)):
            params[i] = utils.count_parameters(j)
        'Per-round tables'
        for round in range(3):
            table_data = []
            for key in history[round]:
                table_data.append([key, history[round][key]['eval_mse'][-1], history[round][key]['eval_loss'][-1], params[key]])
            print(f"Round{round}:")
            print(tabulate(table_data, headers=['Model', 'Eval MSE', 'Eval RMSE', '# Parameters'],
                           tablefmt='grid', floatfmt=['', '.2e', '.2e', '']))
        'Averaged'
        Amse, Armse, C = defaultdict(float), defaultdict(float), defaultdict(int)
        for round in range(3):
            for k, v in history[round].items():
                Amse[k] += v['eval_mse'][-1]; Armse[k] += v['eval_loss'][-1]; C[k] += 1
        Arows = [[k, Amse[k]/C[k], Armse[k]/C[k], params[k]] for k in C]
        print("Average:")
        print(tabulate(Arows, headers=['Model', 'Eval MSE', 'Eval RMSE', '# Parameters'],
                       tablefmt='grid', floatfmt=['','.2e','.2e','']))
    'Plot results'
    def plot_loss_descent(grid_points=None, name=None, history=None):
        colors = np.array([[0.5       , 0.5       , 0.5       , 1.        ],
                           [0.        , 0.64509804, 1.        , 1.        ],
                           [0.9       , 0.7       , 0.        , 1.        ],
                           [1.        , 0.18954248, 0.        , 1.        ],
                           [0.        , 0.06470588, 1.        , 1.        ],
                           [0.05882352, 0.51764705, 0.17647058, 1.        ],
                           [0.28627450, 0.18823529, 0.06666666, 1.        ]])
        df_eval_logging = pd.DataFrame()
        for k in name:
            df_eval_logging[k] = history[k]['train_loss']
        fig, axes = plt.subplots(figsize=(10, 10))
        df_eval_logging.rolling(5).mean().plot(grid=False, ax=axes, color=colors, lw=3.5)
        axes.set_xlabel('Epoch', fontsize=20, fontweight='bold')
        axes.set_ylabel('RMSE', fontsize=20, fontweight='bold')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(mode='expand', bbox_to_anchor=(0, 1, 1, 0), ncol=2, prop={'size': 20, 'weight': 'bold'})
        plt.savefig(f'data/results/numerical_example1/plots/{models}/descent_length{grid_points}_round{round}.eps', bbox_inches='tight')
    for round in range(3):
        if grid_points == 100:
            name = [r'$\rm SigMA_{stride=1}$', r'$\rm SigMA_{stride=10}$', r'$\rm SigMA_{stride=50}$']
            plot_loss_descent(grid_points=grid_points, name=name, history=history[round])
        if grid_points != 100:
            name = [r'$\rm SigMA_{stride=50}$', fr'$\rm SigMA_{{stride={int(grid_points/2)}}}$']
            plot_loss_descent(grid_points=grid_points, name=name, history=history[round])
        plt.show()








