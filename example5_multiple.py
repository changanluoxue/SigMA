import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from tabulate import tabulate
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
import NNmodels

'''自定义损失函数'''
def rmse_loss_fn(y_pre, y):
    return torch.sqrt(F.mse_loss(y_pre, y))

__name__ = "__results__" # "__training__", "__results__"

if __name__ == "__training__":
    start = time.time()
    '''参数输入'''
    models = 'fOU'  # 'fOU', 'rHeston'
    grid_points = 500  # 500
    max_epochs = 150
    lr = 0.0001; optimizer_fn=optim.Adam
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    '''导入数据集并处理'''
    X_train = np.load(f'data/simulated_data/{models}/Xtrain_multiple_grid={grid_points}.npy')
    Y_train = np.load(f'data/simulated_data/{models}/Ytrain_multiple_grid={grid_points}.npy')
    X_eval = np.load(f'data/simulated_data/{models}/Xeval_multiple_grid={grid_points}.npy')
    Y_eval = np.load(f'data/simulated_data/{models}/Yeval_multiple_grid={grid_points}.npy')
    X_train = np.expand_dims(X_train, 1); X_eval = np.expand_dims(X_eval, 1)
    train_dataloader, eval_dataloader, example_batch_x, example_batch_y = utils.generate_torch_batched_data(X_train, Y_train, X_eval, Y_eval,
                                                                                                            train_batch_size=64, test_batch_size=64, device=device)

    '''定义训练器'''
    model_trainer = utils.create_model_supervised_trainer(max_epochs=max_epochs, optimizer_fn=optimizer_fn,
                                                          loss_fn=rmse_loss_fn, train_dataloader=train_dataloader,
                                                          eval_dataloader=eval_dataloader, example_batch_x=example_batch_x, lr=lr, device=device)

    '''训练模型'''
    for round in range(10):
        history={}
        print('******开始训练Transformer******')
        transformer = NNmodels.transformer_multiple(augment_include_original=True, augment_include_time=True, T=1, grid_points=grid_points).to(device)
        model_trainer(transformer, 'Transformer', history)
        torch.save(transformer, f'data/results/numerical_example5/trained_models/{models}/transformer_multiple_round{round}.pth')
        print('******Transformer训练完成******')

        print('******开始训练CNN******')
        cnn = NNmodels.cnn_multiple().to(device)
        model_trainer(cnn, 'CNN', history)
        torch.save(cnn, f'data/results/numerical_example5/trained_models/{models}/cnn_multiple_round{round}.pth')
        print('******CNN训练完成******')

        print('******开始训练LSTM******')
        lstm = NNmodels.lstm_multiple(grid_points=grid_points).to(device)
        model_trainer(lstm, 'LSTM', history)
        torch.save(lstm, f'data/results/numerical_example5/trained_models/{models}/lstm_multiple_round{round}.pth')
        print('******LSTM训练完成******')

        print('******开始训练SigMA******')
        sigma = NNmodels.sigma_multiple(augment_include_original=True, augment_include_time=True, T=1, stride=int(grid_points/2)).to(device)
        model_trainer(sigma, 'SigMA', history)
        torch.save(sigma, f'data/results/numerical_example5/trained_models/{models}/sigma_multiple_round{round}.pth')
        print('******SigMA训练完成******')

        print('******开始训练DeepSigNet******')
        deepsignet = NNmodels.deepsignet_multiple(augment_include_original=True, augment_include_time=True, T=1).to(device)
        model_trainer(deepsignet, 'DeepSigNet', history)
        torch.save(deepsignet, f'data/results/numerical_example5/trained_models/{models}/deepsignet_multiple_round{round}.pth')
        print('******DeepSigNet训练完成******')

        with open(f'data/results/numerical_example5/history/{models}/history_multiple_round{round}.pkl', 'wb') as f:
            pickle.dump(history, f)

    end = time.time()
    print('---------------总耗时 {:.2f}s---------------'.format(end-start))

if __name__ == '__results__':
    def SE(input, target):
        return (input-target)**2
    start = time.time()
    '''参数输入'''
    models = 'rHeston'  # 'fOU', 'rHeston'
    grid_points = 500  # 500
    device = torch.device('cpu')
    '''导入数据集'''
    X_eval = np.load(f'data/simulated_data/{models}/Xeval_multiple_grid={grid_points}.npy'); X_eval = torch.from_numpy(X_eval).unsqueeze(1).float().to(device)
    Y_eval = np.load(f'data/simulated_data/{models}/Yeval_multiple_grid={grid_points}.npy'); Y_eval = torch.from_numpy(Y_eval).float().to(device)
    '''导入模型并计算误差'''
    for round in range(10):
        transformer = torch.load(f'data/results/numerical_example5/trained_models/{models}/transformer_multiple_round{round}.pth', weights_only=False, map_location=torch.device(device)); transformer.eval()
        cnn = torch.load(f'data/results/numerical_example5/trained_models/{models}/cnn_multiple_round{round}.pth', weights_only=False, map_location=torch.device(device)); cnn.eval()
        lstm = torch.load(f'data/results/numerical_example5/trained_models/{models}/lstm_multiple_round{round}.pth', weights_only=False, map_location=torch.device(device)); lstm.eval()
        sigma = torch.load(f'data/results/numerical_example5/trained_models/{models}/sigma_multiple_round{round}.pth', weights_only=False, map_location=torch.device(device)); sigma.eval()
        deepsignet = torch.load(f'data/results/numerical_example5/trained_models/{models}/deepsignet_multiple_round{round}.pth', weights_only=False, map_location=torch.device(device)); deepsignet.eval()

        nn_models = {'SigMA': sigma, 'LSTM': lstm, 'CNN': cnn, 'Transformer': transformer, 'Deepsignet': deepsignet}
        errors_se = {'Transformer': [], 'SigMA': [], 'Deepsignet': [], 'CNN': [], 'LSTM': []}
        for nn_model_name, nn_model in nn_models.items():
            predict = nn_model(X_eval)
            error_se = SE(predict, Y_eval).detach().numpy()
            errors_se[nn_model_name].append(error_se)
        np.savez(f'data/results/numerical_example5/errors/{models}/errors_se_round{round}.npz', **errors_se)
        print(f"已完成第{round}轮...")
    '''绘制误差表'''
    errors_average_rse = {'Transformer': [], 'SigMA': [], 'Deepsignet': [], 'CNN': [], 'LSTM': []}
    errors_average_rmse = {'Transformer': [], 'SigMA': [], 'Deepsignet': [], 'CNN': [], 'LSTM': []}
    for round in range(10):
        data = np.load(f'data/results/numerical_example5/errors/{models}/errors_se_round{round}.npz')
        print(f"{models}_Average_RMSE_round{round}:--------------")
        for nn_model_name in errors_average_rse:
            error_average_rse = np.mean(np.sqrt(data[nn_model_name].squeeze(0)), axis=1)
            error_average_rmse = np.mean(np.sqrt(np.mean(data[nn_model_name].squeeze(0), axis=0)))
            errors_average_rse[nn_model_name].append(error_average_rse)
            errors_average_rmse[nn_model_name].append(error_average_rmse)
            print(f"{nn_model_name:15}: {error_average_rmse:7.3f}")
    end = time.time()
    print('---------------总耗时 {:.2f}s---------------'.format(end - start))




