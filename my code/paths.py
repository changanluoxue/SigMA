import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
import random


def fBm_paths(grid_points, M, H, T):
    """
    This function genrates M trajectories of the process $W^H = \int_0^t (t-s)^{H-0.5} dW_s$ with Cholesky decomposition.
    Source: https://github.com/amuguruza/RoughFCLT/blob/master/rDonsker.ipynb
    Args:
        - grid_points: # points in the simulation grid
        - H: Hurst Index
        - T: time horizon
        - M: # paths to simulate
    """
    assert 0 < H < 1.0 # 判断H的区间是否合理
    # Step1: create partition
    X = np.linspace(0, 1, num=grid_points)
    X = X[1:grid_points]# get rid of starting point
    # Step 2: compute covariance matrix
    Sigma = np.zeros((grid_points-1, grid_points-1))
    for j in range(grid_points-1):
        for i in range(grid_points-1):
            if i == j:
                Sigma[i, j] = np.power(X[i], 2*H)/2/H
            else:
                s = np.minimum(X[i], X[j])
                t = np.maximum(X[i], X[j])
                Sigma[i, j] = np.power(t-s, H-0.5)/(H+0.5)*np.power(s, 0.5+H)*special.hyp2f1(0.5-H, 0.5+H, 1.5+H, -s/(t-s))
    # Step 3: compute Cholesky decomposition
    Cholesky = np.linalg.cholesky(Sigma)
    # Step 4: draw Gaussian random variable
    GV = np.random.normal(loc=0.0, scale=1.0, size=[M, grid_points-1])
    # Step 5: get W^H
    fBms = np.zeros((M, grid_points))
    for i in range(M):
        fBms[i, 1:grid_points] = np.dot(Cholesky, GV[i, :])
    # Use self-similarity to extend to [0,T]
    return fBms * np.power(T, H)
def rBergomi_paths(grid_points, M, H, T, eta, V_0):
    '''
    This function genrates M trajectories of the rBergomi volatility process.
    '''
    assert 0 < H < 1.0
    assert V_0 > 0
    # 生成时间网格
    X = np.linspace(0, T, num=grid_points)
    # 生成分数阶布朗运动
    fBms = fBm_paths(grid_points, M, H, T)
    # 生成波动率过程
    Z = eta*np.sqrt(2*H)*fBms
    V = V_0*np.exp(Z-eta**2/2*X**(2*H))
    return V
def generate_paths(n_paths_train, n_paths_eval, grid_points, Hs, T, eta, V_0):
    def generate_rBergomi(n_paths, grid_points, Hs):
        X = np.zeros((n_paths, grid_points))
        Y = np.zeros((n_paths, 1))
        for i in range(n_paths):
            Y[i, 0] = random.choice(Hs)
            # X[i, :] = rBergomi_paths(grid_points, M=1, H=Y[i, 0], T=T, eta=eta, V_0=V_0)
            X[i, :] = fBm_paths(grid_points, M=1, H=Y[i, 0], T=T)
            print(f'已生成第 {i + 1} 条路径...')
        return X, Y
    print('******开始生成训练集******')
    X_train, Y_train = generate_rBergomi(n_paths_train, grid_points, Hs)
    print('******训练集生成完毕******')
    print('******开始生成验证集******')
    X_test, Y_test = generate_rBergomi(n_paths_eval, grid_points, Hs)
    print('******验证集生成完毕******')
    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    Hs_discrete = [0.1, 0.2, 0.3, 0.4, 0.5]
    Hs_uniform = [0.05, 0.18, 0.29, 0.31, 0.44]
    Hs_beta = [0.02, 0.07, 0.06, 0.13, 0.22]
    grid_points = 500; T = 1
    n_paths_train = 2800; n_paths_eval = 700
    eta = 1; V_0 = 0.01
    X_train, Y_train, X_eval, Y_eval = generate_paths(n_paths_train, n_paths_eval, grid_points, Hs_uniform, T, eta, V_0)
    np.save('data/simulated_data/fBm/Xtrain_grid=500_Huniform.npy', X_train)
    np.save('data/simulated_data/fBm/Ytrain_grid=500_Huniform.npy', Y_train)
    np.save('data/simulated_data/fBm/Xeval_grid=500_Huniform.npy', X_eval)
    np.save('data/simulated_data/fBm/Yeval_grid=500_Huniform.npy', Y_eval)