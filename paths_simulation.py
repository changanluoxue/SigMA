import numpy as np
import scipy.special as special
from scipy.linalg import hankel
import random
import math

def fBm_paths(grid_points, M, H, T):
    """
    This function generates M trajectories of the process $W^H_t = int_0^t (t-s)^{H-0.5}*dW_s$ with Cholesky decomposition.
    Source: https://github.com/amuguruza/RoughFCLT/blob/master/rDonsker.ipynb
    Args:
        - grid_points: # points in the simulation grid
        - H: Hurst Index
        - T: time horizon
        - M: # paths to simulate
    return:
    np.array((M, grid_points))
    """
    assert 0 < H < 1.0 # 判断H的区间是否合理
    'Step1: create partition'
    X = np.linspace(0, 1, num=grid_points)
    X = X[1:grid_points]# get rid of starting point
    'Step 2: compute covariance matrix'
    Sigma = np.zeros((grid_points-1, grid_points-1))
    for j in range(grid_points-1):
        for i in range(grid_points-1):
            if i == j:
                Sigma[i, j] = np.power(X[i], 2*H)/2/H
            else:
                s = np.minimum(X[i], X[j])
                t = np.maximum(X[i], X[j])
                Sigma[i, j] = np.power(t-s, H-0.5)/(H+0.5)*np.power(s, 0.5+H)*special.hyp2f1(0.5-H, 0.5+H, 1.5+H, -s/(t-s))
    'Step 3: compute Cholesky decomposition'
    Cholesky = np.linalg.cholesky(Sigma)
    'Step 4: draw Gaussian random variable'
    GV = np.random.normal(loc=0.0, scale=1.0, size=[M, grid_points-1])
    'Step 5: get W^H'
    fBms = np.zeros((M, grid_points))
    for i in range(M):
        fBms[i, 1:grid_points] = np.dot(Cholesky, GV[i, :])
    'Use self-similarity to extend to [0,T]'
    return fBms*np.power(T, H)
def fOU_paths(grid_points, M, H, T, alpha, mu, sigma, V_0):
    '''
    This function generates M trajectories of the fOU process $dV_t = -alpha(mu-V_t)*dt + sigma*dW_H_t$.
    '''
    assert 0 < H < 1.0
    assert V_0 > 0
    V = np.zeros((M, grid_points))
    dt = T/(grid_points-1)
    # 生成分数阶布朗运动增量
    fBms = fBm_paths(grid_points, M, H, T)
    dWH = np.diff(fBms, axis=1) #(M, grid_points-1)
    # 生成波动率过程
    V[:, 0] = V_0
    for i in range(1, grid_points):
        V[:, i] = V[:, i-1] - alpha*(mu - V[:, i-1])*dt + sigma*dWH[:, i-1]
    return V
def rHeston_paths(grid_points, M, H, T, kappa_1, kappa_2, theta, V_0, reps):
    '''
    This function generates M trajectories of the rHeston process.
    '''
    def myls(A, b, eps):
        (m, n) = np.shape(A)
        (U, S, V) = np.linalg.svd(A); V = V.T
        r = np.sum(S > eps)
        x = np.zeros(n)
        for i in range(r):
            x = x + (np.sum(b * U[:,i]) / S[i]) * V[:,i]
        res = np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b)
        return x, res
    def myls2(A, b, eps):
        (m, n) = np.shape(A)
        (Q, R) = np.linalg.qr(A)
        s = np.diag(R); r = np.sum(abs(s) > eps)
        Q = Q[:, 0:r]; R = R[0:r, 0:r]
        b1 = b[r:m + r]
        x = np.dot(np.linalg.inv(R), (np.dot(Q.T, b1)))
        return x
    def prony(xs, ws):
        M = len(xs); errbnd = 1e-12; h = np.zeros(2 * M)
        for j in range(2 * M):
            h[j] = np.dot(xs ** j, ws)
        C = np.zeros(M); R = np.zeros(M)
        for i in range(M):
            C[i] = h[i]; R[i] = h[i + M - 1]
        H = hankel(C, R); b = -h
        q = myls2(H, b, errbnd); r = len(q); A = np.zeros((2 * M, r))
        Coef = np.insert(np.flipud(q), 0, 1)
        xsnew = np.roots(Coef)
        for j in range(2 * M):
            A[j, :] = xsnew ** j
        (wsnew, res) = myls(A, h, errbnd); ind = np.where(np.real(xsnew) >= 0); p = len(ind[0])
        assert np.sum(abs(wsnew[ind]) < 1e-15) == p
        ind = np.where(np.real(xsnew) < 0)
        xsnew = xsnew[ind]; wsnew = wsnew[ind]
        return wsnew, xsnew
    def SOEapppr(beta, reps, dt, Tfinal):
        delta = dt / Tfinal
        h = 2 * math.pi / (math.log(3) + beta * math.log(1 / math.cos(1)) + math.log(1 / reps))
        tlower = 1 / beta * math.log(reps * math.gamma(1 + beta))
        if beta >= 1:
            tupper = math.log(1 / delta) + math.log(math.log(1 / reps)) + math.log(beta) + 1 / 2
        else:
            tupper = math.log(1 / delta) + math.log(math.log(1 / reps))
        M = math.floor(tlower / h); N = math.ceil(tupper / h)
        xs1 = np.zeros(abs(M)); ws1 = np.zeros(abs(M))
        for n1 in range(M, 0):
            xs1[n1 - M] = -math.exp(h * n1); ws1[n1 - M] = h / math.gamma(beta) * math.exp(beta * h * n1)
        (ws1new, xs1new) = prony(xs1, ws1)
        xs2 = np.zeros(N + 1); ws2 = np.zeros(N + 1)
        for n2 in range(N + 1):
            xs2[n2] = -math.exp(h * n2)
            ws2[n2] = h / math.gamma(beta) * math.exp(beta * h * n2)
        xs = np.append(-np.real(xs1new), -np.real(xs2)); ws = np.append(np.real(ws1new), np.real(ws2))
        xs = xs / Tfinal; ws = ws / Tfinal ** beta
        nexp = len(ws)
        return xs, ws, nexp
    def rH_miuv(kappa_1, theta, V):
        miuv = kappa_1 * (theta-V)
        return miuv
    def rH_sigmav(kappa_2, V):
        sigmav = kappa_2 * np.sqrt(V)
        return sigmav
    assert 0 < H < 1.0
    assert V_0 > 0
    alpha = 0.5-H; dt = T/(grid_points-1)
    V = np.zeros((M, grid_points)); V[:, 0] = V_0
    (xl, wl, nexp) = SOEapppr(alpha, reps, dt, dt * grid_points)
    H = np.zeros((M, nexp)); J = np.zeros((M, nexp))
    B = np.random.normal(size = [M, grid_points])
    for step in range(1, grid_points):
        I1 = rH_miuv(kappa_1, theta, V[:, step-1]) * (dt ** (1 - alpha)) / math.gamma(2 - alpha) + (1 / math.gamma(1 - alpha)) * np.sum(wl * np.exp(-xl * dt) * H, 1)
        I2 = rH_sigmav(kappa_2, V[:, step-1]) * (dt ** (0.5 - alpha)) * B[:, step] / math.gamma(1 - alpha) + (1 / math.gamma(1 - alpha)) * np.sum(wl * np.exp(-xl * dt) * J, 1)
        V[:, step] = np.maximum(V_0 + I1 + I2, 0)
        H = np.exp(-xl * dt) * H + ((1 - np.exp(-xl * dt)) / xl) * np.reshape(rH_miuv(kappa_1, theta, V[:, step-1]), (M, 1))
        J = np.exp(-xl * dt) * J + np.exp(-xl * dt) * np.sqrt(dt) * np.reshape(B[:, step] * rH_sigmav(kappa_2, V[:, step-1]), (M, 1))
    return V
def generate_fBm_paths_single(n_paths_train, n_paths_eval, grid_points, Hs, T):
    def generate_fBm_single(n_paths, grid_points, Hs):
        X = np.zeros((n_paths, grid_points)); Y = np.zeros((n_paths, 1))
        for i in range(n_paths):
            Y[i, 0] = random.choice(Hs)
            X[i, :] = fBm_paths(grid_points, M=1, H=Y[i, 0], T=T)
            print(f'已生成第 {i+1} 条路径...')
        return X, Y
    print('******开始生成训练集******')
    X_train, Y_train = generate_fBm_single(n_paths_train, grid_points, Hs)
    print('******训练集生成完毕******')
    print('******开始生成验证集******')
    X_test, Y_test = generate_fBm_single(n_paths_eval, grid_points, Hs)
    print('******验证集生成完毕******')
    return X_train, Y_train, X_test, Y_test
def generate_fOU_paths_single(n_paths_train, n_paths_eval, grid_points, Hs, T, alpha, mu, sigma, V_0):
    def generate_fOU_single(n_paths, grid_points, Hs, alpha, mu, sigma):
        X = np.zeros((n_paths, grid_points)); Y = np.zeros((n_paths, 1))
        for i in range(n_paths):
            Y[i, 0] = random.choice(Hs)
            X[i, :] = fOU_paths(grid_points, M=1, H=Y[i, 0], T=T, alpha=alpha, mu=mu, sigma=sigma, V_0=V_0)
            print(f'已生成第 {i+1} 条路径...')
        return X, Y
    print('******开始生成训练集******')
    X_train, Y_train = generate_fOU_single(n_paths_train, grid_points, Hs, alpha, mu, sigma)
    print('******训练集生成完毕******')
    print('******开始生成验证集******')
    X_test, Y_test = generate_fOU_single(n_paths_eval, grid_points, Hs, alpha, mu, sigma)
    print('******验证集生成完毕******')
    return X_train, Y_train, X_test, Y_test
def generate_fOU_paths_multiple(n_paths_train, n_paths_eval, grid_points, Hs, T, alphas, mus, sigmas, V_0):
    def generate_fOU_multiple(n_paths, grid_points, Hs, alphas, mus, sigmas):
        X = np.zeros((n_paths, grid_points)); Y = np.zeros((n_paths, 4))
        for i in range(n_paths):
            Y[i, 0] = random.choice(Hs); Y[i, 1] = random.choice(alphas);
            Y[i, 2] = random.choice(mus); Y[i, 3] = random.choice(sigmas)
            X[i, :] = fOU_paths(grid_points, M=1, H=Y[i, 0], T=T, alpha=Y[i, 1], mu=Y[i, 2], sigma=Y[i, 3], V_0=V_0)
            print(f'已生成第 {i+1} 条路径...')
        return X, Y
    print('******开始生成训练集******')
    X_train, Y_train = generate_fOU_multiple(n_paths_train, grid_points, Hs, alphas, mus, sigmas)
    print('******训练集生成完毕******')
    print('******开始生成验证集******')
    X_test, Y_test = generate_fOU_multiple(n_paths_eval, grid_points, Hs, alphas, mus, sigmas)
    print('******验证集生成完毕******')
    return X_train, Y_train, X_test, Y_test
def generate_rHeston_paths_single(n_paths_train, n_paths_eval, grid_points, T, V_0, reps, Hs, kappa1, kappa2, theta):
    def generate_rHeston_single(n_paths, grid_points, Hs, kappa1, kappa2, theta):
        X = np.zeros((n_paths, grid_points)); Y = np.zeros((n_paths, 1))
        for i in range(n_paths):
            Y[i, 0] = random.choice(Hs)
            X[i, :] = rHeston_paths(grid_points, M=1, H=Y[i, 0], T=T, kappa_1=kappa1, kappa_2=kappa2, theta=theta, V_0=V_0, reps=reps)
            print(f'已生成第 {i + 1} 条路径...')
        return X, Y
    print('******开始生成训练集******')
    X_train, Y_train = generate_rHeston_single(n_paths_train, grid_points, Hs, kappa1, kappa2, theta)
    print('******训练集生成完毕******')
    print('******开始生成验证集******')
    X_test, Y_test = generate_rHeston_single(n_paths_eval, grid_points, Hs, kappa1, kappa2, theta)
    print('******验证集生成完毕******')
    return X_train, Y_train, X_test, Y_test
def generate_rHeston_paths_multiple(n_paths_train, n_paths_eval, grid_points, T, V_0, reps, Hs, kappa1s, kappa2s, thetas):
    def generate_rHeston_multiple(n_paths, grid_points, Hs, kappa1s, kappa2s, thetas):
        X = np.zeros((n_paths, grid_points)); Y = np.zeros((n_paths, 4))
        for i in range(n_paths):
            Y[i, 0] = random.choice(Hs); Y[i, 1] = random.choice(kappa1s);
            Y[i, 2] = random.choice(thetas); Y[i, 3] = random.choice(kappa2s)
            X[i, :] = rHeston_paths(grid_points, M=1, H=Y[i, 0], T=T, kappa_1=Y[i, 1], kappa_2=Y[i, 2], theta=Y[i, 3], V_0=V_0, reps=reps)
            print(f'已生成第 {i + 1} 条路径...')
        return X, Y
    print('******开始生成训练集******')
    X_train, Y_train = generate_rHeston_multiple(n_paths_train, grid_points, Hs, kappa1s, kappa2s, thetas)
    print('******训练集生成完毕******')
    print('******开始生成验证集******')
    X_test, Y_test = generate_rHeston_multiple(n_paths_eval, grid_points, Hs, kappa1s, kappa2s, thetas)
    print('******验证集生成完毕******')
    return X_train, Y_train, X_test, Y_test

types = 'single' # 'single', 'multiple'
models = 'fBm' # 'fBm', 'fOU' , 'rHeston'

Hs_fBm = [0.14, 0.23, 0.56, 0.67, 0.89] # Uniform~(0, 1)
Hs_fOU = [0.55, 0.66, 0.72, 0.85, 0.94] # Uniform~(0.5, 1)
Hs_rHeston = [0.04, 0.08, 0.11, 0.16, 0.21] # Beta~(1, 9)

grid_points = 100 # 100, 500, 1000, 1500
n_paths_train = 3000; n_paths_eval = 1000
T = 1; V_0 = 0.01; reps = 1e-6
alpha = 0.5; mu = 0.15; sigma = 0.2
kappa1 = 0.1; kappa2 = 0.03; theta = 0.3

alphas = np.round(np.random.uniform(0.01, 5, 1000), decimals=2) # Uniform~(0, 5)
mus = np.round(np.random.uniform(0.01, 0.5, 1000), decimals=2) # Uniform~(0, 0.5)
sigmas = np.round(np.random.uniform(0.01, 3, 1000), decimals=2) # Uniform~(0, 3)
kappa1s = np.round(np.random.uniform(0.01, 5, 1000), decimals=2) # Uniform~(0, 5)
thetas = np.round(np.random.uniform(0.01, 0.5, 1000), decimals=2) # Uniform~(0, 0.5)
kappa2s = np.round(np.random.uniform(0.01, 3, 1000), decimals=2) # Uniform~(0, 3)
Hs_uniform = np.round(np.random.uniform(0.51, 0.99, 1000), decimals=2) # Uniform~(0.5, 1)
Hs_beta = np.random.beta(1, 9, 1000) # Beta~(1, 9)

if types == 'single':
    if models == 'fBm':
        X_train, Y_train, X_eval, Y_eval = generate_fBm_paths_single(n_paths_train, n_paths_eval, grid_points, Hs_fBm, T)
    if models == 'fOU':
        X_train, Y_train, X_eval, Y_eval = generate_fOU_paths_single(n_paths_train, n_paths_eval, grid_points, Hs_fOU, T, alpha, mu, sigma, V_0)
    if models == 'rHeston':
        X_train, Y_train, X_eval, Y_eval = generate_rHeston_paths_single(n_paths_train, n_paths_eval, grid_points, T, V_0, reps, Hs_rHeston, kappa1, kappa2, theta)
if types == 'multiple':
    if models == 'fOU':
        X_train, Y_train, X_eval, Y_eval = generate_fOU_paths_multiple(n_paths_train, n_paths_eval, grid_points, Hs_uniform, T, alphas, mus, sigmas, V_0)
    if models == 'rHeston':
        X_train, Y_train, X_eval, Y_eval = generate_rHeston_paths_multiple(n_paths_train, n_paths_eval, grid_points, T, V_0, reps, Hs_beta, kappa1s, kappa2s, thetas)

np.save(f'data/simulated_data/{models}/Xtrain_{types}_grid={grid_points}.npy', X_train)
np.save(f'data/simulated_data/{models}/Ytrain_{types}_grid={grid_points}.npy', Y_train)
np.save(f'data/simulated_data/{models}/Xeval_{types}_grid={grid_points}.npy', X_eval)
np.save(f'data/simulated_data/{models}/Yeval_{types}_grid={grid_points}.npy', Y_eval)