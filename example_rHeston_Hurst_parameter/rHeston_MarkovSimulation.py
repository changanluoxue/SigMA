import numpy as np
import scipy.interpolate
import scipy.special
import scipy.stats
from scipy.special import ndtri
from scipy.stats.qmc import Sobol
from numpy.random import default_rng
import psutil
from scipy.optimize import minimize, lsq_linear
from scipy.special import gamma, gammainc
# import orthopy
# import quadpy
from packages import candle
import fbm
import iisignature
import random
from packages import siglayer
import sklearn.base as base
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
from scipy.linalg import hankel


# RoughKernel.py
def sort(a, b):
    """
    Sorts two numpy arrays jointly according to the ordering of the first.
    :param a: First numpy array
    :param b: Second numpy array
    :return: Sorted numpy arrays
    """
    perm = np.argsort(a)
    return a[perm], b[perm]
def rel_err(x, x_approx):
    """
    Computes the absolute relative error of x_approx compared to x.
    :param x: The true value
    :param x_approx: The approximated value
    :return: The absolute relative error
    """
    return np.abs((x - x_approx) / x)
def single_param_search(f, rel_tol=1e-03, n=100, factor=2):
    """
    Finds the optimal parameter n for approximating f.
    :param f: A function having two inputs n and reusable. The input n is some discretization parameter. For example,
        f might be the solution of an ODE and n is the number of time steps. The input reusable is used to supply
        previously computed information, so that it does not have to be computed again. For example, if f is the
        trapezoidal integral of some function, this may be an array of previously computed function values that can
        be reused without needing to recompute them. If there is nothing sensible that could be reused, just leave it as
        a dead parameter. The function gives two outputs a and b. The output a is the result (e.g. the final point of
        the ODE solution), and the output b is the reusable information. In the next call of f, b will be given as the
        parameter reusable.
    :param rel_tol: Relative error tolerance of the result
    :param n: Initial parameter n
    :param factor: Factor by which we should multiply n if we need higher accuracy.
    :return: The approximated result, the final parameter n that was used, and the reusable information
    """
    int_calc = isinstance(n, int)
    approx_res, reusable = f(n=n // factor if int_calc else n / factor, reusable=None)
    current_res, reusable = f(n=n, reusable=reusable)
    while rel_err(current_res, approx_res) > rel_tol:
        n = int(factor * n) if int_calc else factor * n
        approx_res = current_res
        current_res, reusable = f(n=n, reusable=reusable)
    return current_res, n, reusable
def exp_underflow(x):
    """
    Computes exp(-x) while avoiding underflow errors.
    :param x: Float of numpy array
    :return: exp(-x)
    """
    if isinstance(x, np.ndarray):
        if x.dtype == int:
            x = x.astype(np.float)
        eps = np.finfo(x.dtype).tiny
    else:
        if isinstance(x, int):
            x = float(x)
        eps = np.finfo(x.__class__).tiny
    log_eps = -np.log(eps) / 2
    result = np.exp(-np.fmin(x, log_eps))
    result = np.where(x > log_eps, 0, result)
    return result
def fractional_kernel(H, t):
    """
    The fractional kernel.
    :param H: Hurst parameter
    :param t: Time, may also be a numpy array
    :return: The value of the fractional kernel at t
    """
    return t ** (H - 0.5) / gamma(H + 0.5)
def kernel_norm(H, T, p=2.):
    """
    Returns the L^p-norm of the fractional kernel.
    :param H: Hurst parameter
    :param T: Final time
    :param p: The order of the norm
    :return: The L^p-norm (root has been taken) of the fractional kernel
    """
    return T ** (H - 0.5 + 1 / p) / (gamma(0.5 + H) * (1 + p * H - p / 2) ** (1 / p))
def c_H(H):
    """
    Returns the constant c_H.
    :param H: Hurst parameter
    :return: c_H
    """
    return 1. / (gamma(0.5 + H) * gamma(0.5 - H))
def fractional_kernel_laplace(H, t, nodes):
    """
    The Laplace transform of the fractional kernel.
    :param H: Hurst parameter
    :param t: Time, may also be a numpy array
    :param nodes: Laplace transform argument, may also be a numpy array
    :return: The Laplace transform. May be a number, a one-dimensional or a two-dimensional numpy array depending on
        the shape of t and nodes. If both t and nodes are a numpy array, the tensor product that we take is
        nodes x time
    """
    if isinstance(t, np.ndarray) and isinstance(nodes, np.ndarray):
        return c_H(H) * exp_underflow(np.tensordot(nodes, t, axes=0))
    return c_H(H) * exp_underflow(nodes * t)
def fractional_kernel_approximation(H, t, nodes, weights):
    """
    Returns the Markovian approximation of the fractional kernel.
    :param H: Hurst parameter
    :param t: Time points
    :param nodes: Nodes of the quadrature rule
    :param weights: Weights of the quadrature rule
    :return: The approximated kernel using nodes and weights at times t (a numpy array)
    """
    return 1 / c_H(H) * np.tensordot(fractional_kernel_laplace(H, t, nodes), weights, axes=([0, 0]))
def AK_improved_rule(H, N, K=None, T=1.):
    """
    The quadrature rule from Alfonsi and Kebaier in Table 6, left column.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param K: Cutoff point where the regime changes
    :param T: Final time
    :return: The quadrature rule in the form nodes, weights
    """
    if N == 1:
        return np.array([0.]), np.array([0.])

    N = N // 2

    if K is None:
        K = N ** 0.8

    def AK_initial_guess(A_):
        partition = np.empty(2 * N + 1)
        partition[:N + 1] = np.linspace(0, K, N + 1)
        partition[N + 1:] = K * A_ ** np.arange(1, N + 1)
        a = partition ** (1.5 - H)
        b = partition ** (0.5 - H)
        nodes_ = (0.5 - H) / (1.5 - H) * (a[1:] - a[:-1]) / (b[1:] - b[:-1])
        weights_ = c_H(H) / (0.5 - H) * (b[1:] - b[:-1])
        return nodes_, weights_

    def error_func(A_):
        nodes_, weights_ = AK_initial_guess(A_[0])
        return error_l2(H, nodes_, weights_, T)

    res = minimize(fun=lambda A_: error_func(A_), x0=np.array([1.2]), bounds=((0, None),))
    A = res.x
    nodes, weights = AK_initial_guess(A[0])

    res = minimize(fun=lambda x: error_l2(H, nodes, x * weights, T), x0=np.array([1]), bounds=((0, None),))
    return nodes, res.x * weights
def AbiJaberElEuch_quadrature_rule(H, N, T):
    """
    Computes the quadrature as suggested in "Multi-factor approximation of rough volatility models" by Abi Jaber and
    El Euch.
    :param H: Hurst parameter
    :param N: Number of quadrature nodes
    :param T: Maturity / Final time
    :return: The nodes and weights, two numpy arrays
    """
    pi_n = N ** (-0.2) / T * (np.sqrt(10) * (1 - 2 * H) / (5 - 2 * H)) ** 0.4
    eta = pi_n * np.arange(N + 1)
    c_vec = (eta[1:] ** (0.5 - H) - eta[:-1] ** (0.5 - H)) / (gamma(H + 0.5) * gamma(1.5 - H))
    gamma_vec = (eta[1:] ** (1.5 - H) - eta[:-1] ** (1.5 - H)) / ((1.5 - H) * gamma(H + 0.5) + gamma(0.5 - H)) / c_vec
    return gamma_vec, c_vec
def Gaussian_parameters(H, N, T, mode):
    """
    Returns the parameters of the Gaussian quadrature rule.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param T: Final time
    :param mode: The kind of theorem or observation that should be used
    :return: The partition of the middle part, and the quadrature level m
    """
    if ' geometric ' in mode or mode == "OLD" or mode == "GG":
        if mode == "old geometric theorem l2":
            N = N - 1
            A = np.sqrt(1 / H + 1 / (1.5 - H))
            beta = 0.4275
            alpha = 1.06418
            gamma_ = np.exp(alpha * beta)
            exponent = 1 / (3 * gamma_ / (8 * (gamma_ - 1)) + 6 * H - 4 * H * H)
            temp_1 = ((9 - 6 * H) / (2 * H)) ** (gamma_ / (8 * (gamma_ - 1)))
            temp_2 = 5 * np.pi ** 3 * gamma_ * (gamma_ - 1) * A ** (2 - 2 * H) * float(N) ** (1 - H) / (
                        beta ** (2 - 2 * H))
            base_0 = temp_1 * (temp_2 * (3 - 2 * H) / (768 * H)) ** (2 * H)
            a = 1 / T * base_0 ** exponent * np.exp(-alpha / ((1.5 - H) * A) * np.sqrt(N))
            base_n = temp_1 * (temp_2 / 1152) ** (2 * H - 3)
            b = 1 / T * base_n ** exponent * np.exp(alpha / (H * A) * np.sqrt(N))
            m = int(np.fmax(np.round(beta / A * np.sqrt(N)), 1))
            n = int(np.round(N / m))
        elif mode == "old geometric observation l2" or mode == "OLD":
            N = N - 1
            A = np.sqrt(1 / H + 1 / (1.5 - H))
            beta = 0.9
            alpha = 1.8
            a = 0.65 * 1 / T * np.exp(3.1 * H) * np.exp(-alpha / ((1.5 - H) * A) * np.sqrt(N))
            b = 1 / T * np.exp(3 * H ** (-0.4)) * np.exp(alpha / (H * A) * np.sqrt(N))
            m = int(np.fmax(np.round(beta / A * np.sqrt(N)), 1))
            n = int(np.round(N / m))
        elif mode == "new geometric theorem l1" or mode == "GG":
            beta = 1
            alpha = np.log(3 + 2 * np.sqrt(2))
            a = 4 / T
            b = 1 / 2 / T * np.exp(alpha / np.sqrt(H + 0.5) * np.sqrt(N))
            m = int(np.fmax(np.round(beta * np.sqrt((H + 0.5) * N)), 1))
            n = int(np.round(N / m)) - 1
        else:
            raise NotImplementedError(f'The mode {mode} has not been implemented')

        partition = np.exp(np.log(a) + np.log(b / a) * np.linspace(0, 1, n + 1))
    else:
        if mode == 'non-geometric l1' or mode == "NGG":
            beta = 0.92993273
            a = 3 / T
            m = int(np.fmax(np.round(beta * np.sqrt((H + 0.5) * N)), 1))
            c = 3.60585021
        else:
            raise NotImplementedError(f'The mode {mode} has not been implemented')

        kappa = 1 / (2 * beta ** 2)
        n = int(np.round(N / m)) - 1
        partition = np.empty(n + 1)
        partition[0] = a
        for i in range(n):
            partition[i + 1] = partition[i] \
                * ((c + partition[i] ** (kappa / (n + 1))) / (c - partition[i] ** (kappa / (n + 1)))) ** 2
    return partition, m
def Gaussian_interval(H, m, a, b, fractional_weight=True):
    """
    Returns the nodes and weights of the Gauss quadrature rule of level m on [a, b].
    :param H: Hurst parameter
    :param m: Level of the Gaussian quadrature
    :param a: Left end of interval
    :param b: Right end of interval
    :param fractional_weight: If True, computes the Gaussian quadrature rule with respect to the fractional weight. If
        False, computes the Gaussian quadrature with respect to the weight function w(x) = c_H
    :return: The nodes and weights
    """
    if fractional_weight:
        k = np.arange(2 * m) + 0.5 - H
    else:
        k = np.arange(1, 2 * m + 1)
    alpha, beta, int_1 = orthopy.tools.chebyshev(moments=c_H(H) / k * (b ** k - a ** k))
    return quadpy.tools.scheme_from_rc(alpha, beta, int_1)
def Gaussian_on_partition(H, m, partition, fractional_weight=True):
    """
    Returns the nodes and weights of the Gaussian quadrature rule of level m on a partition.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param partition: The partition, where the Gaussian quadrature rule is applied on each interval
    :param fractional_weight: If True, computes the Gaussian quadrature rule with respect to the fractional weight
        function. If False, computes the Gaussian quadrature with respect to the weight function w(x) = c_H, and
        then multiplies the weights by nodes ** (-H - 1/2)
    :return: All the nodes and weights
    """
    nodes = np.empty(m * (len(partition) - 1))
    weights = np.empty(m * (len(partition) - 1))
    for i in range(len(partition) - 1):
        new_nodes, new_weights = Gaussian_interval(H=H, m=m, a=partition[i], b=partition[i + 1],
                                                   fractional_weight=fractional_weight)
        nodes[m * i:m * (i + 1)] = new_nodes
        weights[m * i:m * (i + 1)] = new_weights
    if not fractional_weight:
        weights = weights * nodes ** (-H - 0.5)
    return nodes, weights
def Gaussian_optimal_zero_weight(H, T, nodes, weights):
    """
    Computes the optimal weight in the L^2-approximation of an additional node at 0 given that we are already using the
    specified nodes and weights.
    :param H: Hurst parameter
    :param T: Final time
    :param nodes: The nodes of the Markovian approximation, a numpy array
    :param weights: The weights of the Markovian approximation, a numpy array
    :return: The optimal weight in the L^2-sense of an additional node at 0
    """
    if len(nodes) == 0:
        return T ** (H - 0.5) / gamma(H + 1.5)
    return (T ** (H + 0.5) / gamma(H + 1.5) - np.sum(weights / nodes * (1 - exp_underflow(nodes * T)))) / T
def Gaussian_rule(H, N, T, mode):
    """
    Returns the nodes and weights of the Gaussian rule with roughly N nodes.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time
    :param mode: The Gaussian parameters that should be used
    :return: The nodes and weights, ordered by the size of the nodes
    """
    if isinstance(T, np.ndarray):
        T = T[-1]
    partition, m = Gaussian_parameters(H, N, T, mode)

    if mode == 'old geometric theorem l2' or mode == 'old geometric observation l2':
        if N == 1:
            w_0 = Gaussian_optimal_zero_weight(H=H, T=T, nodes=np.array([]), weights=np.array([]))
            nodes, weights = np.array([0.]), np.array([w_0])
        else:
            nodes, weights = np.zeros(m * (len(partition) - 1) + 1), np.empty(m * (len(partition) - 1) + 1)
            nodes[1:], weights[1:] = Gaussian_on_partition(H=H, m=m, partition=partition, fractional_weight=True)
            weights[0] = Gaussian_optimal_zero_weight(H=H, T=T, nodes=nodes[1:], weights=weights[1:])
    else:
        nodes, weights = np.empty(m * len(partition)), np.empty(m * len(partition))
        nodes[:m], weights[:m] = Gaussian_interval(H=H, m=m, a=0, b=partition[0], fractional_weight=True)
        if len(partition) > 1:
            nodes[m:], weights[m:] = Gaussian_on_partition(H=H, m=m, partition=partition,
                                                           fractional_weight='old' in mode)
    return nodes, weights
def error_l2(H, nodes, weights, T, output='error'):
    """
    Computes an error estimate of the squared L^2-norm of the difference between the rough kernel and its approximation
    on [0, T].
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param weights: The weights of the approximation
    :param T: Final time, may also be a numpy array
    :param output: If error, returns the error. If gradient, returns the error and the gradient of the error
    :return: An error estimate
    """
    nodes = np.fmin(np.fmax(nodes, 1e-08), 1e+150)
    weights = np.fmin(weights, 1e+75)
    weight_matrix = np.outer(weights, weights)
    summand = T ** (2 * H) / (2 * H * gamma(H + 0.5) ** 2)
    node_matrix = nodes[:, None] + nodes[None, :]
    if isinstance(T, np.ndarray):
        gamma_ints = gammainc(H + 0.5, np.outer(T, nodes))
        nmT = np.einsum('i,jk->ijk', T, node_matrix)
        exp_node_matrix = exp_underflow(nmT)
        sum_1 = np.sum((weight_matrix / node_matrix)[None, :, :] * (1 - exp_node_matrix), axis=(1, 2))
        sum_2 = 2 * np.sum((weights / nodes ** (H + 0.5))[None, :] * gamma_ints, axis=1)
    else:
        gamma_ints = gammainc(H + 0.5, nodes * T)
        nmT = node_matrix * T
        exp_node_matrix = exp_underflow(nmT)
        sum_1 = np.sum(weight_matrix / node_matrix * (1 - exp_node_matrix))
        sum_2 = 2 * np.sum(weights / nodes ** (H + 0.5) * gamma_ints)
    err = summand + sum_1 - sum_2
    if output == 'error' or output == 'err':
        return err

    N = len(nodes)
    if isinstance(T, np.ndarray):
        grad = np.empty((len(T), 2 * N))
        exp_node_vec = exp_underflow(np.outer(T, nodes)) / nodes[None, :]
        first_summands = (weight_matrix / (node_matrix * node_matrix))[None, :] * (1 - (1 + nmT) * exp_node_matrix)
        second_summands = weights[None, :] * ((T ** (H + 1 / 2) / gamma(H + 1 / 2))[:, None] * exp_node_vec - (
                ((H + 1 / 2) * nodes ** (-H - 3 / 2))[None, :] * gamma_ints))
        grad[:, :N] = -2 * np.sum(first_summands, axis=2) - 2 * second_summands
        third_summands = np.einsum('ijk,k->ij', ((1 - exp_node_matrix) / node_matrix[None, :, :]), weights)
        forth_summands = (nodes ** (-(H + 1 / 2)))[None, :] * gamma_ints
        grad[:, N:] = 2 * third_summands - 2 * forth_summands
    else:
        grad = np.empty(2 * N)
        exp_node_vec = np.zeros(N)
        indices = nodes * T < 300
        exp_node_vec[indices] = np.exp(- T * nodes[indices]) / nodes[indices]
        first_summands = weight_matrix / (node_matrix * node_matrix) * (1 - (1 + nmT) * exp_node_matrix)
        second_summands = weights * (T ** (H + 1 / 2) / gamma(H + 1 / 2) * exp_node_vec - (H + 1 / 2) * nodes ** (
                -H - 3 / 2) * gamma_ints)
        grad[:N] = -2 * np.sum(first_summands, axis=1) - 2 * second_summands
        third_summands = ((1 - exp_node_matrix) / node_matrix) @ weights
        forth_summands = nodes ** (-(H + 1 / 2)) * gamma_ints
        grad[N:] = 2 * third_summands - 2 * forth_summands
    return err, grad
def error_l1(H, nodes, weights, T, method='intersections', tol=1e-08):
    """
    Computes an error estimate of the L^1-norm of the difference between the rough kernel and its approximation
    on [0, T].
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param weights: The weights of the approximation
    :param T: Final time, may also be a numpy array
    :param method: Method for computing the error
    :param tol: Relative error tolerance with which the error should be computed (relative error of the error)
    :return: An error estimate
    """
    if method == 'trapezoidal':
        def error_(n, reusable):
            t = np.linspace(0, T, n + 1)[1:]
            if reusable is None:
                reusable = np.abs(fractional_kernel(H, t) - fractional_kernel_approximation(H, t, nodes, weights))
            else:
                error_t_1 = np.empty(n)
                error_t_1[1::2] = reusable
                error_t_1[::2] = np.abs(fractional_kernel(H, t[1::2])
                                        - fractional_kernel_approximation(H, t[1::2], nodes, weights))
                reusable = error_t_1
            total_error = np.trapz(reusable, dx=T / n)
            return total_error + np.abs(fractional_kernel(H, T / (2 * n))
                                        - fractional_kernel_approximation(H, T / (2 * n), nodes, weights)) * T / n, \
                reusable
        return single_param_search(f=error_, rel_tol=tol, n=100, factor=2)[0:2]
    elif method == 'exact - trapezoidal':
        gamma_ = gamma(H + 0.5)

        def find_first_intersection():
            current_error_ = 10.
            current_t = 0.
            current_kernel_approximation = fractional_kernel_approximation(H=H, t=current_t, nodes=nodes,
                                                                           weights=weights)
            while current_error_ > tol and current_t < T:
                current_t = (current_kernel_approximation * gamma_) ** (1 / (H - 0.5))
                current_kernel_approximation = fractional_kernel_approximation(H=H, t=current_t, nodes=nodes,
                                                                               weights=weights)
                current_kernel = fractional_kernel(H=H, t=current_t)
                current_error_ = rel_err(current_kernel, current_kernel_approximation)
            return np.fmin(current_t, T)

        def error_(n, reusable):
            t = np.linspace(t_0, T, n + 1)
            if reusable is None:
                reusable = np.abs(fractional_kernel(H, t) - fractional_kernel_approximation(H, t, nodes, weights))
            else:
                error_t_1 = np.empty(n + 1)
                error_t_1[::2] = reusable
                error_t_1[1::2] = np.abs(fractional_kernel(H, t[1::2])
                                         - fractional_kernel_approximation(H, t[1::2], nodes, weights))
                reusable = error_t_1
            total_error = np.trapz(reusable, dx=(T - t_0) / n)
            return error_to_t_0 + total_error, reusable

        t_0 = find_first_intersection()
        error_to_t_0 = t_0 ** (H + 0.5) / (gamma_ * (H + 0.5)) \
            - np.sum(weights / nodes * (1 - exp_underflow(nodes * t_0)))
        if t_0 == T:
            return error_to_t_0
        return single_param_search(f=error_, rel_tol=tol, n=100, factor=2)[0:2]
    elif method == 'upper bound':
        gamma_ = gamma(H + 0.5)
        nm = nodes[:, None] + nodes[None, :]
        wm = weights[:, None] * weights[None, :]
        err = - 2 * gamma_ * gamma(1.5 + H) * np.sum(weights / nodes ** (1.5 + H) * gammainc(1.5 + H, nodes * T))\
            + gamma_ ** 2 * gamma(2) * np.sum(wm / nm ** 2 * gammainc(2, nm * T))
        return err, 0
    elif method == 'intersections':
        gamma_1 = gamma(H + 0.5)

        def step(t_, ker_, ker_approx_):
            nonlocal n_steps
            n_steps = n_steps + 1
            ker_larger = ker_ > ker_approx_
            rel_err_ = rel_err(ker_, ker_approx_)
            if rel_err_ > tol:
                d_ker = (H - 0.5) / gamma_1 * t_ ** (H - 1.5)
                d_ker_approx = - np.sum(weights * nodes * exp_underflow(nodes * t_))
                if ker_larger:
                    dd_ker_approx = np.sum(weights * nodes ** 2 * exp_underflow(nodes * t_))
                    return t_ + (d_ker - d_ker_approx
                                 + np.sqrt((d_ker - d_ker_approx) ** 2 - 2 * dd_ker_approx * (ker_approx_ - ker_))) \
                        / dd_ker_approx
                else:
                    dd_ker = (H - 0.5) * (H - 1.5) / gamma_1 * t_ ** (H - 2.5)
                    return t_ + (d_ker_approx - d_ker + np.sqrt((d_ker - d_ker_approx) ** 2
                                                                - 2 * dd_ker * (ker_ - ker_approx_))) / dd_ker
            else:
                t_1 = t_ + tol * ker_ / np.sum(weights * nodes * exp_underflow(nodes * t_))
                t_2 = t_ + tol * ker_ ** 2 / (tol * ker_ + ker_approx_) * gamma_1 / (0.5 - H) * t_ ** (1.5 - H)
                return np.fmin(t_1, t_2)

        def find_next_intersection(t_):
            ker_ = fractional_kernel(H=H, t=t_)
            ker_approx_ = fractional_kernel_approximation(H=H, t=t_, nodes=nodes, weights=weights)
            ker_larger = ker_ > ker_approx_
            t_old = t_
            while (ker_ > ker_approx_) == ker_larger and t_ < T:
                t_old = t_
                t_ = step(t_=t_, ker_=ker_, ker_approx_=ker_approx_)
                ker_ = fractional_kernel(H=H, t=t_)
                ker_approx_ = fractional_kernel_approximation(H=H, t=t_, nodes=nodes, weights=weights)
            return t_old, np.fmin(t_, T)

        n_steps = 0
        err = 0
        ker_approx = fractional_kernel_approximation(H=H, t=0, nodes=nodes, weights=weights)
        last_t = 0
        while last_t < T:
            t_left, t_right = find_next_intersection(t_=(ker_approx * gamma_1) ** (1 / (H - 0.5))
                                                     if last_t == 0 else last_t)
            this_t = (t_left + t_right) / 2 if t_right < T else T
            err = err + np.abs((this_t ** (H + 0.5) - last_t ** (H + 0.5)) / (gamma_1 * (H + 0.5))
                               - np.sum(weights / nodes * exp_underflow(last_t * nodes)
                                        * (1 - exp_underflow((this_t - last_t) * nodes))))
            last_t = this_t
        return err, n_steps
    elif method == 'reparametrized trapezoidal':

        def error_(n, reusable):
            t = np.linspace(0, T ** (0.5 + H), n + 1)[1:] ** (1 / (0.5 + H))
            if reusable is None:
                reusable = np.empty(n + 1)
                reusable[1:] = np.abs(1 / gamma(H + 1.5)
                                      - t ** (0.5 - H) * fractional_kernel_approximation(H, t, nodes,
                                                                                         weights) / (0.5 + H))
                reusable[0] = 1 / gamma(H * 1.5)
            else:
                error_t_1 = np.empty(n + 1)
                error_t_1[::2] = reusable
                error_t_1[1::2] = np.abs(1 / gamma(H + 1.5)
                                         - t[1::2] ** (0.5 - H) * fractional_kernel_approximation(H, t[1::2], nodes,
                                                                                                  weights) / (0.5 + H))
                reusable = error_t_1
            total_error = np.trapz(reusable, dx=T ** (0.5 + H) / n)
            return total_error, reusable

        return single_param_search(f=error_, rel_tol=tol, n=100, factor=2)[0:2]
    elif method == 'gaussian':
        return kernel_norm(H=H, T=T, p=1.) - np.sum(weights / nodes * (1 - exp_underflow(nodes * T)))
    else:
        raise NotImplementedError(f'The method {method} for computing the L^1 kernel error has not been implemented.')
def error_l2_optimal_weights(H, T, nodes, output='error'):
    """
    Computes an error estimate of the squared L^2-norm of the difference between the rough kernel and its approximation
    on [0, T]. Uses the best possible weights given the nodes specified.
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param output: If error, returns the error and the optimal weights. If gradient, returns the error, the gradient
        (of the nodes only), and the optimal weights. If hessian, returns the error, the gradient, the Hessian, and
        the optimal weights
    :param T: Final time, may also be a numpy array
    :return: An error estimate
    """
    if len(nodes) == 1:
        node = np.fmax(1e-04, nodes[0])
        gamma_1 = gamma(H + 0.5)

        if isinstance(T, np.ndarray):
            nT = node * T
            gamma_ints = gammainc(H + 0.5, nT)
            exp_node_matrix = exp_underflow(2 * nT)
            exp_node_vec = exp_underflow(nT)
            A = (1 - exp_node_matrix) / (2 * node)
            b = -2 * gamma_ints / node ** (H + 0.5)
            c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
            v = b / A
            err = c - 0.25 * b * v
            opt_weights = -0.5 * v
            if len(opt_weights.shape) > 1:
                opt_weights = opt_weights[-1, ...]
            if output == 'error' or output == 'err':
                return err, opt_weights

            A_grad = (-1 + (1 + 2 * nT) * exp_node_matrix) / (2 * node) ** 2
            b_grad = -2 * (nT ** (H + 0.5) * exp_node_vec[None, :] / gamma_1 - (H + 0.5) * gamma_ints) \
                / node ** (H + 1.5)
            grad = 0.5 * A_grad * v ** 2 - 0.5 * b_grad * v
            if output == 'gradient' or output == 'grad':
                return err, grad, opt_weights

            A_hess = 2 * (1 - (1 + 2 * nT + 2 * nT ** 2) * exp_node_matrix) / (8 * node ** 3)
            b_hess = -2 * (-(nT ** (H + 1.5) + (H + 1.5) * nT ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
                    H + 1.5) * gamma_ints) / nodes ** (H + 2.5)
            U = b_grad / A
            Y = 2 * A_grad * v
            hess = 0.5 * (2 * Y * U - Y ** 2 / A + 2 * A_hess * v ** 2 - b_hess * v - b_grad * U)
            return err, grad, hess, opt_weights

        gamma_ints = gammainc(H + 0.5, node * T)
        exp_node_matrix = exp_underflow(2 * node * T)
        exp_node_vec = exp_underflow(node * T)
        A = (1 - exp_node_matrix) / (2 * node)
        b = -2 * gamma_ints / node ** (H + 0.5)
        if H > 0:
            c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
            v = b / A
            err = c - 0.25 * b * v
            opt_weight = np.array([-0.5 * v])
            if output == 'error' or output == 'err':
                return err, opt_weight
        else:
            v = b / A
            err = - 0.25 * b * v
            opt_weight = np.array([-0.5 * v])
            if output == 'error' or output == 'err':
                return err, opt_weight

        A_grad = (-1 + (1 + 2 * node * T) * exp_node_matrix) / (4 * node ** 2)
        b_grad = -2 * ((node * T) ** (H + 0.5) * exp_node_vec / gamma_1 - (H + 0.5) * gamma_ints) / node ** (H + 1.5)
        grad = 0.5 * (A_grad * v - b_grad) * v
        if output == 'gradient' or output == 'grad':
            return err, grad, opt_weight

        A_hess = 2 * (1 - (1 + 2 * node * T + 2 * (node * T) ** 2) * exp_node_matrix) / (8 * node ** 3)
        b_hess = -2 * (-((node * T) ** (H + 1.5) + (H + 1.5) * (node * T) ** (H + 0.5)) * exp_node_vec / gamma_1
                       + (H + 0.5) * (H + 1.5) * gamma_ints) / node ** (H + 2.5)
        U = b_grad / A
        Y = 2 * A_grad * v
        hess = 0.5 * (2 * Y * U - Y ** 2 / A + 2 * A_hess * v ** 2 - b_hess * v - b_grad * U)
        return err, grad, hess, opt_weight

    def invert_permutation(p):
        s = np.empty_like(p)
        s[p] = np.arange(p.size)
        return s

    perm = np.argsort(nodes)
    nodes = nodes[perm]
    nodes[0] = np.fmax(1e-04, nodes[0])
    for i in range(len(nodes) - 1):
        if 1.01 * nodes[i] > nodes[i + 1]:
            nodes[i + 1] = nodes[i] * 1.01
    nodes = nodes[invert_permutation(perm)]

    node_matrix = nodes[:, None] + nodes[None, :]
    gamma_1 = gamma(H + 0.5)

    if isinstance(T, np.ndarray):
        nT = np.outer(T, nodes)
        nmT = np.einsum('i,jk->ijk', T, node_matrix)
        gamma_ints = gammainc(H + 0.5, nT)
        exp_node_matrix = exp_underflow(nmT)
        exp_node_vec = exp_underflow(nT)
        A = (1 - exp_node_matrix) / node_matrix[None, :, :]
        b = -2 * gamma_ints / nodes[None, :] ** (H + 0.5)
        c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
        try:
            v = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            v = np.empty((len(T), len(nodes)))
            for i in range(len(T)):
                try:
                    v[i, :] = np.linalg.solve(A[i, ...], b[i, ...])
                except np.linalg.LinAlgError:
                    v[i, :] = np.linalg.lstsq(A[i, ...], b[i, ...], rcond=None)[0]
        err = c - 0.25 * np.sum(b * v, axis=1)
        opt_weights = -0.5 * v
        if len(opt_weights.shape) > 1:
            opt_weights = opt_weights[-1, ...]
        if output == 'error' or output == 'err':
            return err, opt_weights

        def mvp(A_, b_):
            return np.sum(A_ * b_[:, None, :], axis=-1)

        A_grad = (-1 + (1 + nmT) * exp_node_matrix[None, :, :]) / node_matrix[None, :, :] ** 2
        b_grad = -2 * (nT ** (H + 0.5) * exp_node_vec[None, :] / gamma_1 - (H + 0.5) * gamma_ints) \
            / nodes[None, :] ** (H + 1.5)
        grad = 0.5 * v * mvp(A_grad, v) - 0.5 * b_grad * v
        if output == 'gradient' or output == 'grad':
            return err, grad, opt_weights

        def diagonalize(x):
            new_x = np.empty((x.shape[0], x.shape[1], x.shape[1]))
            for k in range(x.shape[0]):
                new_x[k, :, :] = np.diag(x[k, :])
            return new_x

        def trans(x):
            return np.transpose(x, (0, 2, 1))

        A_hess = 2 * (1 - (1 + nmT + nmT ** 2 / 2) * exp_node_matrix[None, :, :]) / node_matrix[None, :, :] ** 3
        b_hess = -2 * (-(nT ** (H + 1.5) + (H + 1.5) * nT ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
                    H + 1.5) * gamma_ints) / nodes[None, :] ** (H + 2.5)
        try:
            U = np.linalg.solve(A, diagonalize(b_grad))
        except np.linalg.LinAlgError:
            diag_b = diagonalize(b_grad)
            U = np.empty((len(T), len(nodes), len(nodes)))
            for i in range(len(T)):
                for j in range(len(nodes)):
                    try:
                        U[i, j, :] = np.linalg.solve(A[i, ...], diag_b[i, j, :])
                    except np.linalg.LinAlgError:
                        U[i, j, :] = np.linalg.lstsq(A[i, ...], diag_b[i, j, :])[0]
        Y = diagonalize(mvp(A_grad, v)) + A_grad * v[:, None, :]
        YTU = trans(Y) @ U
        hess = 0.5 * (YTU - trans(np.linalg.solve(A, Y)) @ Y + diagonalize(v * mvp(A_hess, v))
                      + v[:, None, :] * v[:, :, None] * A_hess - diagonalize(b_hess * v) - b_grad[:, :, None] * U
                      + trans(YTU))
        return err, grad, hess, opt_weights

    nT = nodes * T
    nmT = node_matrix * T
    gamma_ints = gammainc(H + 0.5, nT)
    exp_node_matrix = exp_underflow(nmT)
    A = (1 - exp_node_matrix) / node_matrix
    b = -2 * gamma_ints / nodes ** (H + 0.5)
    c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        v = np.linalg.lstsq(A, b, rcond=None)[0]
    if np.amax(v) > 0:
        v = lsq_linear(A, b).x
    err = 0.25 * v @ A @ v - 0.5 * np.dot(b, v) + c  # c - 0.25 * np.dot(b, v)
    opt_weights = -0.5 * v
    if output == 'error' or output == 'err':
        return err, opt_weights

    exp_node_vec = exp_underflow(nT)
    A_grad = (-1 + (1 + nmT) * exp_node_matrix) / node_matrix ** 2
    b_grad = -2 * (nT ** (H + 0.5) * exp_node_vec / gamma_1 - (H + 0.5) * gamma_ints) / nodes ** (H + 1.5)
    grad = 0.5 * v * (A_grad @ v) - 0.5 * b_grad * v
    if output == 'gradient' or output == 'grad':
        return err, grad, opt_weights

    A_hess = 2 * (1 - (1 + nmT + nmT ** 2 / 2) * exp_node_matrix) / node_matrix ** 3
    b_hess = -2 * (-(nT ** (H + 1.5) + (H + 1.5) * nT ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
            H + 1.5) * gamma_ints) / nodes ** (H + 2.5)
    try:
        U = np.linalg.solve(A, np.diag(b_grad))
    except np.linalg.LinAlgError:
        U = np.linalg.lstsq(A, b, rcond=None)[0]
    Y = np.diag(A_grad @ v) + A_grad * v[None, :]
    YTU = Y.T @ U
    hess = 0.5 * (YTU - np.linalg.solve(A, Y).T @ Y + np.diag(v * (A_hess @ v)) + v[None, :] * v[:, None] * A_hess
                  - np.diag(b_hess * v) - b_grad[:, None] * U + YTU.T)
    return err, grad, hess, opt_weights
def optimize_error_l2(H, N, T, tol=1e-08, bound=None, method='gradient', force_order=False, init_nodes=None,
                      iterative=False):
    """
    Optimizes the L^2 error with N points for the fractional kernel. Always uses the best weights and only numerically
    optimizes over the nodes.
    :param H: Hurst parameter
    :param N: Number of points
    :param T: Final time, may be a numpy array (only if grad is False and fast is True)
    :param tol: Error tolerance
    :param bound: Upper bound on the nodes. If no upper bound is desired, use None
    :param method: If error, uses only the error estimates for optimizing over the nodes, and uses the optimizer
        L-BFGS-B. If gradient, uses also the gradient of the error with respect to the nodes, and uses the optimizer
        L-BFGS-B. If hessian, uses also the gradient and the Hessian of the error with respect to the nodes, and uses
        the optimizer trust-constr
    :param force_order: Forces the nodes to stay in order, i.e. not switch places. May improve numerical stability
    :param init_nodes: May specify a starting point for the nodes
    :param iterative: If True, starts with 1 node and iteratively solves the optimization problem before adding another
        node
    :return: The minimal error together with the associated nodes and weights.
    """
    error_fun = error_l2_optimal_weights
    all_errors = np.empty(1)

    if iterative and not init_nodes and N >= 2:
        all_errors = np.empty(N)
        init_nodes = np.empty(N)
        all_errors[:-1], init_nodes[:-1], _ = optimize_error_l2(H=H, N=N - 1, T=T, tol=tol, bound=bound, method=method,
                                                                force_order=force_order, init_nodes=None,
                                                                iterative=iterative)
        init_nodes[:-1] = init_nodes[:N - 1] / 1.03 ** np.fmin(np.arange(1, N) ** 2, 100)
        if bound is not None:
            init_nodes[N - 1] = np.fmax(bound, 10 * init_nodes[N - 2])
        else:
            init_nodes[N - 1] = 5 * init_nodes[N - 2]

    # get starting value and bounds for the optimization problem
    if init_nodes is None:
        if bound is None:
            bound = 1e+100
            nodes_, w = quadrature_rule(H, N, T, mode='old geometric observation l2')
            if N == 2:
                bound = np.fmax(bound, np.amax(nodes_))
            if len(nodes_) < N:
                nodes = np.zeros(N)
                nodes[:len(nodes_)] = nodes_
                for i in range(len(nodes_), N):
                    nodes[i] = nodes_[-1] * 10 ** (i - len(nodes_) + 1)
            else:
                nodes = nodes_[:N]
        else:
            nodes = np.exp(np.linspace(0, np.log(np.fmin(bound, 5. ** (np.fmin(140, N - 1)) / T)), N))
    else:
        if bound is None:
            bound = 1e+100
        nodes = init_nodes
    lower_bound = 1 / (10 * N * np.amin(T)) * ((0.5 - H) / 0.4) ** 2
    nodes = np.fmin(np.fmax(nodes, lower_bound), bound)
    bounds = ((np.log(lower_bound), np.log(bound)),) * N
    original_error, original_weights = error_fun(H=H, T=T, nodes=nodes, output='error')
    original_nodes = nodes.copy()

    # carry out the optimization
    if force_order:
        constraints = []
        for i in range(1, N):
            def jac_here(x):
                res_ = np.zeros(N)
                res_[i] = 1
                res_[i - 1] = -1
                return res_
            constraints = constraints + [{'type': 'ineq', 'fun': lambda x: x[i] - x[i - 1] - 0.3, 'jac': jac_here}]

        if method == 'error' or method == 'err':
            def func(x):
                return error_fun(H, T, np.exp(x), output='error')[0] \
                    * (1 + np.sum(np.exp(- (x[1:] - x[:-1]) ** 2 * 3 * N / (0.72 * np.log(5 / H) - np.log(T)))))

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, constraints=constraints)

        else:
            def func(x):
                err_, grad, _ = error_fun(H, T, np.exp(x), output='gradient')
                return err_, np.exp(x) * grad

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, constraints=constraints, jac=True)

    else:
        if method == 'error' or method == 'err':
            def func(x):
                return error_fun(H, T, np.exp(x), output='error')[0]

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds)

        elif method == 'gradient' or method == 'grad':
            def func(x):
                err_, grad, _ = error_fun(H, T, np.exp(x), output='gradient')
                return err_, np.exp(x) * grad

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, jac=True)

        else:
            def func(x):
                err_, grad, _ = error_fun(H, T, np.exp(x), output='gradient')
                return err_, np.exp(x) * grad

            def hess(x):
                _, grad, hessian, _ = error_fun(H, T, np.exp(x), output='hessian')
                return hessian * np.exp(x[None, :] + x[:, None]) + np.diag(grad * np.exp(x))

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, jac=True, hess=hess, method='trust-constr')

    # post-processing, ensuring that the results are of good quality
    nodes = np.exp(res.x)
    err, weights = error_fun(H=H, T=T, nodes=nodes, output='error')
    if H > 0:
        if err > 2 * np.fmax(original_error, 1e-9):
            # return np.sqrt(np.fmax(original_error, 0)) / kernel_norm(H, T), original_nodes, original_weights
            return np.sqrt(np.fmax(original_error, 0)), original_nodes, original_weights
        # all_errors[-1] = np.sqrt(np.fmax(err, 0)) / kernel_norm(H, T)
        all_errors[-1] = np.sqrt(np.fmax(err, 0))
    else:
        if err > 0.5 * original_error:
            # return np.sqrt(np.fmax(original_error, 0)) / kernel_norm(H, T), original_nodes, original_weights
            return original_error, original_nodes, original_weights
        # all_errors[-1] = np.sqrt(np.fmax(err, 0)) / kernel_norm(H, T)
        all_errors[-1] = err
    return all_errors, nodes, weights
def optimize_error_l1(H, N, T, iterative=False, init_nodes=None, init_weights=None):
    """
    Optimizes the L^1 error with N points for the fractional kernel.
    :param H: Hurst parameter
    :param N: Number of points
    :param T: Final time, may be a numpy array (only if grad is False and fast is True)
    :param iterative: If True, starts with one node and iteratively adds nodes, while always optimizing
    :param init_nodes: May specify a starting point for the nodes
    :param init_weights: May specify a starting point for the weights
    :return: The minimal relative error together with the associated nodes and weights.
    """
    def optimize_error_given_rule(nodes_1, weights_1):
        N_ = len(nodes_1)
        coefficient = 1 / kernel_norm(H=H, T=T, p=1.)
        rule = np.log(np.concatenate((nodes_1, weights_1)))

        def func(x):
            err_, grad = error_l1(H=H, nodes=np.exp(x[:N_]), weights=np.exp(x[N_:]), T=T, method='intersections')
            return coefficient * err_

        res = minimize(func, rule, tol=1e-04)
        nodes_1, weights_1 = sort(np.exp(res.x[:N_]), np.exp(res.x[N_:]))
        return res.fun, nodes_1, weights_1

    if not iterative:
        if init_nodes is not None and init_weights is not None:
            nodes_, weights_ = init_nodes, init_weights
        else:
            nodes_, weights_ = quadrature_rule(H=H, N=N, T=T, mode='non-geometric l1')
        if len(nodes_) < N:
            nodes = np.zeros(N)
            weights = np.zeros(N)
            nodes[:len(nodes_)] = nodes_
            weights[:len(weights_)] = weights_
            for i in range(len(nodes_), N):
                nodes[i] = nodes_[-1] * 2 ** (i - len(nodes_) + 1)
                weights[i] = weights_[-1]
        else:
            nodes = nodes_[:N]
            weights = weights_[:N]

        return optimize_error_given_rule(nodes, weights)

    if init_nodes is None or init_weights is None:
        nodes, weights = quadrature_rule(H=H, N=1, T=T, mode='non-geometric l1')
        err, nodes, weights = optimize_error_given_rule(nodes, weights)
    else:
        err, nodes, weights = -1, init_nodes, init_weights

    while len(nodes) < N:
        print(len(nodes))
        nodes = np.append(nodes, 2 * nodes[-1])
        weights = np.append(weights, np.amax(weights))
        err, nodes, weights = optimize_error_given_rule(nodes, weights)

    return err, nodes, weights
def european_rule(H, N, T):
    """
    Returns a quadrature rule that is optimized for pricing European options under the rough Heston model.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time/Maturity
    :return: Nodes and weights
    """

    def optimizing_func(N_, tol_, bound_):
        if N_ == 1:
            nod = np.array([1 / T])
        else:
            nod = np.empty(N_)
            if len(last_nodes) == N_:
                nod = last_nodes
            else:
                nod[:-1] = last_nodes
                nod[-1] = bound_
        nod = nod / 1.03 ** np.fmin(np.arange(1, N_ + 1) ** 2, 100)
        return optimize_error_l2(H=H, N=N_, T=T, tol=tol_, bound=bound_, method='gradient', force_order=False,
                                 init_nodes=nod)

    if H > 0:
        _, nodes, weights = optimizing_func(N_=1, tol_=1e-06, bound_=None)
    else:
        _, nodes, weights = optimize_error_l1(H=H, N=1, T=T)
    if N == 1:
        return nodes, weights

    L_step = 1.15
    bound = np.amax(nodes) / L_step
    current_N = 1
    last_nodes = nodes

    while current_N < N:
        increase_N = 0
        L_step = 1.15

        while increase_N < 2:
            bound = bound * L_step
            error_, nodes, weights = optimizing_func(N_=current_N+1, tol_=1e-07/current_N, bound_=bound)
            p = np.argsort(nodes)
            nodes = nodes[p]
            weights = weights[p]
            if np.amin(nodes[1:] / nodes[:-1]) < 1.4 or np.abs(np.amin(weights)) < 1e-02 \
                    or np.abs(np.amin(weights[1:] / weights[:-1])) < 0.4:
                increase_N = 0
                L_step = 1.15
            elif error_ < optimizing_func(N_=current_N, tol_=1e-07/current_N, bound_=bound)[0]:
                increase_N += 1
                if L_step > 1.06:
                    L_step = 1.05
                    bound = bound / 1.15
            else:
                increase_N = 0
                L_step = 1.15

        current_N = current_N + 1
        last_nodes = nodes

    if N >= 4:
        return nodes, weights
    if N == 2:
        L_4 = bound * 2
        L_5 = bound * 3
        L_6 = bound * 4
    else:  # N == 3
        L_4 = bound
        L_5 = bound * 1.25
        L_6 = bound * 1.5
    error_4, nodes_4, weights_4 = optimizing_func(N_=N, tol_=1e-08, bound_=L_4)
    error_5, nodes_5, weights_5 = optimizing_func(N_=N, tol_=1e-08, bound_=L_5)
    error_6, nodes_6, weights_6 = optimizing_func(N_=N, tol_=1e-08, bound_=L_6)
    if error_4 <= error_5 and error_4 <= error_6:
        return nodes_4, weights_4
    if error_5 <= error_6:
        return nodes_5, weights_5
    return nodes_6, weights_6
def harms_rule(H, n, m):
    """
    The quadrature rule for fBm proposed by Harms.
    :param H: Hurst parameter
    :param n: Number of Gaussian quadrature intervals
    :param m: Degree of Gaussian quadrature
    :return: The nodes and weights of the quadrature rule
    """
    alpha_, beta_, gamma_, delta_ = H + 1/2, m - 1, 1/2 - H, H
    r = delta_ * m / (1 - alpha_ - beta_ + delta_ + m)
    xi_0 = n ** (-r / gamma_)
    xi_n = n ** (r / delta_)
    xi = xi_0 * np.exp(np.log(xi_n / xi_0) * np.linspace(0, 1, n + 1))
    return Gaussian_on_partition(H=H, m=m, partition=xi, fractional_weight=True)
def quadrature_rule(H, N, T, mode="european"):
    """
    Returns the nodes and weights of a quadrature rule for the fractional kernel with Hurst parameter H. The nodes are
    sorted in increasing order.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param T: Final time
    :param mode: The kind of quadrature rule that should be used
    :return: All the nodes and weights, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    if isinstance(T, np.ndarray):
        if N == 1:
            T = np.amin(T) ** (3 / 5) * np.amax(T) ** (2 / 5)
        elif N == 2:
            T = np.amin(T) ** (1 / 2) * np.amax(T) ** (1 / 2)
        elif N == 3:
            T = np.amin(T) ** (1 / 3) * np.amax(T) ** (2 / 3)
        elif N == 4:
            T = np.amin(T) ** (1 / 4) * np.amax(T) ** (3 / 4)
        elif N == 5:
            T = np.amin(T) ** (1 / 6) * np.amax(T) ** (5 / 6)
        elif N == 6:
            T = np.amin(T) ** (1 / 10) * np.amax(T) ** (9 / 10)
        else:
            T = np.amax(T)

    if mode == "optimized l2" or mode == "OL2":
        nodes, weights = optimize_error_l2(H=H, N=N, T=T)[1:3]
    elif mode == "optimized l1" or mode == "OL1":
        nodes, weights = optimize_error_l1(H=H, N=N, T=T, iterative=True)[1:3]
    elif mode == "european" or mode == "BL2":
        nodes, weights = european_rule(H=H, N=N, T=T)
    elif mode == "abi jaber" or mode == "AE":
        nodes, weights = AbiJaberElEuch_quadrature_rule(H=H, N=N, T=T)
    elif mode == "alfonsi" or mode == "AK":
        nodes, weights = AK_improved_rule(H=H, N=N, T=T)
    elif mode == "paper" or mode == "OLD":
        nodes, weights = Gaussian_rule(H=H, N=N, T=T, mode="old geometric observation l2")
    else:
        nodes, weights = Gaussian_rule(H=H, N=N, T=T, mode=mode)
    weights[np.logical_and(nodes < 1, np.abs(weights) > 100)] = 0
    return sort(nodes, weights)

# rHestonMarkovSimulation.py
def get_necessary_memory(N, return_times, m):
    """
    Estimates the (square root of the) number of bytes needed to simulate the required number of paths.
    :param N: Dimension of the Markovian approximation. 0 if only the stock is saved
    :param return_times: Number of time steps that should be returned
    :param m: Number of sample paths
    :return: Square root of the number of bytes that are required for storing m paths with return_times time steps
        where the volatility has N dimensions
    """
    return 2.5 * np.sqrt(N + 2) * np.sqrt(return_times + 1) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)
def get_n_batches(N, return_times, m):
    """
    Returns the number of batches that need to be used to simulate m samples.
    :param N: Dimension of the Markovian approximation. 0 if only the stock is saved
    :param return_times: Number of time steps that should be returned
    :param m: Number of sample paths
    :return: Number of batches and number of samples per batch
    """
    available_memory = np.sqrt(psutil.virtual_memory().available) / 2
    necessary_memory = get_necessary_memory(N=N, return_times=return_times, m=m)
    n_batches = int(np.ceil((necessary_memory / available_memory) ** 2))
    m_batch = int(np.ceil(m / n_batches))
    return n_batches, m_batch
def samples(lambda_, nu, theta, V_0, T, nodes, weights, rho, S_0, r, m, N_time, sample_paths=False, return_times=None,
            vol_only=False, stock_only=False, euler=False, qmc=True, rng=None, rv_shift=False, verbose=0):
    """
    Simulates sample paths under the Markovian approximation of the rough Heston model.
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps for the simulation
    :param S_0: Initial stock price
    :param r: Interest rate
    :param m: Number of samples
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility and the components of the volatility
    :param return_times: Integer that specifies how many time steps are returned. Only relevant if sample_paths is True.
        E.g., N_time is 100 and return_times is 25, then the paths are simulated using 100 equispaced time steps, but
        only the 26 = 25 + 1 values at the times np.linspace(0, T, 26) are returned. May be used especially for storage
        saving reasons, as only these (in this case 26) values are ever stored. The number N_time must be divisible by
        return_times. If return_times is None, it is set to N_time, i.e. we return every time step that was simulated.
    :param vol_only: If True, simulates only the volatility process, not the stock price process
    :param stock_only: If True, returns only the stock price process. This saves memory
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param rng: Can specify a sampler to use for sampling the underlying random variables. If qmc is true, expects
        an instance of scipy.stats.qmc.Sobol() with the correctly specified dimension of the simulated random variables.
        If qmc is False, expects an instance of np.random.default_rng()
    :param rv_shift: Only relevant when using QMC. Can specify a shift by which the uniform random variables in [0,1)^d
        are drawn. When the random variables X are drawn from Sobol, instead uses (X + rv_shift) mod 1. If True,
        randomly generates such a random shift
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Numpy array of the simulations, and the rng that was used for generating the underlying random variables
    """
    if sample_paths is False:
        return_times = 1
    if return_times is None:
        return_times = N_time
    if N_time % return_times != 0:
        raise ValueError(f'The number of time steps for the simulation N_time={N_time} is not divisible by the number'
                         f'of time steps that should be returned return_times={return_times}.')
    saving_steps = N_time // return_times
    dt = T / N_time
    N = len(nodes)
    if N == 1:
        nodes = np.array([nodes[0], 2 * nodes[0] + 1])
        weights = np.array([weights[0], 0])
        N = 2
        one_node = True
    else:
        one_node = False

    if rng is None:
        if qmc:
            if vol_only:
                rng = Sobol(d=N_time, scramble=False)
            else:
                if euler:
                    rng = Sobol(d=2 * N_time, scramble=False)
                else:
                    rng = Sobol(d=3 * N_time, scramble=False)
        else:
            rng = np.random.default_rng()
    if isinstance(rv_shift, bool) and rv_shift:
        dim = N_time
        if not vol_only and euler:
            dim = 2 * N_time
        elif not vol_only and not euler:
            dim = 3 * N_time
        rv_shift = np.random.uniform(0, 1, dim)

    m_input = m  # the original input of how many samples should be simulated
    # m itself is the number of samples that we simulate
    # We always have m >= m_input. At the end, we discard the additionally simulated paths.

    """
    if qmc:
        if int(2 ** np.ceil(np.log2(m)) + 0.001) != m_input:
            print(f'Using QMC requires simulating a number m of samples that is a power of 2. The input m={m_input} '
                  f'is not a power of 2.')
    """

    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = get_necessary_memory(N=0 if stock_only else N, return_times=return_times, m=m)
    if necessary_memory > available_memory:
        raise MemoryError(f'Not enough memory to store the sample paths of the rough Heston model with '
                          f'{N} Markovian dimensions, {return_times} time points where the sample paths should be '
                          f'returned and {m} sample paths. Roughly {necessary_memory}**2 bytes needed, '
                          f'while only {available_memory}**2 bytes are available.')

    available_memory_for_random_variables = available_memory / 3
    if qmc:
        necessary_memory_for_random_variables = np.sqrt(3 * N_time) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)
    else:
        necessary_memory_for_random_variables = np.sqrt(3 * m) * np.sqrt(np.array([0.]).nbytes)
    n_batches = int(np.ceil((necessary_memory_for_random_variables / available_memory_for_random_variables) ** 2))
    m_batch = int(np.ceil(m / n_batches))
    m = m_batch * n_batches

    V_init = V_0 / nodes / (np.sum(weights / nodes))

    if euler:
        A = np.eye(N) + np.diag(nodes) * dt + lambda_ * weights[None, :] * dt
        A_inv = np.linalg.inv(A)
        b = theta * dt + (nodes * V_init)[:, None] * dt

        if vol_only:
            def step_SV(V_comp_, dW_):
                # sq_V = np.sqrt(np.fmax(weights @ V_comp_, 0))---------------------------------------------------------
                sq_V = 1
                # dW_samples = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                return A_inv @ (V_comp_ + nu * (sq_V * dW_)[None, :] + b)
        else:
            def step_SV(log_S_, V_comp_, dBW_):
                # sq_V = np.sqrt(np.fmax(weights @ V_comp_, 0))---------------------------------------------------------
                sq_V = 1
                # dW = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                # dB = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                log_S_ = log_S_ + r * dt + sq_V * (rho * dBW_[:, 1] + np.sqrt(1 - rho ** 2) * dBW_[:, 0]) \
                    - 0.5 * sq_V ** 2 * dt
                V_comp_ = A_inv @ (V_comp_ + nu * (sq_V * dBW_[:, 1])[None, :] + b)
                return log_S_, V_comp_

    else:
        weight_sum = np.sum(weights)
        A = -(np.diag(nodes) + lambda_ * weights[None, :]) * dt / 2
        exp_A = scipy.linalg.expm(A)
        b = (nodes * V_init + theta) * dt / 2
        ODE_b = np.linalg.solve(A, (exp_A - np.eye(N)) @ b)[:, None]
        z = weight_sum ** 2 * nu ** 2 * dt
        rho_bar_sq = 1 - rho ** 2
        rho_bar = np.sqrt(rho_bar_sq)

        def ODE_step_V(V_):
            return exp_A @ V_ + ODE_b

        B = (6 + np.sqrt(3)) / 4
        A = B - 0.75

        def SDE_step_V(V_, dW_):
            x = weights @ V_
            # dW_ = cf.rand_uniform(size=m, antithetic=antithetic)
            temp = np.sqrt((3 * z) * x + (B * z) ** 2)
            p_1 = (z / 2) * x * ((A * B - A - B + 1.5) * z + (np.sqrt(3) - 1) / 4 * temp + x) / (
                    (x + B * z - temp) * temp * (temp - (B - A) * z))
            p_2 = x / (1.5 * x + A * (B - A / 2) * z)
            test_1 = dW_ < p_1
            test_2 = p_1 + p_2 <= dW_
            x_step = A * z * np.ones(len(temp))
            x_step[test_1] = B * z - temp[test_1]
            x_step[test_2] = B * z + temp[test_2]
            return V_ + (x_step / weight_sum)[None, :]

        def step_V(V_, dW_):
            return ODE_step_V(SDE_step_V(ODE_step_V(V_), dW_))

        def SDE_step_B(log_S_, V_, dB_):
            # dB_ = cf.rand_normal(loc=0, scale=np.sqrt(dt / 2), size=m, antithetic=antithetic)
            x = weights @ V_
            return log_S_ + np.sqrt(x) * rho_bar * dB_ - (0.5 * rho_bar_sq * dt / 2) * x, V_

        drift_SDE_step_W = - (nodes[0] * V_init[0] + theta) * dt
        fact_1 = dt / 2 * (lambda_ - 0.5 * rho * nu)

        def SDE_step_W(log_S_, V_, dW_):
            V_new = step_V(V_, dW_)
            dY = V_ + V_new
            log_S_new = log_S_ + r * dt + rho / nu * (drift_SDE_step_W + (dt / 2 * nodes[0]) * dY[0, :]
                                                      + fact_1 * (weights @ dY) + (V_new[0, :] - V_[0, :]))
            return log_S_new, V_new

        if vol_only:
            def step_SV(V_, dW_):
                return step_V(V_, dW_)
        else:
            def step_SV(S_, V_, dBW_):
                return SDE_step_B(*SDE_step_W(*SDE_step_B(S_, V_, dBW_[:, 0]), dBW_[:, 2]), dBW_[:, 1])

    def generate_samples():
        if vol_only:
            if qmc:
                rv = rng.random(n=m_batch)
                if isinstance(rv_shift, np.ndarray):
                    rv = (rv + rv_shift) % 1.
                if euler:
                    if rv[0, 0] == 0.:
                        rv[1:, :] = np.sqrt(dt) * ndtri(rv[1:, :])  # first sample is 0 and cannot be inverted
                        rv[0, :] = 0.
                    else:
                        rv = np.sqrt(dt) * ndtri(rv)
                return lambda index: rv[:, index]
            else:
                if euler:
                    return lambda index: np.sqrt(dt) * rng.standard_normal(m_batch)
                else:
                    return lambda index: rng.uniform(0, 1, m_batch)
        else:
            if qmc:
                rv = rng.random(n=m_batch)
                if isinstance(rv_shift, np.ndarray):
                    rv = (rv + rv_shift) % 1.
                if euler:
                    if rv[0, 0] == 0.:
                        rv[1:, :] = np.sqrt(dt) * ndtri(rv[1:, :])  # first sample is 0 and cannot be inverted
                        rv[0, :] = 0.
                    else:
                        rv = np.sqrt(dt) * ndtri(rv)
                else:
                    if rv[0, 0] == 0.:  # first sample is 0 and cannot be inverted
                        rv[1:, :2 * N_time] = np.sqrt(dt / 2) * ndtri(rv[1:, :2 * N_time])
                        rv[0, :2 * N_time] = 0.
                    else:
                        rv[:, :2 * N_time] = np.sqrt(dt / 2) * ndtri(rv[:, :2 * N_time])
                return lambda index: rv[:, index::N_time]
            else:
                if euler:
                    return lambda index: np.sqrt(dt) * rng.standard_normal((m_batch, 2))
                else:
                    def rv_fun(index):
                        rv_vars = np.empty((m_batch, 3))
                        rv_vars[:, :2] = np.sqrt(dt / 2) * rng.standard_normal((m_batch, 2))
                        rv_vars[:, 2] = rng.uniform(0, 1, m_batch)
                        return rv_vars
                    return rv_fun

    if vol_only:
        result = np.empty((N + 1, return_times + 1, m)) if sample_paths else np.empty((N + 1, m))
        for j in range(n_batches):
            dW = generate_samples()
            current_V_comp = np.empty((N, m_batch))
            current_V_comp[:, :] = V_init[:, None]
            for i in range(N_time):
                if verbose >= 1:
                    print(f'Simulation round {j + 1} of {n_batches}, step {i + 1} of {N_time}')
                current_V_comp = step_SV(current_V_comp, dW(i))
                if sample_paths and (i + 1) % saving_steps == 0:
                    result[1:, (i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_V_comp
            if not sample_paths:
                result[1:, j * m_batch:(j + 1) * m_batch] = current_V_comp

        if sample_paths:
            result[1:, 0, :] = V_init[:, None]
            result[0, :, :] = np.fmax(np.einsum('i,ijk->jk', weights, result[1:, :, :]), 0)
        else:
            result[0, :] = np.fmax(np.einsum('i,ij->j', weights, result[1:, :]), 0)
    elif stock_only:
        result = np.empty((return_times + 1, m)) if sample_paths else np.empty(m)
        for j in range(n_batches):
            dBW = generate_samples()
            current_V_comp = np.empty((N, m_batch))
            current_V_comp[:, :] = V_init[:, None]
            current_log_S = np.full(m_batch, np.log(S_0))
            for i in range(N_time):
                if verbose >= 1:
                    print(f'Simulation round {j + 1} of {n_batches}, step {i + 1} of {N_time}')
                current_log_S, current_V_comp = step_SV(current_log_S, current_V_comp, dBW(i))
                if sample_paths and (i + 1) % saving_steps == 0:
                    result[(i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_log_S
            if not sample_paths:
                result[j * m_batch:(j + 1) * m_batch] = current_log_S
        if sample_paths:
            result[0, :] = np.log(S_0)
    else:
        result = np.empty((N + 2, return_times + 1, m)) if sample_paths else np.empty((N + 2, m))
        for j in range(n_batches):
            dBW = generate_samples()
            current_V_comp = np.empty((N, m_batch))
            current_V_comp[:, :] = V_init[:, None]
            current_log_S = np.full(m_batch, np.log(S_0))
            for i in range(N_time):
                if verbose >= 1:
                    print(f'Simulation round {j + 1} of {n_batches}, step {i + 1} of {N_time}')
                current_log_S, current_V_comp = step_SV(current_log_S, current_V_comp, dBW(i))
                if sample_paths and (i + 1) % saving_steps == 0:
                    result[0, (i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_log_S
                    result[2:, (i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_V_comp
            if not sample_paths:
                result[0, j * m_batch:(j + 1) * m_batch] = current_log_S
                result[2:, j * m_batch:(j + 1) * m_batch] = current_V_comp
        if sample_paths:
            result[0, 0, :] = np.log(S_0)
            result[2:, 0, :] = V_init[:, None]
            result[1, :, :] = np.fmax(np.einsum('i,ijk->jk', weights, result[2:, :, :]), 0)
        else:
            result[1, :] = np.fmax(np.einsum('i,ij->j', weights, result[2:, :]), 0)

    result = result[..., :m_input]
    if stock_only:
        result = np.exp(result)
    else:
        result[0, ...] = np.exp(result[0, ...])
        if one_node:
            result = result[:-1, ...]
    return result, rng

class AddTime(base.BaseEstimator, base.TransformerMixin):
    """Augments the path with time."""

    def __init__(self, init_time=0.):
        self.init_time = init_time

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        t = np.linspace(self.init_time, self.init_time + 1, len(X))
        return np.c_[t, X]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]

def generate_rHeston(n_paths, n_samples, hurst_exponents):
    """Generate FBM paths"""
    X = np.zeros((n_paths, n_samples+1))
    y = np.zeros(n_paths)
    for j in range(n_paths):
        y[j] = random.choice(hurst_exponents)
        nodes, weights = quadrature_rule(H=y[j], N=3, T=1)
        X[j,:] = samples(lambda_=1, nu=0.1, theta=0.3, V_0=0, T=1, nodes=nodes, weights=weights, rho=-0.7, S_0=100, r=0, m=1, N_time=n_samples, sample_paths=True, return_times=None,
            vol_only=False, stock_only=False, euler=True, qmc=False, rng=None, rv_shift=False, verbose=0)[0][1,:,0]
    return X, y
def generate_data(n_paths_train, n_paths_test, n_samples, hurst_exponents):
    """Generate train and test datasets"""

    # generate dataset
    x_train, y_train = generate_rHeston(n_paths_train, n_samples, hurst_exponents)
    x_test, y_test = generate_rHeston(n_paths_test, n_samples, hurst_exponents)

    # reshape targets
    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    return x_train, y_train, x_test, y_test


def preprocess_data(x_train, x_test, flag=None):
    """Peforms model-dependent preprocessing."""
    if flag == 'neuralsig':
        # We don't need to backprop through the signature if we're just building a model on top
        # so we actually perform the signature here as a feature transformation, rather than in
        # the model.
        path_transform = AddTime()
        x_train = np.array([iisignature.sig(x, 4) for x in path_transform.fit_transform(x_train)])
        x_test = np.array([iisignature.sig(x, 4) for x in path_transform.fit_transform(x_test)])
    elif flag == 'lstm':
        # LSTM wants another dimension in one place...
        x_train = np.expand_dims(x_train, 2)
        x_test = np.expand_dims(x_test, 2)
    else:
        # ...everyone else wants the extra dimension in another
        x_train = np.expand_dims(x_train, 1)
        x_test = np.expand_dims(x_test, 1)
    return x_train, x_test


def generate_torch_batched_data(x_train, y_train, x_test, y_test, train_batch_size, test_batch_size):
    """Generate torch dataloaders"""

    # make torch dataset
    train_dataset = torchdata.TensorDataset(torch.tensor(x_train, dtype=torch.float),
                                            torch.tensor(y_train, dtype=torch.float))
    test_dataset = torchdata.TensorDataset(torch.tensor(x_test, dtype=torch.float),
                                           torch.tensor(y_test, dtype=torch.float))

    # process with torch dataloader
    train_dataloader = torchdata.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    test_dataloader = torchdata.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    example_batch_x, example_batch_y = next(iter(train_dataloader))

    return train_dataloader, test_dataloader, example_batch_x, example_batch_y


def hurst_rescaled_range(ts):
    """Uses the rescaled range method to estimate the Hurst parameter."""

    # calculate standard deviation of differenced series using various lags
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # calculate Hurst as slope of log-log plot
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0] * 2.0
    return hurst
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, final_nonlinearity=lambda x: x, **kwargs):
        super(LSTM, self).__init__(**kwargs)

        self.mod = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.final = final_nonlinearity

    def forward(self, x):
        out, _ = self.mod(x)
        out = out[:, -1, :]
        return self.final(self.fc(out))
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, final_nonlinearity=lambda x: x, **kwargs):
        super(GRU, self).__init__(**kwargs)

        self.mod = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.final = final_nonlinearity

    def forward(self, x):
        out, _ = self.mod(x)
        out = out[:, -1, :]
        return self.final(self.fc(out))
# Now THIS is deep signatures!
def deep_recurrent(output_shape, sig=True, sig_depth=4, final_nonlinearity=lambda x: x,
                   augment_layer_sizes=(32, 32, 2), augment_kernel_size=8, augment_include_original=True,
                   augment_include_time=True,
                   lengths=(5, 5, 10), strides=(1, 1, 5), adjust_lengths=(0, 0, 0), memory_sizes=(8, 8, 8),
                   layer_sizes_s=((32,), (32,), (32, 16)), hidden_output_sizes=(8, 8)):
    """This model stacks multiple layers of signatures on top of one another in a natural way.

    - Augment the features with something learnable
    - Slide a window across the augmented features
    - Take the signature of each window
    - Put this list of signatures back together to recover the path dimension
    - Apply an RNN across the path dimension, preserving the intermediate outputs, so the path dimension is preserved
    - Slide another window
    - Take another signature
    - Reassemble signatures along path dimension
    - Another RNN
    - ...
    - etc. for some number of times
    - ...
    - Slide another window
    - Take another signature
    - Reassemble signatures along path dimension
    - Another RNN; this time throw away intermediate outputs and just present the final output as the overall output.
    If :sig: is falsy then the signature layers will be replaced with flattening instead.
    It expects input tensors of three dimensions: (batch, channels, length).

    For a simpler example in the same vein, see siglayer.examples.create_windowed.

    Arguments:
        output_shape: The final output shape from the network.
        sig: Optional, whether to use signatures in the network. If True a signature will be applied between each
            window. If False then the output is simply flattened. Defaults to True.
        sig_depth: Optional. If signatures are used, then this specifies how deep they should be truncated to.
        final_nonlinearity: Optional. What final nonlinearity to feed the final tensors of the network through, e.g. a
            sigmoid when desiring output between 0 and 1. Defaults to the identity.
        augment_layer_sizes: Optional. A tuple of integers specifying the size of the hidden layers of the feedforward
            network that is swept across the input stream to augment it. May be set to the empty tuple to do no
            augmentation.
        augment_kernel_size: Optional. How far into the past the swept feedforward network (that is doing augmenting)
            should take inputs from. For example if this is 1 then it will just take data from a single 'time', making
            it operate in a 'pointwise' manner. If this is 2 then it will take the present and the most recent piece of
            past information, and so on.
        augment_include_original: Optional. Whether to include the original path in the augmentation.
        augment_include_time: Optional. Whether to include an increasing 'time' parameter in the augmentation.
        lengths, strides, adjust_lengths, memory_sizes: Optional. Should each be a tuple of integers, all of the same
            length as one another. The length of these arguments determines the number of windows; this length must be
            at least one. The ith values determine the length, stride and adjust_length arguments of the ith Window,
            and the size of the memory of the ith RNN.
        layer_sizes_s: Optional. Should be a tuple of the same length as lengths, strides, adjust_lengths,
            memory_sizes. Each element of the tuple should itself be a tuple of integers specifying the sizes of the
            hidden layers of each RNN.
        hidden_output_sizes: Optional. Should be a tuple of integers one shorter than the length of lengths, strides,
            adjust_lengths, memory_sizes. It determines the output size of each RNN. It is of a slightly shorter length
            because the final output size is actually already determined by the output_shape argument!
    """

    num_windows = len(lengths)
    assert num_windows >= 1
    assert len(strides) == num_windows
    assert len(adjust_lengths) == num_windows
    assert len(layer_sizes_s) == num_windows
    assert len(memory_sizes) == num_windows
    assert len(hidden_output_sizes) == num_windows - 1

    if sig:
        transformation = siglayer.Signature(depth=sig_depth)
    else:
        transformation = lambda x: candle.batch_flatten(x.contiguous())

    final_output_size = torch.Size(output_shape).numel()
    output_sizes = (*hidden_output_sizes, final_output_size)

    recurrent_layers = []
    for (i, length, stride, adjust_length, layer_sizes, memory_size, output_size
         ) in zip(range(num_windows), lengths, strides, adjust_lengths, layer_sizes_s, memory_sizes, output_sizes):

        window_layers = []
        for layer_size in layer_sizes:
            window_layers.append(layer_size)
            window_layers.append(F.relu)

        intermediate_outputs = (num_windows - 1 != i)

        recurrent_layers.append(candle.Window(length=length, stride=stride, adjust_length=adjust_length,
                                              transformation=transformation))
        recurrent_layers.append(candle.Recur(module=candle.CannedNet((candle.Concat(),
                                                                      *window_layers,
                                                                      memory_size + output_size,
                                                                      candle.Split((memory_size, output_size)))),
                                             memory_shape=(memory_size,),
                                             intermediate_outputs=intermediate_outputs))

    return candle.CannedNet((siglayer.Augment(layer_sizes=augment_layer_sizes,
                                              kernel_size=augment_kernel_size,
                                              include_original=augment_include_original,
                                              include_time=augment_include_time),
                             *recurrent_layers,
                             candle.View(output_shape),
                             final_nonlinearity))
