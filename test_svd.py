from __future__ import print_function

from blockpower_svd import block_power_iteration, \
    private_distributed_block_power_iteration, mat_to_sym, \
    sym_eigen_to_mat_singular
from sklearn_interface import BlockIterSVD
from learning_outcome_expm import gen_samples


import numpy as np
from sklearn.utils.extmath import svd_flip
from math import exp, log, ceil
import pytest

import sys

if sys.version_info[0] == 3:
    from functools import reduce


def test_gen_samples():
    np.random.seed(0)
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 11, 10)
    combined_data, stacked = gen_samples([X, y], 3, 0)
    ind = 0

    for i, comb in enumerate(combined_data):
        ni = comb[0].shape[0]
        for j, arr in enumerate(comb):

            assert np.allclose(arr, stacked[j][ind:(ind+ni), :])
        ind += ni

    combined_data, stacked = gen_samples([X], 3, 0)
    ind = 0

    for i, comb in enumerate(combined_data):

        if type(comb)==list:
            ni = comb[0].shape[0]
            for j, arr in enumerate(comb):
                assert np.allclose(arr, stacked[j][ind:(ind + ni), :])
        else:
            ni = comb.shape[0]

            assert np.allclose(comb, stacked[ind:(ind + ni), :])
        ind += ni

@pytest.mark.parametrize('M',
                         [1,3,30])
def test_embedding_quality(M, n=60, d=100, seed=0):
    np.random.seed(0)
    X = np.random.randn(n, d)
    k = 10
    s = np.linalg.svd(X, compute_uv=False)
    #error as computed by Eckart-Young-Mirsky theorem
    err = np.sqrt(np.sum(s[k:]**2))
    tol = 1e-5

    BISVD = BlockIterSVD(k=k, n_parties=M)
    Xh = BISVD.fit_transform(X)
    Xh = BISVD.inverse_transform(Xh)
    recons = np.linalg.norm(Xh - X, 'fro')
    total = np.linalg.norm(X, 'fro')
    print('recons ={} err={} total={}\n'.format(recons, err, total))

    assert np.abs(recons-err) <= tol


@pytest.mark.parametrize(('n', 'd', 'k', 'M', 'nbits', 'seed'),
                         [
                             (50, 30, 7, 1, 23, 0),
                             (10, 200, 7, 5, 20, 0)
                         ]
                         )
def test_convergence_to_np_linalg_int_precision(n, d, k, M, nbits, seed):
    np.random.seed(seed)

    Xs = [np.random.randn(n, d) for _ in range(M)]
    # Xs = [Xi-np.mean(Xi, 0) for Xi in Xs]
    X = reduce(lambda x, y: np.vstack((x, y)), Xs)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    _, Vt = svd_flip(U, Vt, u_based_decision=False)
    V = Vt.T
    D = np.diff(np.log(s))
    i = np.argmax(D[0:(k + 1)])
    max_spectral_ratio = exp(D[i])

    eps = 1e-10
    # the maximum initial distance between V and a random vector is 2
    # solve for n_iter: 2*(max_ratio)^n_iter = eps
    n_iter = int(ceil(log(eps / 2.0) / log(max_spectral_ratio)))
    print(n_iter)
    sym_method = 'mult'

    for j in range(1):
        Ss = [mat_to_sym(Xi, sym_method) for Xi in Xs]

        rtv = private_distributed_block_power_iteration(Ss, k, n_iter,
                                                        nbits=nbits,
                                                        random_seed=j)
        V_est, s_est = rtv['V'], rtv['s']
        V_est, s_est = sym_eigen_to_mat_singular(V_est, s_est, sym_method)
        deltas = frob_err_by_vector(V, V_est)
        print(np.max(deltas))
        assert np.all(deltas <= 5e-7)


@pytest.mark.parametrize(('n', 'd', 'k', 'seed'),
                         [
                             (50, 30, 7, 0),
                             (10, 200, 7, 0),
                         ]
                         )
def test_convergence_to_np_linalg(n, d, k, seed):
    np.random.seed(seed)
    X = np.random.randn(n, d)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    _, Vt = svd_flip(U, Vt, u_based_decision=False)
    V = Vt.T
    D = np.diff(np.log(s))
    i = np.argmax(D[0:(k + 1)])
    max_spectral_ratio = exp(D[i])

    eps = 1e-10
    # the maximum initial distance between V and a random vector is 2
    # solve for n_iter: 2*(max_ratio)^n_iter = eps
    n_iter = int(ceil(log(eps / 2.0) / log(max_spectral_ratio)))
    print(n_iter)
    sym_method = 'mult'

    for j in range(1):
        S = mat_to_sym(X, sym_method)
        rtv = block_power_iteration(S, k, n_iter, random_seed=j)
        V_est, s_est = rtv['V'], rtv['s']
        V_est, s_est = sym_eigen_to_mat_singular(V_est, s_est, sym_method)
        deltas = frob_err_by_vector(V, V_est)
        print(np.max(deltas))
        assert np.all(deltas <= eps)


def frob_err_by_vector(V, V_est, *args, **kwargs):
    k = V_est.shape[1]
    return np.sqrt(np.sum((V[:, 0:k] - V_est) ** 2, 0))
