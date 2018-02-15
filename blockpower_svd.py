from __future__ import print_function
import numpy as np
from scipy import stats
from math import sqrt, log, ceil, floor
from sklearn.utils.extmath import svd_flip


def _to_fixed(f, precision=20):
    f = f * (1 << precision)
    return f.astype(np.int64)


def _from_fixed(x, precision=20):
    x = x / float((1 << precision))
    return x.astype(np.double)


def check_overflow(As, nbits=0):
    """
    Checks whether there is potential for an overflow, or returns max safe nbits
    Parameters
    ----------
    As : list of ndarrays
        List of symmetric ndarrays that represent what each party would be
        running on
    nbits : unsigned int, optional
        Number of bits to try. 0 by default

    Returns
    -------
    nbits_max : int
        The largest nbits safe to use
    """
    A = sum([A for A in As])
    max_norm_sq = np.sum(A**2)
    bits_s = np.log(max_norm_sq) / np.log(2)
    bits_a = nbits*2
    bits_max = 63
    nbits_max = floor((63-ceil(bits_s))/2)
    #print('bits_s={} max nbits={}'.format(bits_s, nbits_max))
    if bits_a + bits_s > bits_max:
        raise OverflowError('bits(max_norm_sq)={} bits(base)={}'.format(bits_s,
                                                             nbits))
    return int(nbits_max)

def private_distributed_block_power_iteration(As, k, T, nbits=18,
                                              random_seed=0):
    shapes = np.array([np.shape(A)[0] for A in As]+[np.shape(A)[1] for A in As])
    assert np.all(np.diff(shapes)==0)  # all As are square of the same shape
    check_overflow(As, nbits)
    n = As[0].shape[0]
    rtv = {}
    if random_seed:
        np.random.seed(random_seed)

    N = stats.norm(0, 1.0 / n)
    V = N.rvs((n, k))

    for t in range(T):
        V0 = V
        V = np.zeros_like(V, dtype=np.int64)
        for A in As:
            V += _to_fixed(np.dot(A, V0), nbits)

        nv = np.sqrt((V ** 2).sum(0))
        V = V / nv.astype(np.double)
        V = _from_fixed(V, nbits)
        V, _ = np.linalg.qr(V)

    s = np.sqrt(sum([(np.dot(A, V) ** 2).sum(0) for A in As]))
    rtv['s'] = s
    rtv['V'] = V
    return rtv

def block_power_iteration(A, k, T, random_seed=0):
    n, m = A.shape
    assert n == m
    rtv = {}
    if random_seed:
        np.random.seed(random_seed)

    N = stats.norm(0, 1.0 / n)
    V = N.rvs((n, k))

    for t in range(T):
        V = np.dot(A, V)

        nv = np.sqrt((V ** 2).sum(0))
        V = V / nv
        V, _ = np.linalg.qr(V)

    s = np.sqrt((np.dot(A, V) ** 2).sum(0))
    rtv['s'] = s
    rtv['V'] = V
    return rtv


# variables needed to remember how to undo block construction
A_m = None
A_n = None


def mat_to_sym(X, method='mult'):
    global A_m, A_n
    if method == 'mult':
        return np.dot(X.T, X)
    elif method == 'block':
        A_m, A_n = X.shape
        up = np.hstack((np.zeros((A_m, A_m)), X))
        btm = np.hstack((X.T, np.zeros((A_n, A_n))))
        B = np.vstack((up
                       ,
                       btm
                       )
                      )
        return B
    else:
        raise Exception('method {} not recognized'.format(method))


def sym_eigen_to_mat_singular(V, s, method='mult'):
    """returns V, sigmas"""
    if method == 'mult':
        V, s = V, np.sqrt(s)
    elif method == 'block':
        rank = np.argwhere(s > 1e-15).size

        assert rank % 2 == 0  # rank should be even, else we have a
        # non-convergence issue
        ind_vo = 0
        Vo = np.empty((A_n, rank / 2))
        for i in range(0, s.size - 1, 2):
            j = i
            if i == 0:
                if np.abs(np.inner(V[A_m:, 1], V[A_m:, 1]) - 1) < np.abs(
                                np.inner(V[A_m:, 0], V[A_m:, 0]) - 1):
                    j = 1
            else:
                viol_ni = np.abs(np.inner(V[A_m:, i], V[A_m:, i]) - 1)
                viol_nj = np.abs(np.inner(V[A_m:, i + 1], V[A_m:, i + 1]) - 1)
                if viol_ni < viol_nj / 10:
                    pass
                elif viol_ni > viol_nj * 10:
                    j = i + 1
                else:
                    ipi = 0
                    ip1 = 0
                    for k in range(i / 2 - 1):
                        ipi += np.abs(np.inner(V[A_m:, i], Vo[:, k]))
                        ip1 += np.abs(np.inner(V[A_m:, i + 1], Vo[:, k]))
                    if ip1 < ipi:
                        j = i + 1
            if s[j] < 1e-15:  # no more eigenvalues
                continue
            Vo[:, ind_vo] = V[A_m:, j]
            ind_vo += 1

        s = s[0:rank:2]
        V, s = Vo, s
    else:
        raise Exception('method {} not recognized'.format(method))
    k = V.shape[1]
    _, V = svd_flip(np.eye(k), V.T, u_based_decision=False)
    V = V.T
    return V, s


def private_power_iteration(A, T, eps=np.inf, delta=0, coh_ub=np.inf,
                            diagnostics=[], random_seed=0):
    rtv = setup_diagnostics(diagnostics)

    # define sigma based on delta, T, and eps
    if np.isfinite(eps):
        sigma = 2 * (1 / eps) * sqrt(4 * T * log(1.0 / delta))
    else:
        sigma = 0.0

    n, m = A.shape
    assert n == m

    # pass random_seed=None to skip setting it
    if random_seed:
        np.random.seed(random_seed)

    # noise for initialization
    N0 = stats.norm(0, 1.0 / n)
    x = N0.rvs(n)
    # noise added in each iteration
    Ni = stats.norm(0, (coh_ub * sigma ** 2) / n)

    for t in range(T):
        # (a) check coherence fail condition
        if np.argmax(x) ** 2 > coh_ub / n:
            raise ValueError('Coherence upper bound violated')
        # (b) draw noise
        if sigma > 0:
            g = Ni.rvs(n)
        else:
            g = 0
        # (c) power iteration
        x_new = np.dot(A, x) + g
        # Ax = x_new => Ax-x_new is small
        # Ax = -x_new => Ax+x_new is small

        if np.sum(x + x_new) > np.sum(x - x_new):
            x_new *= -1
        x = x_new
        # (d) normalize
        x = x / np.linalg.norm(x, ord=2)

        # capture diagnostics
        rtv = run_diagnostics(diagnostics, rtv, locals())
    rtv['x'] = x
    return rtv


print_deflation_results = False
def private_top_k_eigenvectors(A, k, T, eps=np.inf, delta=0, coh_ub=np.inf,
                               diagnostics=[], random_seed=0):
    n, m = A.shape
    assert n == m
    rtv = {}
    if random_seed:
        np.random.seed(random_seed)

    # copy A since we will be modifying it
    A = A.copy()

    if np.isfinite(eps):
        eps = eps / sqrt(4 * k * log(1 / delta))
    delta = delta / k
    V = np.empty((n, k))
    s = np.empty((k,))

    L = stats.laplace(0, 1 / eps)

    for i in range(k):
        V[:, i] = private_power_iteration(A, T, eps=eps, delta=delta,
                                          coh_ub=coh_ub, random_seed=None)['x']

        s[i] = np.linalg.norm(np.dot(A, V[:, i]), 2)
        if np.isfinite(eps):
            s[i] += L.rvs(1)

        # can the deflation step be done privately?
        if print_deflation_results:
            strue = np.linalg.svd(A, compute_uv=False)
            print('{}\nDeflating {}'.format(strue, s[i]))
        A = A - s[i] * np.outer(V[:, i], V[:, i])
        if print_deflation_results:
            strue = np.linalg.svd(A, compute_uv=False)
            print('{}'.format(strue))
    rtv['V'] = V
    rtv['s'] = s
    return rtv


def setup_diagnostics(diagnostics, rtv={}):
    if type(diagnostics) is not list:
        diagnostics = [diagnostics]

    if len(diagnostics) > 0:
        rtv['diagnostics'] = {}
        for (i, func) in enumerate(diagnostics):
            rtv['diagnostics'][func.func_name] = []
    return rtv


def run_diagnostics(diagnostics, rtv, kwargs):
    if len(diagnostics) > 0:
        for func in diagnostics:
            rtv['diagnostics'][func.func_name].append(func(**kwargs))
    return rtv
