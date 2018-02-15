from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from math import ceil, log, exp

from blockpower_svd import private_distributed_block_power_iteration, \
    check_overflow, mat_to_sym, sym_eigen_to_mat_singular

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# INFO or WARNING
logger.setLevel(logging.INFO)

class BlockIterSVD(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, max_spectral_ratio=0.9, eps=1e-10, nbits=0, \
                 n_parties=1, random_seed=0):
        """

        Parameters
        ----------
        k : positive integer, optional
            The rank of the truncated SVD
        max_spectral_ratio : float, optional
            Among the top k singular values, the largest ratio of
            sigma_i/sigma_i+1 . Used to derive the number of iterations
            needed until convergence.
            Solve for :math:`n_{iter}` :math:`2(max_{ratio})^n_{iter} = eps`
        eps : float, optional
            A desired UB on :math:`max_i ||\hat V_{:i} - V_{:i}||_2`,
            this will be used along with `max_spectral_ratio` to compute the
            number of iterations to be done.
        random_seed : integer, optional
            Random seed for the power iteration initialization
        nbits : positive integer, optional
            Number of bits per integer when representing floats as integers
            for the purpose of normalized secure sum
        n_parites : positive integer, optional
            Used to simulate the SVD being computed among multiple parties
            with conversion to integers in between rounds
        """
        self.n_iter = int(ceil(log(eps / 2.0) / log(max_spectral_ratio)))
        self.random_seed = random_seed
        self.k = k
        self.nbits = nbits
        self.n_parties = n_parties

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        I = lambda i: np.arange(i, X.shape[0], self.n_parties)
        Xs = [X[I(i), :] for i in range(self.n_parties)]
        Ss = [mat_to_sym(Xi, 'mult') for Xi in Xs]
        logger.info('len(Ss)={}'.format(len(Ss)))
        nbits_max = check_overflow(Ss, 0)
        if nbits_max < self.nbits:
            logger.warning('Initialized with nbits={} but nbiis_max={}'
                           'based on X. Lowering.'.format(self.nbits,
                                                          nbits_max))
        logger.info('Setting nbits to nbits_max={}.'.format(nbits_max))
        self.nbits = nbits_max
        rtv = private_distributed_block_power_iteration(Ss, k=self.k,
                T=self.n_iter, nbits=self.nbits, random_seed=self.random_seed)
        V_est, s_est = rtv['V'], rtv['s']
        V_est, s_est = sym_eigen_to_mat_singular(V_est, s_est, 'mult')
        self.V = V_est
        return np.dot(X, self.V)

    def transform(self, X, y=None):
        return np.dot(X, self.V)

    def inverse_transform(self, X):
        # XVV^T is a rank k representation of X
        return np.dot(X, self.V.T)