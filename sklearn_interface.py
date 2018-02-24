from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
import numpy as np
from math import ceil, log, exp

from blockpower_svd import private_distributed_block_power_iteration, \
    check_overflow, mat_to_sym, sym_eigen_to_mat_singular, \
    block_power_iteration, private_top_k_eigenvectors

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# INFO or WARNING
logger.setLevel(logging.INFO)


class BlockIterSVD(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, max_spectral_ratio=0.9, eps=1e-10, nbits=20, \
                 n_parties=1, random_seed=0, eps_diff_priv=None):
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
        eps_dif_priv : positive float or None, optional
            The epsilon to use for differentially private SVD
        """
        self.n_iter = int(ceil(log(eps / 2.0) / log(max_spectral_ratio)))
        logger.info('n_iter={}'.format(self.n_iter))
        self.random_seed = random_seed
        self.k = k
        self.nbits = nbits
        self.n_parties = n_parties
        self.eps_diff_priv=None

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        check_array(X, accept_sparse=False)
        n, d = X.shape
        logger.info('Transforming X={}x{}'.format(n,d))
        r = min([n, d])
        assert self.k <= r, 'asking for k ={} > min(m,n)={}'.format(self.k, r)

        I = lambda i: np.arange(i, X.shape[0], self.n_parties)
        Xs = [X[I(i), :] for i in range(self.n_parties)]
        Ss = [mat_to_sym(Xi, 'mult') for Xi in Xs]
        logger.info('len(Ss)={}'.format(len(Ss)))

        if self.nbits == 0:
            nbits_max = check_overflow(Ss, 0)
        else:
            nbits_max = self.nbits
        logger.info('Setting nbits to nbits_max={}.'.format(nbits_max))
        self.nbits_ = nbits_max

        if self.eps_diff_priv is None:
            if self.n_parties>1:
                logger.info('Using private block power iteration')
                rtv = private_distributed_block_power_iteration(Ss, k=self.k,
                    T=self.n_iter, nbits=self.nbits_,
                    random_seed=self.random_seed, perform_check_overflow=False)
            elif self.n_parties==1:
                logger.info('n_parties==1, using centralized block power iteration')
                rtv = block_power_iteration(Ss[0], k=self.k, T=self.n_iter,
                                            random_seed=self.random_seed)
        else:
            # pick generous parameters for delta and coherence
            delta = 0.1
            coh_ub = 1.0
            e_dp = float(self.eps_diff_priv)
            logger.info('eps={:.1e} using private low rank'.format(e_dp))
            rtv = private_top_k_eigenvectors(Ss[0],k=self.k, T=self.n_iter,
             random_seed=self.random_seed, eps=e_dp, delta=delta, coh_ub=coh_ub)
        V_est, s_est = rtv['V'], rtv['s']
        V_est, s_est = sym_eigen_to_mat_singular(V_est, s_est, 'mult')
        self.V_ = V_est
        return np.dot(X, self.V_)

    def transform(self, X, y=None):
        check_array(X, accept_sparse=False)
        check_is_fitted(self, ['V_'])
        return np.dot(X, self.V_)

    def inverse_transform(self, X):
        """XVV^T is a rank k representation of X"""
        check_array(X, accept_sparse=False)
        check_is_fitted(self, ['V_'])
        return np.dot(X, self.V_.T)

    def recons_err(self, X):
        Xh = self.transform(X)
        Xh = self.inverse_transform(Xh)
        return np.linalg.norm(Xh-X,'fro')


class PrincipalComponentRegression(BlockIterSVD):
    def fit(self, X, y):
        check_X_y(X, y, accept_sparse=False)

        self.mu_x_ = np.mean(X, 0)
        X = X - self.mu_x_
        self.mu_y_ = np.mean(y)
        y = y - self.mu_y_
        W = super(PrincipalComponentRegression, self).fit_transform(X)
        self.coef_, self._residues, self.rank_, self.singular_ \
            = np.linalg.lstsq(W, y)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['V_', 'mu_x_', 'mu_y_', 'coef_'])

        X = X - np.mean(X, 0)
        return super(PrincipalComponentRegression, self).transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        check_is_fitted(self, ['V_', 'mu_x_', 'mu_y_', 'coef_'])

        X = self.transform(X)
        return np.dot(X, self.coef_) + self.mu_y_

    def rmse(self, Xtest, ytest):
        check_is_fitted(self, ['V_', 'mu_x_', 'mu_y_', 'coef_'])
        check_X_y(Xtest, ytest, accept_sparse=False)

        ypred = self.predict(Xtest)
        return np.sqrt(np.mean((ytest - ypred) ** 2))














