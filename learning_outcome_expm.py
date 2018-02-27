import luigi
import itertools
import numpy as np
import pickle
from sklearn.utils import resample
from math import ceil
import copy

from sklearn_interface import BlockIterSVD, PrincipalComponentRegression

from experiment_utils.luigi_interface.MTask import \
    AutoLocalOutputMixin, LoadInputDictMixin
from matlabinterface import datasets
from topic_model_evaluation.goodness_of_fit import change_of_dictionary
from matrixops import transform

base_path = '/mnt/WD8TB/experiments/svd/'

import sys

if sys.version_info[0] == 3:
    from functools import reduce

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# INFO or WARNING
logger.setLevel(logging.INFO)

# pick the first n_terms_lra for the LRA problem
n_terms_lra = 1000
min_doc_length = 1
# we do this because the d*d symmetric matrices take too much memory
# and scipy's sparse matrix algebra support still isn't worth building
# around.

class EvalAll(luigi.WrapperTask):
    def requires(self):
        Ks = [21]#[7, 14, 21, 42]  # k for k-truncated SVD
        Ms = [3, 9, 27, 81]  # number of parties, each samples 1/M of the data
        Nbits = [10, 20, 60]
        Epsilons_DP = [0.1, 10.0]#, 1.0]
        problem_settings = [
            {
                'problem': 'pcr',  # principal component regression
                'dataset_name': 'MillionSongs'
            },
            {
                'problem': 'lra',  # low rank approximation
                'dataset_name': 'Enron'
            },
            {
                'problem': 'pcr',  # principal component regression
                'dataset_name': 'OnlineNewsPopularity'
            },
            {
                'problem': 'lra',  # low rank approximation
                'dataset_name': '20NG'
            }
        ]
        trials = range(5)  # random seed for each party's local data sample
        reqs = []
        for k, M, nbits, problem_setting, trial in itertools.product(Ks, Ms,
                                            Nbits, problem_settings, trials):
            reqs.append(EvalLocalVsGlobal(
                k=k,
                M=M,
                nbits=nbits,
                problem_setting=problem_setting,
                trial=trial
            ))
        for k, M, eps_dp, problem_setting, trial in itertools.product(Ks, Ms,
                                 Epsilons_DP,  problem_settings, trials):
            problem_setting['eps'] = eps_dp
            reqs.append(EvalLocalVsGlobal(
                k=k,
                M=M,
                nbits=-1,
                problem_setting=problem_setting,
                trial=trial
            ))
        yield reqs


class EvalLocalVsGlobal(AutoLocalOutputMixin(base_path=base_path + 'eval/'),
                        LoadInputDictMixin,
                        luigi.Task):
    k = luigi.IntParameter()
    M = luigi.IntParameter()
    nbits = luigi.IntParameter()
    problem_setting = luigi.DictParameter()
    trial = luigi.IntParameter()

    def requires(self):
        req = {
            'global': ComputeGlobalModel(k=self.k,
                                         M=self.M,
                                         nbits=self.nbits,
                                         problem_setting=self.problem_setting,
                                         trial=self.trial
                                         )
        }
        return req

    def run(self):
        # load global model
        inp = self.load_input_dict()
        Mod_global = inp['global']

        logger.info('Loading data')
        # get test portion for each problem
        problem = self.problem_setting['problem']
        dn = self.problem_setting['dataset_name']
        if problem == 'pcr':
            ds = datasets.load_regression_dataset(dn)
            Xte, yte = ds['Xte'], ds['yte']
        elif problem == 'lra':
            # get the same top-n words as we did for training the global model
            ds = datasets.load_dataset('Enron', train=False,
                                       min_words_per_doc=min_doc_length,
                                       dict_size=n_terms_lra)
            Xte = ds['X']

        # get each party's local data
        each_party_data, _ = load_data(problem, dn, self.M, self.trial)
        assert len(each_party_data) == self.M


        logger.info('Evaluating global model {} {}'.format(problem, dn))
        # fit a local model and compare evaluation to global model
        if problem == 'lra':
            gs = Mod_global.recons_err(Xte)
        elif problem == 'pcr':
            gs = Mod_global.rmse(Xte, yte)

        base_eval = {
            'k': self.k, 'M': self.M, 'party_pct': 1.0 / self.M,
            'global_score':gs, 'trial': self.trial, 'bits_precision': self.nbits
        }
        base_eval.update(self.problem_setting)

        evals = []
        for m in range(self.M):
            logger.info('Evaluating model {} for {} {}'.format(m, problem, dn))
            self.set_progress_percentage(100.0*float(m)/self.M)
            self.set_status_message('local model {}'.format(m))
            evc = copy.deepcopy(base_eval)
            if problem == 'lra':
                X = each_party_data[m]
                Mod = BlockIterSVD(k=self.k, n_parties=1)
                Mod = Mod.fit(X)
                ls = Mod.recons_err(Xte)
            elif problem == 'pcr':
                X, y = each_party_data[m][0], each_party_data[m][1]
                Mod = PrincipalComponentRegression(k=self.k, n_parties=1)
                Mod = Mod.fit(X, y)
                ls = Mod.rmse(Xte, yte)
            evc['local_score'] = ls
            evc['pct_change'] = (evc['global_score']-evc['local_score']) / \
                                evc['local_score']
            evals.append(evc)

        with self.output().open('w') as f:
            pickle.dump(evals, f, 2)


def gen_samples(samp, M, trial):
    if type(samp) is not list:
        raise ValueError('gen_samples expects samp to be a list')
    for (i, it) in enumerate(samp):
        if len(it.shape) == 1:
            samp[i] = samp[i][:, np.newaxis]  # so vstack works correctly
    each_party_data = [None] * M
    n_part = int(ceil(samp[0].shape[0] * 1.0 / M))
    for m in range(M):
        sample = resample(*samp, replace=False, n_samples=n_part,
                          random_state=trial + 1000 * m)
        each_party_data[m] = sample

    if type(each_party_data[0]) == list:
        stacked = [None] * len(each_party_data[0])
        for i in range(len(stacked)):
            stacked[i] = reduce(lambda x, y: np.vstack((x, y)),
                                [arr[i] for arr in each_party_data])
    else:
        stacked = reduce(lambda x, y: np.vstack((x, y)), each_party_data)

    return each_party_data, stacked


def load_data(problem, dn, M, trial):
    if problem == 'pcr':
        ds = datasets.load_regression_dataset(dn)
        samp = [ds['Xtr'], ds['ytr']]
    elif problem == 'lra':
        ds = datasets.load_dataset(dn,
                                   min_words_per_doc=min_doc_length,
                                   dict_size=n_terms_lra)
        samp = [ds['X'].toarray()]

    each_party_data, stacked = gen_samples(samp, M, trial)
    return each_party_data, stacked


class ComputeGlobalModel(
    AutoLocalOutputMixin(base_path=base_path),
    LoadInputDictMixin,
    luigi.Task):
    k = luigi.IntParameter()
    M = luigi.IntParameter()
    nbits = luigi.IntParameter()
    problem_setting = luigi.DictParameter()
    trial = luigi.IntParameter()

    def run(self):
        problem = self.problem_setting['problem']
        dn = self.problem_setting['dataset_name']

        _, stacked = load_data(problem, dn, self.M, self.trial)


        # normal experiments with no eps but with nbits
        if 'eps' not in self.problem_setting:
            logger.info('Fitting global model using PD-SVD')
            if problem == 'lra':
                X = stacked
                Mod = BlockIterSVD(k=self.k, n_parties=self.M, nbits=self.nbits)
                Mod = Mod.fit(X)
            elif problem == 'pcr':
                X, y = stacked[0], stacked[1]
                Mod = PrincipalComponentRegression(k=self.k, n_parties=self.M,
                                                   nbits=self.nbits)
                Mod = Mod.fit(X, y)
        else:
        # experiments with eps
            logger.info('Fitting global model using Diff Priv SVD')
            eps = self.problem_setting['eps']
            if problem == 'lra':
                X = stacked
                Mod = BlockIterSVD(k=self.k, eps_diff_priv=eps)
                Mod = Mod.fit(X)
            elif problem == 'pcr':
                X, y = stacked[0], stacked[1]
                Mod = PrincipalComponentRegression(k=self.k, eps_diff_priv=eps)
                Mod = Mod.fit(X, y)

        with self.output().open('w') as f:
            pickle.dump(Mod, f, 2)
