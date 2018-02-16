import luigi
import itertools
import numpy as np
from sklearn.utils import resample
from math import ceil

from sklearn_interface import BlockIterSVD

from experiment_utils.luigi_interface.MTask import \
    AutoLocalOutputMixin, LoadInputDictMixin
from matlabinterface import datasets

base_path = '/mnt/WD8TB/experiments/svd/'

import sys

if sys.version_info[0] == 3:
    from functools import reduce


class EvalAll(luigi.WrapperTask):
    def requires(self):
        Ks = [7, 14, 21]
        Ms = [3, 9, 27, 81]
        problem_settings = [
            {
                'problem': 'pcr',  # principal component regression
                'dataset_name': 'MillionSongs'
            },
            {
                'problem': 'lra',  # low rank approximation
                'dataset_name': 'Enron'
            }
        ]
        trials = range(5)
        reqs = []
        for k, M, problem_setting, trial in itertools.product(Ks, Ms,
                                                              problem_settings,
                                                              trials):
            for local_party in range(M):
                reqs.append(EvalLocalVsGlobal(
                    k=k,
                    M=M,
                    local_party=local_party,
                    problem_setting=problem_setting,
                    trial=trial
                ))
        yield reqs


class EvalLocalVsGlobal(AutoLocalOutputMixin(base_path=base_path + 'eval/'),
                        LoadInputDictMixin,
                        luigi.Task):
    k = luigi.IntParameter()
    M = luigi.IntParameter()
    local_party = luigi.IntParameter()
    problem_setting = luigi.DictParameter()
    trial = luigi.IntParameter()

    def requires(self):
        d = {'base': [], 'part': []}
        d['merged'] = \
            ComputeGlobalModel(data_name=self.data_name,
                               merge_alg=self.merge_alg,
                               num_splits=self.num_splits,
                               part_trial_num=self.part_trial_num,
                               train_test_trial_num=self.train_test_trial_num,
                               k_value=self.k_value
                               )
        d['base'] = \
            ComputeLocalModel(
                data_name=self.data_name,
                num_splits=self.num_splits,
                part_trial_num=self.part_trial_num,
                train_test_trial_num=self.train_test_trial_num,
                k_value=self.k_value
            )

        d['part'] = \
            GeneratePartition(
                data_name=self.data_name,
                num_splits=self.num_splits,
                part_trial_num=self.part_trial_num,
                train_test_trial_num=self.train_test_trial_num
            )

        return d


def gen_samples(samp, M, trial):
    if type(samp) is not list:
        raise ValueError('gen_samples expects samp to be a list')
    for (i, it) in enumerate(samp):
        if len(it.shape) == 1:
            samp[i] = samp[i][:, np.newaxis]  # so vstack works correctly
    combined_data = [None] * M
    n_part = int(ceil(samp[0].shape[0] * 1.0 / M))
    for m in range(M):
        sample = resample(*samp, replace=False, n_samples=n_part,
                          random_state=trial + 1000 * m)
        combined_data[m] = sample

    if type(combined_data[0]) == list:
        stacked = [None] * len(combined_data[0])
        for i in range(len(stacked)):
            stacked[i] = reduce(lambda x, y: np.vstack((x, y)),
                                [arr[i] for arr in combined_data])
    else:
        stacked = reduce(lambda x, y: np.vstack((x, y)), combined_data)

    return combined_data, stacked


class ComputeGlobalModel(
    AutoLocalOutputMixin(base_path=base_path),
    LoadInputDictMixin,
    luigi.Task):
    k = luigi.IntParameter()
    M = luigi.IntParameter()
    problem_setting = luigi.DictParameter()
    trial = luigi.IntParameter()

    def run(self):
        problem = self.problem_setting['problem']
        dn = self.problem_setting['dataset_name']
        if problem == 'pcr':
            ds = datasets.load_regression_dataset(dn)
            samp = [ds['Xtr'], ds['ytr']]
        elif problem == 'lra':
            ds = datasets.load_dataset(dn)
            samp = [ds['X'].toarray()]

        combined_data, stacked = gen_samples(samp, self.M, self.trial)

        if problem == 'lra':
            X = stacked
            Mod = BlockIterSVD(k=self.k, n_parties=self.M)
        elif problem== 'pcr':
            X, y = stacked[0], stacked[1]
