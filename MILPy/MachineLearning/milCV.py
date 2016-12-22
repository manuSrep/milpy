#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Model selection module for MIL datasets. API is tried to keep consistent with
sklearn. Several parts of the code are also copied directly from sklearn and are
therefore released under their license.

:author: Manuel Tuschen
:date: 04.07.2016
:license: BSD
"""

import time
import warnings
from joblib import logger
from copy import deepcopy
from collections import Mapping, namedtuple, Sized, defaultdict, Sequence
from itertools import combinations, product
from functools import partial, reduce
import operator
import numbers
from scipy.stats import rankdata
from numpy.ma import MaskedArray
from scipy import sparse
from scipy.misc import comb
import numpy as np
import sklearn as skl
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid, ParameterSampler
from joblib import Parallel, delayed

__all__ = ["KFold", "train_test_split", "cross_val_predict",
    "cross_val_predictions", "cross_val_score", "validation_curve",
    "GridSearchCV"]


################################################################################
#                                                                              #
#                       Splitter Classes                                       #
#                                                                              #
################################################################################

class BaseCrossValidator:
    """
    Base class for all cross-validators
    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def split(self, data):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        data : milData
            The dataset to split

        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        indices = np.arange(data.N_B)
        for test_index in self._iter_test_masks(data):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, data):
        """
        Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices(data)
        """
        for test_index in self._iter_test_indices(data):
            test_mask = np.zeros(data.N_B, dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, data):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    def get_n_splits(self, data):
        """Returns the number of splitting iterations in the cross-validator"""


class _BaseKFold(BaseCrossValidator):
    """
    Base class for KFold, GroupKFold, and StratifiedKFold
    """

    def __init__(self, n_splits, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                'The number of folds must be of Integral type. ' '%s of type %s was passed.' % (
                n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one" " train/test split by setting n_splits=2 or more," " got n_splits={0}.".format(
                    n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError(
                "shuffle must be True or False;"" got {0}".format(shuffle))

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_samples = data.N_B
        if self.n_splits > n_samples:
            raise ValueError((
            "Cannot have number of splits n_splits={0} greater" " than the number of samples: {1}.").format(
                self.n_splits, n_samples))

        for train, test in super(_BaseKFold, self).split(data):
            yield train, test

    def get_n_splits(self, data):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class KFold(_BaseKFold):
    """K-Folds cross-validator
    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.
    random_state : None, int or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(KFold, self).__init__(n_splits, shuffle, random_state)

    def _iter_test_indices(self, data):
        n_samples = data.N_B
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class LeaveOneOut(BaseCrossValidator):
    """
    Leave-One-Out cross-validator
    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples form the training set.
    """

    def _iter_test_indices(self, data):
        return range(data.N_B)

    def get_n_splits(self, data):
        """
        Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        data : milData
            The dataset to split

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return data.N_B


class LeavePOut(BaseCrossValidator):
    """
    Leave-P-Out cross-validator
    Provides train/test indices to split data in train/test sets. This results
    in testing on all distinct samples of size p, while the remaining n - p
    samples form the training set in each iteration.
    Note: ``LeavePOut(p)`` is NOT equivalent to
    ``KFold(n_splits=n_samples // p)`` which creates non-overlapping test sets.

    Parameters
    ----------
    p : int
        Size of the test sets.
    """

    def __init__(self, p):
        self.p = p

    def _iter_test_indices(self, data):
        for combination in combinations(range(data.N_B), self.p):
            yield np.array(combination)

    def get_n_splits(self, data):
        """
        Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        data : milData
            Training data.
        """
        return int(comb(data.N_B, self.p, exact=True))


class BaseShuffleSplit():
    """Base class for ShuffleSplit and StratifiedShuffleSplit"""

    def __init__(self, n_splits=10, test_size=0.1, train_size=None,
            random_state=None):
        _validate_shuffle_split_init(test_size, train_size)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, data):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        data : milData
            The dataset to split

        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        for train, test in self._iter_indices(data):
            yield train, test

    def _iter_indices(self, X, y=None, groups=None):
        """Generate (train, test) indices"""

    def get_n_splits(self, data):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class ShuffleSplit(BaseShuffleSplit):
    """
    Random permutation cross-validator
    Yields indices to split data into training and test sets.
    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    n_splits : int (default 10)
        Number of re-shuffling & splitting iterations.
    test_size : float, int, or None, default 0.1
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
    """

    def _iter_indices(self, data):
        n_samples = data.N_B
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size,
            self.train_size)
        np.random.seed(self.random_state)
        for i in range(self.n_splits):
            # random partition
            permutation = np.random.permutation(n_samples)
            ind_test = permutation[:n_test]
            ind_train = permutation[n_test:(n_test + n_train)]
            yield ind_train, ind_test


class PredefinedSplit(BaseCrossValidator):
    """Predefined split cross-validator
    Splits the data into training/test set folds according to a predefined
    scheme. Each sample can be assigned to at most one test set fold, as
    specified by the user through the ``test_fold`` parameter.
    """

    def __init__(self, test_fold):
        self.test_fold = np.array(test_fold, dtype=np.int)
        self.unique_folds = np.unique(self.test_fold)
        self.unique_folds = self.unique_folds[self.unique_folds != -1]

    def split(self, data):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        ind = np.arange(len(self.test_fold))
        for test_index in self._iter_test_masks():
            train_index = ind[np.logical_not(test_index)]
            test_index = ind[test_index]
            yield train_index, test_index

    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets."""
        for f in self.unique_folds:
            test_index = np.where(self.test_fold == f)[0]
            test_mask = np.zeros(len(self.test_fold), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def get_n_splits(self, data):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.unique_folds)


def _validate_shuffle_split_init(test_size, train_size):
    """Validation helper to check the test_size and train_size at init
    NOTE This does not take into account the number of samples which is known
    only at split
    """
    if test_size is None and train_size is None:
        raise ValueError('test_size and train_size can not both be None')

    if test_size is not None:
        if np.asarray(test_size).dtype.kind == 'f':
            if test_size >= 1.:
                raise ValueError('test_size=%f should be smaller '
                                 'than 1.0 or be an integer' % test_size)
        elif np.asarray(test_size).dtype.kind != 'i':
            # int values are checked during split based on the input
            raise ValueError("Invalid value for test_size: %r" % test_size)

    if train_size is not None:
        if np.asarray(train_size).dtype.kind == 'f':
            if train_size >= 1.:
                raise ValueError("train_size=%f should be smaller "
                                 "than 1.0 or be an integer" % train_size)
            elif (np.asarray(test_size).dtype.kind == 'f' and (
                train_size + test_size) > 1.):
                raise ValueError('The sum of test_size and train_size = %f, '
                                 'should be smaller than 1.0. Reduce '
                                 'test_size and/or train_size.' % (
                                 train_size + test_size))
        elif np.asarray(train_size).dtype.kind != 'i':
            # int values are checked during split based on the input
            raise ValueError("Invalid value for train_size: %r" % train_size)


def _validate_shuffle_split(n_samples, test_size, train_size):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    """
    if (test_size is not None and np.asarray(
            test_size).dtype.kind == 'i' and test_size >= n_samples):
        raise ValueError('test_size=%d should be smaller than the number of '
                         'samples %d' % (test_size, n_samples))

    if (train_size is not None and np.asarray(
            train_size).dtype.kind == 'i' and train_size >= n_samples):
        raise ValueError("train_size=%d should be smaller than the number of"
                         " samples %d" % (train_size, n_samples))

    if np.asarray(test_size).dtype.kind == 'f':
        n_test = np.ceil(test_size * n_samples)
    elif np.asarray(test_size).dtype.kind == 'i':
        n_test = float(test_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif np.asarray(train_size).dtype.kind == 'f':
        n_train = np.floor(train_size * n_samples)
    else:
        n_train = float(train_size)

    if test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n_samples))

    return int(n_train), int(n_test)


################################################################################
#                                                                              #
#                       Splitter Functions                                     #
#                                                                              #
################################################################################

def train_test_split(data, test_size=None, train_size=None, random_state=None):
    """
    Split a mil dataset into test and training set.

    Parameters
    ----------
    data : milData
        The dataset to split
    test_size : float, int, or None (default is None), optional
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is automatically
        set to the complement of the train size. If train size is also None,
        test size is set to 0.25.
    train_size : float, int, or None (default is None), optional
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the train split. If int, represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.
    random_state : int, optional
        Pseudo-random number generator state used for random sampling.

    Returns
    --------
    data_train : milData
        The training set.
    data_test : milData
        The test set.
    """
    keys_train, keys_test, _, _ = skl.model_selection.train_test_split(
        data.keys, data.keys, test_size=test_size, train_size=train_size,
        random_state=random_state)

    data_train = deepcopy(data)
    for key in keys_test:
        data_train.del_B(key)

    data_test = deepcopy(data)
    for key in keys_train:
        data_test.del_B(key)

    return data_train, data_test


################################################################################
#                                                                              #
#                       Hyper-parameter optimizers                             #
#                                                                              #
#################################################################################
class BaseSearchCV(BaseEstimator):
    """
    Base class for hyper parameter search with cross-validation.
    """

    def __init__(self, estimator, scoring=None, fit_params=None, n_jobs=1,
            iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
            error_score='raise', return_train_score=True):

        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.fit_params = fit_params if fit_params is not None else {}
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def score(self, data):
        """Returns the score on the given data, if the estimator has been refit.
        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        Returns
        -------
        score : float
        """
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s" % self.best_estimator_)
        return self.scorer_(self.best_estimator_, data)

    def _fit(self, data, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = self.cv
        self.scorer_ = self.scoring

        n_splits = cv.get_n_splits(data)
        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                n_candidates * n_splits))

        base_estimator = _clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        cv_iter = list(cv.split(data))
        out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch)(
            delayed(_fit_and_score)(_clone(base_estimator), data, self.scorer_,
                train, test, self.verbose, parameters,
                fit_params=self.fit_params,
                return_train_score=self.return_train_score,
                return_n_test_samples=True, return_times=True,
                return_parameters=True, error_score=self.error_score) for
                parameters in parameter_iterable for train, test in cv_iter)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores_z, train_scores_y, test_scores_z, test_scores_y,
            test_sample_counts, fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores_z, test_scores_y, test_sample_counts, fit_time,
            score_time, parameters) = zip(*out)

        candidate_params = parameters[::n_splits]
        n_candidates = len(candidate_params)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                n_splits)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s" % (split_i, key_name)] = array[:,
                    split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average((array - array_means[:, np.newaxis]) ** 2, axis=1,
                    weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
            dtype=np.int)

        _store('test_score', test_scores_z, splits=True, rank=True,
            weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store('train_score', train_scores_z, splits=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(MaskedArray, np.empty(n_candidates, ), mask=True,
                dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = _clone(base_estimator).set_params(
                **best_parameters)
            best_estimator.fit(data, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self

    @property
    def best_params_(self):
        return self.cv_results_['params'][self.best_index_]

    @property
    def best_score_(self):
        return self.cv_results_['mean_test_score'][self.best_index_]


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.
    Important members are fit, predict.
    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.
    Read more in the :ref:`User Guide <grid_search>`.
    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.
    fit_params : dict, optional
        Parameters to pass to the fit method.
    n_jobs : int, default=1
        Number of jobs to run in parallel.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
        For instance the below given table
        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_....|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |        0.8      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |        0.7      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+
        will be represented by a ``cv_results_`` dict of::
            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
            'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
            'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }
        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.
        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.
    best_score_ : float
        Score of best_estimator on the left out data.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
            n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
            pre_dispatch='2*n_jobs', error_score='raise',
            return_train_score=True):
        super().__init__(estimator=estimator, scoring=scoring,
            fit_params=fit_params, n_jobs=n_jobs, iid=iid, refit=refit, cv=cv,
            verbose=verbose, pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid

    def fit(self, data):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """
        return self._fit(data, ParameterGrid(self.param_grid))


class RandomizedSearchCV(BaseSearchCV):
    """
    Randomized search on hyper parameters.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.
    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.
    If all parameters are presented as a list, sampling without replacement is
    performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.
    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.
    fit_params : dict, optional
        Parameters to pass to the fit method.
    n_jobs : int, default=1
        Number of jobs to run in parallel.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
        For instance the below given table
        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        will be represented by a ``cv_results_`` dict of::
            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }
        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.
        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.
    best_score_ : float
        Score of best_estimator on the left out data.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    """

    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
            fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
            pre_dispatch='2*n_jobs', random_state=None, error_score='raise',
            return_train_score=True):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(estimator=estimator, scoring=scoring,
            fit_params=fit_params, n_jobs=n_jobs, iid=iid, refit=refit, cv=cv,
            verbose=verbose, pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def fit(self, data):
        """
        Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        data : milData
            Data to fit.
        """
        sampled_params = ParameterSampler(self.param_distributions, self.n_iter,
            random_state=self.random_state)
        return self._fit(data, sampled_params)


################################################################################
#                                                                              #
#                       Model validation                                       #
#                                                                              #
################################################################################

def cross_val_score(estimator, data, scoring, cv, fit_params=None, n_jobs=1,
        pre_dispatch='2*n_jobs', verbose=0):
    """
    Evaluate a score by cross-validation

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    data : milData object
        The data to fit.
    scoring : callable
        A  scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    cv : cross-validation generator
        Determines the cross-validation splitting strategy.
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    Returns
    -------
    scores_z : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    scores_y
    """

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(_clone(estimator), data, scoring, train, test,
            None, fit_params) for train, test in cv.split(data))

    scores_z = np.array(scores)[:, 0]
    scores_y = np.array(scores)[:, 1]
    if np.any(scores_y is None):
        scores_y = None
    return scores_z, scores_y


def cross_val_predict(estimator, data, cv, fit_params=None, n_jobs=1,
        pre_dispatch='2*n_jobs', method='predict'):
    """
    Generate cross-validated estimates for each input data point

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    data : milData object
        The data to fit.
    cv : cross-validation generator
        Determines the cross-validation splitting strategy.
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator.

    Returns
    -------
    predictions_z : ndarray
    predictions_y : ndarray
    """
    # Ensure the estimator has implemented the passed decision function
    if not callable(getattr(estimator, method)):
        raise AttributeError('{} not implemented in estimator'.format(method))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(
        delayed(_fit_and_predict)(_clone(estimator), data, train, test, 0,
            fit_params, method) for train, test in cv.split(data))

    # Concatenate the predictions
    predictions_z = [pred_block_z for pred_block_z, _, _ in prediction_blocks]
    predictions_y = [pred_block_y for _, pred_block_y, _ in prediction_blocks]

    test_keys = np.concatenate([key for _, _, key in prediction_blocks])
    test_indices = [list(test_keys).index(key) for key in data.keys]

    predictions_z = np.concatenate(predictions_z)
    if not None in predictions_y:
        predictions_y = np.concatenate(predictions_y)
        return predictions_z[test_indices], predictions_y[test_indices]
    else:
        return predictions_z[test_indices], None


def cross_val_predictions(estimator, data, cv, fit_params=None, n_jobs=1,
        pre_dispatch='2*n_jobs', method='predict'):
    """
    Generate cross-validated estimates for each CV test split

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    data : milData object
        The data to fit.
    cv : cross-validation generator
        Determines the cross-validation splitting strategy.
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator.

    Returns
    -------
    predictions_z : ndarray
    predictions_y : ndarray
    """
    # Ensure the estimator has implemented the passed decision function
    if not callable(getattr(estimator, method)):
        raise AttributeError('{} not implemented in estimator'.format(method))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(
        delayed(_fit_and_predict)(deepcopy(estimator), data, train, test, 0,
            fit_params, method) for train, test in cv.split(data))

    # Concatenate the predictions
    predictions_z = [pred_block_z for pred_block_z, _, _ in prediction_blocks]
    predictions_y = [pred_block_y for _, pred_block_y, _ in prediction_blocks]
    if np.any(predictions_y is None):
        predictions_y = None
    test_keys = [keys for _, _, keys in prediction_blocks]

    return predictions_z, predictions_y, test_keys


def validation_curve(estimator, data, param_name, param_range, cv, scoring,
        n_jobs=1, pre_dispatch="all", verbose=0, fit_params=None):
    """
    Validation curve.
    Determine training and test scores for varying parameter values.
    Compute scores for an estimator with different values of a specified
    parameter. This is similar to grid search with one parameter. However, this
    will also compute training scores and is merely a utility for plotting the
    results.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    param_name : string
        Name of the parameter that will be varied.
    param_range : array-like, shape (n_values,)
        The values of the parameter that will be evaluated.
    cv : cross-validation generator
        Determines the cross-validation splitting strategy.
    scoring : callable
        A  scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    pre_dispatch : integer or string, optional
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The string can
        be an expression like '2*n_jobs'.

    Returns
    -------
    train_scores_z : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.
    test_scores_z : array, shape (n_ticks, n_cv_folds)
        Scores on test set.
    train_scores_y : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.
    test_scores_y : array, shape (n_ticks, n_cv_folds)
        Scores on test set.
    """

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
    out = parallel(
        delayed(_fit_and_score)(estimator, data, scoring, train, test, verbose,
            {param_name: v}, fit_params, return_train_score=True) for
        train, test in cv.split(data) for v in param_range)
    out = np.asarray(out)
    n_params = len(param_range)
    n_cv_folds = out.shape[0] // n_params

    out = out.reshape(n_cv_folds, n_params, 2, 2).transpose((2, 1, 0, 3))
    train_scores_z = out[0, :, :, 0].astype(np.float)
    test_scores_z = out[1, :, :, 0].astype(np.float)
    if not np.any(out[:, :, :, 1] is None):
        train_scores_y = out[0, :, :, 1].astype(np.float)
        test_scores_y = out[1, :, :, 1].astype(np.float)
    else:
        train_scores_y = test_scores_y = None

    return train_scores_z, test_scores_z, train_scores_y, test_scores_y


def _fit_and_score(estimator, data, scorer, train, test, verbose, parameters,
        fit_params, return_train_score=False, return_parameters=False,
        return_n_test_samples=False, return_times=False, error_score='raise'):
    """
    Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape at least 2D
        The data to fit.
    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    train : array-like, shape (n_train_samples,)
        Keys of training samples.
    test : array-like, shape (n_test_samples,)
        Keys of test samples.
    parameters : dictionary or None
        Parameters to be set on the estimator.
    fit_params : dictionary or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    Returns
    -------
    train_score_z : float, optional
        Score on training set, returned only if `return_train_score` is `True`.
    train_score_y : float, optional
        Score on training set, returned only if `return_train_score` is `True`.
    test_score_z : float
        Score on test set.
    test_score_y : float
        Score on test set.
    """
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (
            ', '.join('%s=%s' % (k, v) for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

        # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, v) for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    data_train = _safe_split(data, train)
    data_test = _safe_split(data, test)

    try:
        estimator.fit(data_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e))
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        test_score_z, test_score_y = scorer(estimator, data_test)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_score_z, train_score_y = scorer(estimator, data_train)

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score_z, train_score_y, test_score_z,
        test_score_y] if return_train_score else [test_score_z, test_score_y]

    if return_n_test_samples:
        ret.append(data_test.N_B)
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret


def _fit_and_predict(estimator, data, train, test, verbose, fit_params, method):
    """
    Fit estimator and predict values for a given dataset split.
    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    milData : milData object
        The data to fit.
    train : array-like, shape (n_train_samples,)
        Keys of training samples.
    test : array-like, shape (n_test_samples,)
        Keys of test samples.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    method : string
        Invokes the passed method name of the passed estimator.

    Returns
    -------
    predictions : sequence
        Result of calling 'estimator.method'
    test : array-like
        This is the value of the test parameter
    """
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, v) for k, v in fit_params.items()])

    data_train = _safe_split(data, train)
    data_test = _safe_split(data, test)

    estimator.fit(data_train, **fit_params)
    func = getattr(estimator, method)
    predictions_z, predictions_y = func(data_test)
    return predictions_z, predictions_y, data_test.keys


def _safe_split(data, key_index):
    """
    Create milData subset.
    """
    new = deepcopy(data)

    for k, key in enumerate(data.keys):
        if k not in key_index:
            new.del_B(key)

    return new


def _clone(estimator, safe=True):
    """
    Constructs a new estimator with the same parameters.
    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fit on any data.

    Parameters
    ----------
    estimator: estimator object, or list, tuple or set of objects
        The estimator or group of estimators to be cloned
    safe: boolean, optional
        If safe is false, clone will fall back to a deepcopy on objects
        that are not estimators.
    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([_clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params'):
        if not safe:
            return deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a scikit-learn estimator "
                            "as it does not implement a 'get_params' methods." % (
                            repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = _clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is param2:
            # this should always happen
            continue
        if isinstance(param1, np.ndarray):
            # For most ndarrays, we do not test for complete equality
            if not isinstance(param2, type(param1)):
                equality_test = False
            elif (param1.ndim > 0 and param1.shape[0] > 0 and isinstance(param2,
                np.ndarray) and param2.ndim > 0 and param2.shape[0] > 0):
                equality_test = (
                    param1.shape == param2.shape and param1.dtype == param2.dtype and (
                    _first_and_last_element(param1) == _first_and_last_element(
                        param2)))
            else:
                equality_test = np.all(param1 == param2)
        elif sparse.issparse(param1):
            # For sparse matrices equality doesn't work
            if not sparse.issparse(param2):
                equality_test = False
            elif param1.size == 0 or param2.size == 0:
                equality_test = (
                    param1.__class__ == param2.__class__ and param1.size == 0 and param2.size == 0)
            else:
                equality_test = (param1.__class__ == param2.__class__ and (
                _first_and_last_element(param1) == _first_and_last_element(
                    param2)) and param1.nnz == param2.nnz and param1.shape == param2.shape)
        else:
            # fall back on standard equality
            equality_test = param1 == param2
        if equality_test:
            warnings.warn("Estimator %s modifies parameters in __init__."
                          " This behavior is deprecated as of 0.18 and "
                          "support for this behavior will be removed in 0.20." % type(
                estimator).__name__, DeprecationWarning)
        else:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'does not seem to set parameter %s' % (
                               estimator, name))

    return new_object


def _first_and_last_element(arr):
    """Returns first and last element of numpy array or sparse matrix."""
    if isinstance(arr, np.ndarray) or hasattr(arr, 'data'):
        # numpy array or sparse matrix with .data attribute
        data = arr.data if sparse.issparse(arr) else arr
        return data.flat[0], data.flat[-1]
    else:
        # Sparse matrices without .data attribute. Only dok_matrix at
        # the time of writing, in this case indexing is fast
        return arr[0, 0], arr[-1, -1]
