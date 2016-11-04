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

from __future__ import division, absolute_import, unicode_literals, print_function

import warnings
from copy import deepcopy
from scipy import sparse

import numpy as np
import sklearn as skl
from sklearn import model_selection
from joblib import Parallel, delayed

__all__ = ["KFold","train_test_split", "cross_val_predict", "cross_val_score"]


################################################################################
#                                                                              #
#                       Splitter Classes                                       #
#                                                                              #
################################################################################


class KFold():
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state


    def split(self, milData):
        """
        Generate keys to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
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
        n_samples = milData.N_B

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
            test_mask = np.zeros(milData.N_B, dtype=np.bool)
            test_mask[indices[start:stop]] = True
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            yield np.array(milData.keys)[train_index], np.array(milData.keys)[test_index]
            current = stop


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
    keys_train, keys_test, _, _ = skl.model_selection.train_test_split(data.keys, data.keys, test_size=test_size, train_size=train_size, random_state=random_state)

    data_train = deepcopy(data)
    for key in keys_test:
        data_train.del_B(key)

    data_test = deepcopy(data)
    for key in keys_train:
        data_test.del_B(key)

    return data_train, data_test


################################################################################
#                                                                              #
#                       Model validation                                       #
#                                                                              #
################################################################################

def cross_val_score(estimator, milData, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs'):

    """
    Evaluate a score by cross-validation
    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    milData : milData object
        The data to fit.
    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
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
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : integer, optional
        The verbosity level.
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
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score)(deepcopy(estimator), milData, scoring, train, test, verbose, None, fit_params) for train, test in cv.split(milData))
    return np.array(scores)[:, 0]


def cross_val_predict(estimator, milData, groups=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', method='predict'):
    """
    Generate cross-validated estimates for each input data point

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    milData : milData object
        The data to fit.
    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.
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
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : integer, optional
        The verbosity level.
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
    predictions : ndarray
        This is the result of calling ``method``
    """
    # Ensure the estimator has implemented the passed decision function
    if not callable(getattr(estimator, method)):
        raise AttributeError('{} not implemented in estimator'.format(method))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(deepcopy(estimator), milData, train, test, verbose, fit_params, method)for train, test in cv.split(milData))

    # Concatenate the predictions
    predictions_z = [pred_block_z for pred_block_z, _, _ in prediction_blocks]
    predictions_y = [pred_block_y for _, pred_block_y, _ in prediction_blocks]

    test_keys = np.concatenate([key for _, _, key in prediction_blocks])
    test_indices = [list(test_keys).index(key) for key in milData.keys]
    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices), dtype=int)

    predictions_z = np.concatenate(predictions_z)
    if not None in predictions_y:
        predictions_y = np.concatenate(predictions_y)
        return predictions_z[inv_test_indices], predictions_y[inv_test_indices]
    else:
        return predictions_z[inv_test_indices], None

def cross_val_predictions():
    pass


def validation_curve(estimator,data, param_name, param_range, groups=None,
                     cv=None, scoring=None, n_jobs=1, pre_dispatch="all",
                     verbose=0):
    """Validation curve.
    Determine training and test scores for varying parameter values.
    Compute scores for an estimator with different values of a specified
    parameter. This is similar to grid search with one parameter. However, this
    will also compute training scores and is merely a utility for plotting the
    results.
    Read more in the :ref:`User Guide <learning_curve>`.
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
    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.
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
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    pre_dispatch : integer or string, optional
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The string can
        be an expression like '2*n_jobs'.
    verbose : integer, optional
        Controls the verbosity: the higher, the more messages.
    Returns
    -------
    train_scores : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.
    test_scores : array, shape (n_ticks, n_cv_folds)
        Scores on test set.
    Notes
    -----
    See :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`
    """

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)
    out = parallel(delayed(_fit_and_score)(
        estimator,data, scoring, train, test, verbose,
        parameters={param_name: v}, fit_params=None, return_train_score=True)
        for train, test in cv.split(data) for v in param_range)

    out = np.asarray(out)
    n_params = len(param_range)
    n_cv_folds = out.shape[0] // n_params

    out = out.reshape(n_cv_folds, n_params, 2, 2).transpose((2,1,0,3))
    return out[0], out[1]


def _fit_and_score(estimator, milData, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score='raise'):
    """Fit estimator and compute scores for a given dataset split.
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
        Indices of training samples.
    test : array-like, shape (n_test_samples,)
        Indices of test samples.
    verbose : integer
        The verbosity level.
    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : boolean, optional, default: False
        Compute and return score on training set.
    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    Returns
    -------
    train_score : float, optional
        Score on training set, returned only if `return_train_score` is `True`.
    test_score : float
        Score on test set.
    n_test_samples : int
        Number of test samples.
    fit_time : float
        Time spent for fitting in seconds.
    score_time : float
        Time spent for scoring in seconds.
    parameters : dict or None, optional
        The parameters that have been evaluated.
    """

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, v) for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    data_train = _safe_split(milData, train)
    data_test = _safe_split(milData, test)

    estimator.fit(data_train, **fit_params)
    test_score = scorer(estimator, data_test)

    if return_train_score:
        train_score = scorer(estimator, data_train)

    ret = [train_score, test_score] if return_train_score else [test_score]
    return ret


def _fit_and_predict(estimator,milData, train, test, verbose, fit_params,
                     method):
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
        Indices of training samples.
    test : array-like, shape (n_test_samples,)
        Indices of test samples.
    verbose : integer
        The verbosity level.
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

    data_train = _safe_split(milData, train)
    data_test = _safe_split(milData, test)

    estimator.fit(data_train, **fit_params)
    func = getattr(estimator, method)
    predictions_z, predictions_y = func(data_test)
    return predictions_z, predictions_y, test


def _safe_split(data, keys):
    """
    Create milData subset.
    """
    new = deepcopy(data)

    for key in data.keys:
        if key not in keys:
            new.del_B(key)

    return new

def clone(estimator, safe=True):
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
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params'):
        if not safe:
            return deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a scikit-learn estimator "
                            "as it does not implement a 'get_params' methods."
                            % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
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
            elif (param1.ndim > 0
                    and param1.shape[0] > 0
                    and isinstance(param2, np.ndarray)
                    and param2.ndim > 0
                    and param2.shape[0] > 0):
                equality_test = (
                    param1.shape == param2.shape
                    and param1.dtype == param2.dtype
                    and (_first_and_last_element(param1) ==
                         _first_and_last_element(param2))
                )
            else:
                equality_test = np.all(param1 == param2)
        elif sparse.issparse(param1):
            # For sparse matrices equality doesn't work
            if not sparse.issparse(param2):
                equality_test = False
            elif param1.size == 0 or param2.size == 0:
                equality_test = (
                    param1.__class__ == param2.__class__
                    and param1.size == 0
                    and param2.size == 0
                )
            else:
                equality_test = (
                    param1.__class__ == param2.__class__
                    and (_first_and_last_element(param1) ==
                         _first_and_last_element(param2))
                    and param1.nnz == param2.nnz
                    and param1.shape == param2.shape
                )
        else:
            # fall back on standard equality
            equality_test = param1 == param2
        if equality_test:
            warnings.warn("Estimator %s modifies parameters in __init__."
                          " This behavior is deprecated as of 0.18 and "
                          "support for this behavior will be removed in 0.20."
                          % type(estimator).__name__, DeprecationWarning)
        else:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'does not seem to set parameter %s' %
                               (estimator, name))

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
