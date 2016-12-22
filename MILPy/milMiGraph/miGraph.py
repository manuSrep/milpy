#!/usr/bin/python
# -*- coding: utf8 -*-

"""
miGraph

:author: Manuel Tuschen
:date: 08.12.2016
:license: GPL3
"""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC

from .kernel import calcOmegas, miGraph_Mat

__all__ = ["miGraph"]


class miGraph(BaseEstimator):
    """
    Estimator for miGraph.
    """

    def __init__(self, metric="gaussian", gamma="num_feature", tau="local",
            C=10.0, degree=3, probability=False, shrinking=True, tol=0.001,
            cache_size=200, max_iter=None, random_state=None):
        """
        Initialize the estimator

        Parameters:
        -----------
        metric : string, 'gaussian' or any method supported by
        scipy.spatial.distance.pdist(), optional
            The metric used for distance calculations.
        gamma : float, 'num_feature' or 'med_dist', optional
            rbf bandwidth. If 'num_feature the bandwidth is calculated according
            to the number of features.'If 'med_dist', the caclulation is based
            on the median distance of the features.
        tau : float between 0 and 1, 'local' or 'global, optional' If 'local',
            tau is calculated as the mean distance for each bag. If 'global',
             tau is calculated as the mean distacne over all bags.
        C : float, optional (default=10)
            Penalty parameter C of the error term.
        degree : int, optional (default=3)
            Degree of the polynomial kernel function ('poly').
            Ignored by all other kernels.
        probability : boolean, optional (default=False)
            Whether to enable probability estimates. This must be enabled prior
            to calling `fit`, and will slow down that method.
        shrinking : boolean, optional (default=True)
            Whether to use the shrinking heuristic.
        tol : float, optional (default=1e-3)
            Tolerance for stopping criterion.
        cache_size : float, optional
            Specify the size of the kernel cache (in MB).
        max_iter : int, optional
            Hard limit on iterations within solver, or None for no limit.
        random_state : int seed, RandomState instance, or None (default)
            The seed of the pseudo random number generator to use when
            shuffling the data for probability estimation.
        """
        self.metric = metric
        self.gamma = gamma
        self.tau = tau
        self.C = C
        self.degree = degree
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.random_state = random_state

        self._data_train = None
        self._omegas_train = None
        self._kernel_train = None

    def estimate_gamma(self, data, method='num_feature'):
        """
        Find a gamma estimate for the data

        Parameters:
        -----------
        data : milData
            The training data
        method: string
            The method to estimate sigma. 'If 'med_dist', the caclulation is
            based on the median distance of the features.
        """
        if method == 'num_feature':
            gamma = 1.0 / (data.N_D) / 2
        elif method == 'med_dist':
            selection = data.X
            gamma = 1.0 / (np.median(pdist(selection)) ** 2)
        else:
            gamma = self.gamma

        return gamma

    def fit(self, data):
        """
        Fit the training data.

        Parameters:
        -----------
        data : milData
            The training data.
        """
        self.gamm_estimate = self.estimate_gamma(data, method=self.gamma)
        if self.max_iter == None:
            self.max_iter = -1

        self._data_train = data
        self._omegas_train, self.tau_estimate = calcOmegas(self._data_train,
            self.gamm_estimate, self.tau, metric=self.metric)
        self._kernel_train = miGraph_Mat(self._data_train, self._data_train,
            self.gamm_estimate, self._omegas_train, self._omegas_train)

        self.SVM = SVC(kernel="precomputed", C=self.C, shrinking=self.shrinking,
            probability=self.probability, tol=self.tol,
            cache_size=self.cache_size, class_weight=None, verbose=False,
            max_iter=self.max_iter, decision_function_shape=None,
            random_state=self.random_state)
        self.SVM.fit(self._kernel_train, self._data_train.z)

        return self.gamm_estimate, self.tau_estimate

    def predict(self, data):
        """
        Predict bag labels.

        Parameters:
        -----------
        data : milData
            The training data.

        Returns
        ----------
        z_pred : ndarray
            The predicted bag labels.
        : None
        """

        if self._kernel_train is None:
            raise NotFittedError("You must run .fit before .predict!")

        omegas_test, _ = calcOmegas(data, self.gamm_estimate, self.tau,
            metric=self.metric)
        kernel_test = miGraph_Mat(data, self._data_train, self.gamm_estimate,
            omegas_test, self._omegas_train)

        z_pred = self.SVM.predict(kernel_test)

        return z_pred, None

    def predict_proba(self, data):
        """
        Predict bag label probabilities.

        Parameters:
        -----------
        data : milData
            The training data.

        Returns:
        --------
        z_prob : ndarray
            The predicted bag labels.
        : None
        """

        if self._kernel_train is None:
            raise NotFittedError("You must run .fit before .predict!")

        omegas_test, _ = calcOmegas(data, self.gamm_estimate, self.tau,
            metric=self.metric)
        kernel_test = miGraph_Mat(data, self._data_train, self.gamm_estimate,
            omegas_test, self._omegas_train)

        z_prob = self.SVM.predict_proba(kernel_test)

        return z_prob, None

    def decision_function(self, data):
        """
        Predict bag label probabilities.

        Parameters:
        -----------
        data : milData
            The training data.

        Returns:
        --------
        z_prob : ndarray
            The predicted bag labels.
        : None
        """

        if self._kernel_train is None:
            raise NotFittedError("You must run .fit before .predict!")

        omegas_test, _ = calcOmegas(data, self.gamm_estimate, self.tau,
            metric=self.metric)
        kernel_test = miGraph_Mat(data, self._data_train, self.gamm_estimate,
            omegas_test, self._omegas_train)

        z_prob = self.SVM.decision_function(kernel_test)

        return z_prob, None
