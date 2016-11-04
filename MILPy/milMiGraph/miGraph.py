#!/usr/bin/python
# -*- coding: utf8 -*-

"""
miGraph

:author: Manuel Tuschen
:date: 23.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.spatial.distance import sqeuclidean, pdist, squareform, cdist
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from miscpy import prepareSaving

from .kernel import calcOmegas, miGraph_Mat

__all__ = ["miGraph"]


class miGraph(BaseEstimator):
    """
    Estimator for miGraph.

    Attributes:
    -----------
    gamma : float
        rbf bandwidth.
    tau : string
        If float: threshold to overcome in order to connect two instances.
        If "local": tau is calculated as the mean distance for each bag.
        If "global": tau is calculated as the mean distance over all bags.
    metric : string
        'gaussian' or any method supported by scipy.spatial.distance.pdist().

    Methods:
    --------
    set_params(**params)
        Set the parameters.
    get_params(deep=True)
        Return the parameters.
    save_params(file, path)
        Save the parameters to file
    fit(data, fit_params=None)
        Fit the trainings data
    predict(data)
        Return prediction.
    """

    def __init__(self, kernel_metric="gaussian", gamma="num_feature", tau="local", C=10.0, degree=3, shrinking=True, probability=False, tol=0.001, cache_size=200, max_iter=-1, random_state=None):
        """
        Initialize the estimator

        Parameters:
        -----------
        C : float
            milSVM regularizer.
        gamma : float
            rbf bandwidth.
        tau : string
            If float: threshold to overcome in order to connect two instances.
            If "local": tau is calculated as the mean distance for each bag.
            If "global": tau is calculated as the mean distacne over all bags.
        metric : string
            'gaussian' or any method supported by scipy.spatial.distance.pdist().
        """
        self.metric = kernel_metric
        self.gamma = gamma
        self.tau = tau
        self.C = C

        self.degree = degree
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.chache_size = cache_size
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
            The method to estimate sigma. Either 'med_dist', the median
            distance or 'num_feature' the inverse number of features.
        """
        if method == 'num_feature':
            self.gamma = 1.0 / (data.N_D) * 2

        if method == 'med_dist':
            selection = data.X
            self.gamma = 1.0 / (np.median(pdist(selection)) ** 2)


    def save_params(self, file, path):
        """
        Predict bag labels.

        Parameters:
        -----------
        file : string
            The name of the file to dave to
        path : string
            The path to store in.
        """

        params = pd.Series(self.get_params())
        fname = prepareSaving(file, path, ".json")
        params.to_json(fname)


    def fit(self, data):
        """
        Fit the training data.

        Parameters:
        -----------
        data : milData
            The training data.
        """
        if self.gamma == "num_feature" or self.gamma == "med_dist":
            self.estimate_gamma(data, method=self.gamma)


        self._data_train = data
        self._omegas_train = calcOmegas(self._data_train, self.gamma, self.tau, metric=self.metric)
        self._kernel_train = miGraph_Mat(self._data_train, self._data_train, self.gamma, self._omegas_train, self._omegas_train, tau=self.tau, metric=self.metric)

        self.SVM = SVC(kernel="precomputed", C=self.C, probability=self.probability)
        self.SVM.fit(self._kernel_train, self._data_train.z)


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
            raise ValueError("You must run .fit before .predict!")

        omegas_test = calcOmegas(data, self.gamma, self.tau, metric=self.metric)
        kernel_test = miGraph_Mat(data, self._data_train, self.gamma, omegas_test, self._omegas_train, tau=self.tau, metric=self.metric)

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
            raise ValueError("You must run .fit before .predict!")

        omegas_test = calcOmegas(data, self.gamma, self.tau, metric=self.metric)
        kernel_test = miGraph_Mat(data, self._data_train, self.gamma, omegas_test, self._omegas_train, tau=self.tau, metric=self.metric)

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
            raise ValueError("You must run .fit before .predict!")

        omegas_test = calcOmegas(data, self.gamma, self.tau, metric=self.metric)
        kernel_test = miGraph_Mat(data, self._data_train, self.gamma, omegas_test,
                                  self._omegas_train, tau=self.tau, metric=self.metric)

        z_prob = self.SVM.decision_function(kernel_test)

        return z_prob, None