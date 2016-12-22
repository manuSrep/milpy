#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Citation kNN MIL classifier.

:author: Manuel Tuschen
:date: 20.08.2016
:license: GPL3
"""

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ._hausdorff import kth_hausdorff


class CitationKNN(BaseEstimator):
    def __init__(self, refs=2, citers=None, indicator=0):
        """
        Initialize the estimator

        Parameters:
        -----------
        refs : int, optional
            The number of refrences.
        citers : int, optional
            The number of citers
        """

        self.refs = refs
        if citers is None:
            self.citers = refs + 2
        else:
            self.citers = citers
        self.indicator = indicator

    def fit(self, data, **fit_params):
        """
        Fit the training data.

        Parameters:
        -----------
        data : milData
            The training data.
        """
        self.n_train = data.N_B
        self.neg_train = deepcopy(data)
        for key in data.keys:
            if self.neg_train.get_B(key)[1] > 0:
                self.neg_train.del_B(key)
        self.n_neg = self.neg_train.N_B
        self.pos_train = deepcopy(data)
        for key in data.keys:
            if self.pos_train.get_B(key)[1] <= 0:
                self.pos_train.del_B(key)
        self.n_pos = self.pos_train.N_B

        # We compute the minimal hausdorff distance between all training
        # samples. The resulting distance matrix will be diagonal but we only
        # compute the upper triangular matrix and set the lower triangular
        # matrix including the diagonal to infinity for proper distance
        # comparison. The distance matrix is designed to have two blocks.
        # The first "n_neg" entries contains only negative bags, the
        # remaining part contains the positive bags.
        self.dist_train = np.full((self.n_train, self.n_train), np.inf)

        for i in range(0, self.n_train):
            if i < self.n_neg:
                bag_i = self.neg_train.get_B(self.neg_train.keys[i])[0]
            else:
                bag_i = \
                    self.pos_train.get_B(self.pos_train.keys[i - self.n_neg])[0]

            for j in range(i + 1, self.n_train):
                if j < self.n_neg:
                    bag_j = self.neg_train.get_B(self.neg_train.keys[j])[0]
                else:
                    bag_j = \
                    self.pos_train.get_B(self.pos_train.keys[j - self.n_neg])[0]

                self.dist_train[i, j] = kth_hausdorff(bag_i, bag_j, k_A=0,
                    k_B=0, metric="euclidean")
                self.dist_train[j, i] = self.dist_train[i, j]

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

        if self.dist_train is None:
            raise NotFittedError("You must run fit before predict")

        n_test = data.N_B

        labels = np.zeros(n_test)

        for i in range(n_test):

            # compare one test bag to all training bags
            dist = np.full((self.n_train + 1, self.n_train + 1), np.inf)
            dist[1:None, 1:None] = self.dist_train
            for j in range(0, self.n_train):
                if j < self.n_neg:
                    bag_j = self.neg_train.get_B(self.neg_train.keys[j])[0]
                else:
                    bag_j = \
                    self.pos_train.get_B(self.pos_train.keys[j - self.n_neg])[0]

                dist[0, j + 1] = kth_hausdorff(data.get_B(data.keys[i])[0],
                    bag_j, k_A=0, k_B=0, metric="euclidean")

            rp = 0  # get the references and citers of current test bag
            rn = 0
            cp = 0
            cn = 0

            idx = np.argsort(dist[0])  # count reference labels
            for r in range(self.refs):
                if idx[r] >= self.n_neg:
                    rp += 1
                else:
                    rn += 1

            for c in range(self.n_neg, self.n_train):  # count positive citers
                idx = np.argsort(dist[c])
                if np.where(idx == 0)[0] <= self.citers:
                    cp += 1
            for c in range(0, self.n_neg):  # count negative citers
                idx = np.argsort(dist[c])
                if np.where(idx == 0)[0] <= self.citers:
                    cn += 1

            # weigt references and citers
            pos = rp + cp
            neg = rn + cn
            if pos > neg:
                labels[i] = 1
            else:
                if pos == neg:
                    if self.indicator == 1:
                        labels[i] = 1
        return labels, None
