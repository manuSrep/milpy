#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Bayesian kNN MIL classifier.

:author: Manuel Tuschen
:date: 20.08.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator

from ._hausdorff import kth_hausdorff




class BayesianKNN(BaseEstimator):

    def __init__(self, k=2):
        """
        Initialize the estimator
        """
        self.k = k


    def fit(self, data, **fit_params):
        """
        Fit data for Bayesian knn classification
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

        for i in range(0,self.n_train):
            if i < self.n_neg:
                bag_i = self.neg_train.get_B(self.neg_train.keys[i])[0]
            else:
                bag_i = self.pos_train.get_B(self.pos_train.keys[i - self.n_neg])[0]

            for j in range(i+1,self.n_train):
                if j < self.n_neg:
                    bag_j = self.neg_train.get_B(self.neg_train.keys[j])[0]
                else:
                    bag_j = self.pos_train.get_B(self.pos_train.keys[j - self.n_neg])[0]

                self.dist_train[i,j] = kth_hausdorff(bag_i,bag_j, k_A=0, k_B=0, metric="euclidean")
                #self.dist_train[j, i] = self.dist_train[i,j]


        # For a Bayesin kNN classifier we need to count which label is most
        # probable depending on how many k nearest neighbours fall in which
        # class.

        # we calculate the overall probability for each class
        p_neg = self.n_neg / self.n_train
        p_pos = self.n_pos / self.n_train

        # we need to find the most probable class depending on k
        self.max_propab= np.zeros((self.k+1,2)) # entries for k labels being in class 1 and 0 respectively

        # find order of nearest neighbours
        n_given_neg = np.argsort(self.dist_train[0:self.n_neg], axis=1)[:,0:self.k]
        n_given_pos = np.argsort(self.dist_train[self.n_neg:self.n_train], axis=1)[:,0:self.k]

        # count how positive neighbours there are given the label is negative or positive
        count_neg = np.sum((n_given_neg >= self.n_neg), axis=1)
        count_pos = np.sum((n_given_pos >= self.n_neg), axis=1)
        for i in range(self.k+1):
            self.max_propab[i,0] = np.sum(count_neg == i)
            self.max_propab[i,1] = np.sum(count_pos == i)

        self.max_propab[:,0] /= np.sum(self.max_propab[:,0]) # normalize for number of entries
        self.max_propab[:,1] /= np.sum(self.max_propab[:,1]) # normalize for number of entries

        self.max_propab[:,0] *= p_neg
        self.max_propab[:,1] *= p_pos


    def predict(self, data):
        """
        Predict on new test data.
        """

        if self.dist_train is None:
            raise ValueError("You must run fit before predict")

        n_test = data.N_B

        zPred = np.zeros(n_test)

        for i in range(n_test):

            # compare one test bag to all training bags
            dist = np.full((self.n_train+1, self.n_train+1), np.inf)
            dist[1:None,1:None] = self.dist_train
            for j in range(0,self.n_train):
                if j < self.n_neg:
                    bag_j = self.neg_train.get_B(self.neg_train.keys[j])[0]
                else:
                    bag_j = self.pos_train.get_B(self.pos_train.keys[j - self.n_neg])[0]

                dist[0,j+1] = kth_hausdorff(data.get_B(data.keys[i])[0],bag_j,  k_A=0, k_B=0, metric="euclidean")

            # find k minimal distances between test and training sets
            dist_idx = np.argsort(dist[0])[0:self.k]
            # are there more positive labels?
            pos_count = np.sum(dist_idx >= self.n_neg)
            zPred[i] = np.argmax(self.max_propab[pos_count])

        return zPred, None