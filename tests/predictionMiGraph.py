#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 23.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import matplotlib
matplotlib.use('agg')
import numpy as np
from sklearn import metrics

from milpy import miGraph
from milpy.milUtil.milCV import train_test_split, KFold, cross_val_predict, cross_val_score
from milpy.milUtil.milMetrics import make_scorer, accuracy_score
from milpy.milUtil.milData import milData


if __name__ == "__main__":


    MUSK1 = True
    MUSK2 = False
    ELEPHANT = False
    TIGER = False
    FOX = False


    ################################################################################
    #                                                                              #
    #                                  Musk 1                                      #
    #                                                                              #
    ################################################################################

    if MUSK1:

        print("Start Musk1 for miGraph:")
        musk1 = milData("musk1")
        musk1.load("../Datasets/Musk/Musk1")
        musk1_train, musk1_test = train_test_split(musk1,test_size=0.4,random_state=12345)

        # Prepare estimator
        estimator = miGraph()
        # Fit trainings data
        estimator.fit(musk1_train)
        # Predict trainings and test data
        zPredTrain, _ = estimator.predict(musk1_train)
        zPredTest, _ = estimator.predict(musk1_test)
        print('Test Score: ', metrics.accuracy_score(musk1_test.z, zPredTest))
        print('Training Score: ', metrics.accuracy_score(musk1_train.z, zPredTrain), '\n')

        # perform cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=12345)
        scorere = make_scorer(accuracy_score)
        print(cross_val_score(estimator,musk1,scoring=scorere,cv=cv,fit_params={"k":10, "n":1}, n_jobs=-1 ))