#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Prepare MIL datasets

:author: Manuel Tuschen
:date: 23.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function


import numpy as np
import pandas as pd
from miscpy import prepareLoading
from sklearn.preprocessing import scale

from milpy import milData


def prepareMusk1(path, name):

    print("Prepare Musk1:")
    fname = prepareLoading(name, path)

    # Create MIL problem from Musk1 data
    musk1_raw = np.array(pd.read_csv(fname, usecols=range(0,169), skip_blank_lines=True, header=None))

    # Scale data by z-score normalization
    features = musk1_raw[:, 2:-1].astype(np.float64)
    features = scale(features)
    keys = musk1_raw[:, 0]
    labels = musk1_raw[:, -1]

    musk1 = milData('musk1')
    for i, row in enumerate(features):
        key = keys[i]
        x = features[i]
        z = labels[i]
        musk1.add_x(key, x, z, UPDATE=False)
    musk1.save(path)

    print("Bags: ", musk1.N_B, "(+: {p}, -: {n})".format(p=np.sum(musk1.z==1), n=np.sum(musk1.z==0)))
    print("Instances: ", musk1.N_X, "(+: {p}, -: {n})".format(p=np.sum(musk1.y==1), n=np.sum(musk1.y==0)))
    print ("Features: ", musk1.N_D)
    print("\n")

    return musk1


def prepareMusk2(path, name):

    print("Prepare Musk2:")

    fname = prepareLoading(name, path)

    # Create MIL problem from Musk2 data
    musk2_raw = np.array(pd.read_csv(fname, usecols=range(0,169), skip_blank_lines=True, header=None))

    # Scale data by z-score normalization
    features = musk2_raw[:, 2:-1].astype(np.float64)
    features = scale(features)
    keys = musk2_raw[:, 0]
    labels = musk2_raw[:, -1]

    musk2 = milData('musk2')
    for i, row in enumerate(features):
        key = keys[i]
        x = features[i]
        z = labels[i]
        musk2.add_x(key, x, z, UPDATE=False )
    musk2.save(path)

    print("Bags: ", musk2.N_B, "(+: {p}, -: {n})".format(p=np.sum(musk2.z==1), n=np.sum(musk2.z==0)))
    print("Instances: ", musk2.N_X, "(+: {p}, -: {n})".format(p=np.sum(musk2.y==1), n=np.sum(musk2.y==0)))
    print ("Features: ", musk2.N_D)
    print("\n")

    return musk2


if __name__ == "__main__":
    prepareMusk1("Musk1/", "Musk1.csv")
    prepareMusk2("Musk2/", "Musk2.csv")