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
from miscpy import prepareLoading
from scipy.io import loadmat
from sklearn.preprocessing import scale

from milpy import milData




def prepareElephant(path, name):

    print("Prepare Elephant:")
    fname = prepareLoading(name, path)

    # Create MIL problem from Elepahnt data
    features = loadmat(fname)["Data"]
    features = scale(features)
    keys = loadmat(fname)['bags']
    labels = loadmat(fname)['labels']

    elephant = milData('elephant')
    b = 0
    for i, row in enumerate(features):
        x = row
        key = keys[i][0]
        z = labels[b][0]
        elephant.add_x(key,x,z,UPDATE=False)
        if i < len(features)-1 and key != keys[i+1]: # will the next key belong to the next bag?
            b += 1
    elephant.save(path)

    print("Bags: ", elephant.N_B, "(+: {p}, -: {n})".format(p=np.sum(elephant.z==1), n=np.sum(elephant.z==0)))
    print("Instances: ", elephant.N_X, "(+: {p}, -: {n})".format(p=np.sum(elephant.y==1), n=np.sum(elephant.y==0)))
    print ("Features: ", elephant.N_D)
    print("\n")

    return elephant


def prepareTiger(path, name):

    print("Prepare Tiger:")
    fname = prepareLoading(name, path)

    # Create MIL problem from Tiger data
    features = loadmat(fname)["Data"]
    features = scale(features)
    keys = loadmat(fname)['bags']
    labels = loadmat(fname)['labels']

    tiger = milData('tiger')
    b = 0
    for i, row in enumerate(features):
        x = row
        key = keys[i][0]
        z = labels[b][0]
        tiger.add_x(key,x,z,UPDATE=False)
        if i < len(features)-1 and key != keys[i+1]: # will the next key belong to the next bag?
            b += 1
    tiger.save(path)

    print("Bags: ", tiger.N_B, "(+: {p}, -: {n})".format(p=np.sum(tiger.z==1), n=np.sum(tiger.z==0)))
    print("Instances: ", tiger.N_X, "(+: {p}, -: {n})".format(p=np.sum(tiger.y==1), n=np.sum(tiger.y==0)))
    print ("Features: ", tiger.N_D)
    print("\n")

    return tiger


def prepareFox(path, name):

    print("Prepare Fox:")
    fname = prepareLoading(name, path)

    # Create MIL problem from Fox data
    features = loadmat(fname)["Data"]
    features = scale(features)
    keys = loadmat(fname)['bags']
    labels = loadmat(fname)['labels']

    b=0
    fox = milData('fox')
    for i, row in enumerate(features):
        x = row
        key = keys[i][0]
        z = labels[b][0]
        fox.add_x(key,x,z,UPDATE=False)
        if i < len(features)-1 and key != keys[i+1]: # will the next key belong to the next bag?
            b += 1
    fox.save(path)

    print("Bags: ", fox.N_B, "(+: {p}, -: {n})".format(p=np.sum(fox.z==1), n=np.sum(fox.z==0)))
    print("Instances: ", fox.N_X, "(+: {p}, -: {n})".format(p=np.sum(fox.y==1), n=np.sum(fox.y==0)))
    print ("Features: ", fox.N_D)
    print("\n")

    return fox



if __name__ == "__main__":

    prepareElephant("Elephant/", "elephant.mat")
    prepareFox("Fox/", "fox.mat")
    prepareTiger("Tiger/", "tiger.mat")
