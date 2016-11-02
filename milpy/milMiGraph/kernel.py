#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Kernel for miGraph

:author: Manuel Tuschen
:date: 23.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import numpy as np
from scipy.spatial.distance import sqeuclidean, pdist, squareform, cdist




def miGraphKernel_Bag(b1, b2, gamma, omega1, omega2):
    """
    This function calculates one kernel entry comparing two bags.
    Differently than stated in the publication, in their implementation
    Zhou et al. normalized by taking the square-root of the sum over the edge
    coefficients.

    Parameters
    ----------
    b1 : ndarray [N1]
        Instances of bag 1.
    b2 : ndarray [N2]
        Instances of bag 2.
    gamma : float
        rbf bandwidth.
    omega1 : dict
        The precomputed omega values for b1.
    omega2 : dict
        The precomputed omega values for b2.

    Returns
    ----------
    V : ndarray [B,B]
        Kernel output matrix.
    """

    w_b1 = np.sum(omega1, axis=1) # number of edges per instance
    w_b2 = np.sum(omega2, axis=1)  # number of edges per instance

    assert np.all(w_b1) > 0, "Zero encounterd for w_b1."
    assert np.all(w_b2) > 0, "Zero encounterd for w_b2."

    w_b1 = 1.0 / w_b1
    w_b2 = 1.0 / w_b2

    K_b1b2 = rbfKernel_Bag(b1, b2, gamma)
    II = np.ones((len(b1), len(b2)))

    return np.dot(np.dot(w_b1.T, K_b1b2), w_b2) / np.dot(np.dot(w_b1.T, II),w_b2) # written in the paper
    #return np.dot(np.dot(w_b1.T, K_b1b2), w_b2) / np.sqrt(np.dot(np.dot(w_b1.T, II),w_b2)) # original implementation



def miGraph_Mat(data1, data2,  gamma, omegas1=None, omegas2=None, tau="local",  metric="gaussian"):
    """
    This function builds the final normalized miGraph kernel matrix.

    Parameters
    ----------
    data1 : milData
        MIL data set.
    data2 : milData
        MIL data set.
    gamma : float
        rbf bandwidth.
    omegas1 : dict, optional
        The precomputed omega values for all bags of data1.
    omegas2 : dict, optional
        The precomputed omega values for all bags of data2.
    tau : float or string
        If float: threshold to overcome in order to connect two instances.
        If "local": tau is calculated as the mean distance for each bag.
        If "global": tau is calculated as the mean distance over all bags.
    metric : string
        'gaussian' or any method supported by scipy.spatial.distance.pdist()

    Returns
    ----------
    V : numpy.array [B,B]
        Kernel output matrix.
    """

    if omegas1 is None:
        omegas1 = calcOmegas(data1, gamma, tau, metric=metric)
    if omegas2 is None:
        if data1.keys != data2.keys:
            omegas2 = calcOmegas(data2, gamma, tau, metric=metric)
        else:
            omegas2 = omegas1

    # in the original implementation there is some normalization not mentioned in the paper
    #V_i = np.zeros((data1.N_B))
    #V_j = np.zeros((data2.N_B))
    V = np.zeros((data1.N_B, data2.N_B))

    #for i in range(data1.N_B):
    #    key1 = data1.keys[i]
    #    V_i[i] = miGraphKernel_Bag(data1.get_B(key1)[0], data1.get_B(key1)[0],gamma, omegas1[key1], omegas1[key1])
    #for j in range(data2.N_B):
    #    key2 = data2.keys[j]
    #    V_j[j] = miGraphKernel_Bag(data2.get_B(key2)[0], data2.get_B(key2)[0],gamma, omegas2[key2], omegas2[key2])


    if data1.keys == data2.keys:
        for i in range(data1.N_B):
            key1 = data1.keys[i]
            for j in range(i, data2.N_B):
                key2 = data2.keys[j]
                v = miGraphKernel_Bag(data1.get_B(key1)[0], data2.get_B(key2)[0], gamma, omegas1[key1], omegas2[key2])
                V[i, j] = v #/ (np.sqrt(V_i[i]) * np.sqrt(V_j[j]))
                if i != j:
                    V[j, i] = v #/ (np.sqrt(V_i[i]) * np.sqrt(V_j[j]))
        V += np.identity(V.shape[0]) * 1e-7 # add some value to grantee a positive definite kernel
        assert not np.any(np.isnan(V)), 'NaN detected in covMat_V!'
        assert np.allclose(V, V.T), 'covMat_V is not symmetric'
        assert np.all(np.linalg.eigvals(V) > 0), 'covMat_V is not positive definite'

    else:
        for i in range(data1.N_B):
            key1 = data1.keys[i]
            for j in range(data2.N_B):
                key2 = data2.keys[j]
                v = miGraphKernel_Bag(data1.get_B(key1)[0], data2.get_B(key2)[0], gamma, omegas1[key1], omegas2[key2], )
                V[i, j] = v #/ (np.sqrt(V_i[i]) * np.sqrt(V_j[j]))
    return V


################################################################################
#                                                                              #
#                           helper functions                                   #
#                                                                              #
################################################################################

def rbfKernel_Inst(x1, x2, gamma):
    """
    The rbf kernel between two instances
    """

    norm = sqeuclidean(x1,x2) # returns the squared euclidean norm
    k = np.exp(-gamma * norm)
    return k


def rbfKernel_Bag(b1, b2, gamma):
    """
    The rbf kernel between  between all instances of two bags.
    """

    b1 = np.array(b1)
    b2 = np.array(b2)

    # squared euclidean distance between instances of two bags
    sqFrobNorm = cdist(b1, b2, "sqeuclidean")

    # radial basis function
    k = np.exp(-gamma * sqFrobNorm)
    return k


def rbfKernel_Mat(X, gamma):
    """
    The rbf kernel between all instances of a data set.
    """

    X = np.array(X)

    sqFrobNorm = squareform(pdist(X, "sqeuclidean"))

    # Create GP covariance matrices
    K = np.exp(-gamma * sqFrobNorm) # instance level covariance matrix
    K += np.identity(K.shape[0]) * 1e-7  # add some value to grantee a positive definite kernel

    assert not np.any(np.isnan(K)), 'NaN detected in covMat_K!'
    assert np.allclose(K,  K.T), 'covMat_K is not symmetric'
    assert np.all(np.linalg.eigvals(K) > 0), 'covMat_K is not positive definite'

    return K


def distMat_Bag(b, gamma, metric='gaussian'):
    """
    This function calculates the inner bag distance matrix. Differently than
    stated in the publication, in their implementation Zhou et al. did not use
    the gaussian distance but the squared euclidean distance.
    """

    if metric == 'gaussian':
        distMat = squareform(pdist(b, metric="sqeuclidean"))
        distMat = 1 - np.exp(-gamma * distMat)
    else:
        distMat = squareform(pdist(b, metric=metric))

    return distMat


def omega_Bag(distMat, tau):
    """
    The output matrix represents the graph of connected instances in one bag.
    """
    if tau <= 0:
        raise ValueError("tau must not be larger than 0.")
    return (distMat < tau).astype(np.uint8, casting="safe")


def calcOmegas(data, gamma, tau,  metric="gaussian"):
    """
    Caclulate the omegas for all bags.
    """
    distMats = {}
    omegas = {}

    sumDistMats = 0
    numDistMats = 0
    tau_ = tau

    for i in range(data.N_B):
        key = data.keys[i]
        distMats[key] = distMat_Bag(data.get_B(key)[0], gamma, metric=metric)
        if tau == "global":
            sumDistMats += np.sum(distMats[key])
            numDistMats += len(distMats[key])

    if tau == "global":
        tau_ = sumDistMats / numDistMats

    for i in range(data.N_B):
        key = data.keys[i]
        if tau == "local":
            tau_ = np.mean(distMats[key])
        omegas[key] = omega_Bag(distMats[key], tau_)

    return omegas