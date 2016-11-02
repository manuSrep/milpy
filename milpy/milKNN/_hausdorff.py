#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 16.08.2016
:license: GPL3
"""


import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean

def kth_hausdorff(A, B, k_A="max", k_B="max", metric="euclidean"):
    """
    Compute the kth Hausdorff distacnce.
    If k_A and K_B == "max", this resmables the original Hausdorff distance
    """
    N1 = len(A)
    N2 = len(B)

    if k_A == "max":
        k_A = N1-1
    if k_A == "min":
        k_A = 0
    if k_B == "max":
        k_B = N2-1
    if k_B == "min":
        k_B = 0

    if k_A > N1-1:
        raise ValueError("k invalide")
    if k_B > N2-1:
        raise ValueError("k invalide")

    dist1 = cdist(A, B, metric=metric)
    dist2 = cdist(B, A, metric=metric)

    sort_dist1 = sorted(np.min(dist1, axis=1))[k_A]
    sort_dist2 = sorted(np.min(dist2, axis=1))[k_B]

    return(max(sort_dist1, sort_dist2))

