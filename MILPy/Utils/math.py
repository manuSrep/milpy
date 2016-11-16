#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Commonly used scalar functions

:author: Manuel Tuschen
:date: 04.07.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import numpy as np

__all__ = ["sigmoid","d_sigmoid","logsumexp","d_logsumexp"]


def sigmoid(x):
    """
    Sigmoid function.
    """
    x = np.clip(x,-100,100)
    res = 1. / (1. + np.exp(-x))
    return np.round(res, 16)


def d_sigmoid(x):
    """
    Gradient of the sigmoid function
    """
    s = sigmoid(x)
    return s * (1. - s)


def logsumexp(x, c=1.):
    """
    Modified logsumexp function. c can be used to make the function more
    sensible for smaller values by setting c to high values.
    """
    x = x.copy() * c
    x_max = np.max(x)# we use x_max to avoid numerical overflows
    return (x_max + np.log(np.sum(np.exp(x - x_max)))) / c


def d_logsumexp(x, l, c=1.):
    """
    Gradient of modified logsumexp function.
    """
    x = x.copy() * c
    x_max = np.max(x)
    ex = np.exp(x-x_max)
    return ex[l] / np.sum(ex)