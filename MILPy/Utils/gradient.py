#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Simple gradient checker

:author: Manuel Tuschen
:date: 20.07.2016
:license: GPL3
"""

import numpy as np

__all__ = ["check_grad_element", "check_grad_vector", "check_grad_element_wise"]


def check_grad_element(func, grad, x0, *args, **kwargs):
    """
    Verify the gradient of a scalar univariate function.

    Parameters
    ----------
    func : callable func(x0, *args)
        Function whose derivative is to be checked.
    grad : callable grad(x0, *args)
        Gradient of func.
    x0 : number
        Point to check grad against symmetric difference approximation of grad
        using func.
    args : *args, optional
        Extra arguments passed to func and grad.
    epsilon : float, optional
        Step size used for the finite difference approximation. It defaults to
        sqrt(numpy.finfo(float).eps), which is approximately 1.49e-08.

    Returns
    -------
    err : float
        The absolute difference between grad(x0, *args) and the finite
        difference approximation of grad using func at the point x0.
    """
    _epsilon = np.sqrt(np.finfo(float).eps)
    epsilon = kwargs.pop('epsilon', _epsilon)

    f1 = func(x0 - 0.5 * epsilon, *args)
    f2 = func(x0 + 0.5 * epsilon, *args)
    f_approx = (f2 - f1) / (epsilon)

    f_ = grad(x0, *args)

    diff = np.sqrt(np.sum((f_approx - f_) ** 2))
    return diff


def check_grad_vector(func, grad, x0, *args, **kwargs):
    """
    Verify the gradient of a scalar multivariate function.

    Parameters
    ----------
    func : callable func(x0, *args)
        Function whose derivative is to be checked.
    grad : callable grad(x0, *args)
        Gradient of func.
    x0 : ndarray
        Points to check grad against symmetric difference approximation of grad
        using func.
    args : *args, optional
        Extra arguments passed to func and grad.
    epsilon : float, optional
        Step size used for the finite difference approximation. It defaults to
        sqrt(numpy.finfo(float).eps), which is approximately 1.49e-08.
    mode : string, optional
        If "sequential" (default), the gradient will be checked in direction of
        each unit vector. If "random", the gradient will be checked in a
        random direction.

    Returns
    -------
    err : float
        The square root of the sum of squares (i.e. the 2-norm) of the
        difference between grad(x0, *args) and the finite difference
        approximation of grad using func at the points x0.
    """

    _epsilon = np.sqrt(np.finfo(float).eps)
    epsilon = kwargs.pop('epsilon', _epsilon)

    _mode = 'sequential'
    mode = kwargs.pop('mode', _mode)

    if mode == 'random':
        np.random.seed(111)

        ei = np.random.rand(len(x0))
        epsi = epsilon * ei

        f1 = func(x0 - 0.5 * epsi, *args)
        f2 = func(x0 + 0.5 * epsi, *args)
        f_approx = (f2 - f1) / (epsilon)

        f_ = np.dot(grad(x0, *args), ei)

        diff = np.sqrt(np.sum((f_approx - f_) ** 2))

    else:
        f_approx = np.zeros((len(x0)))
        ei = np.zeros(len(x0))
        for i in range(len(x0)):
            ei[i] = 1
            epsi = epsilon * ei

            f1 = func(x0 - 0.5 * epsi, *args)
            f2 = func(x0 + 0.5 * epsi, *args)
            f_approx[i] = (f2 - f1) / (epsilon)

            ei[i] = 0
        diff = np.sqrt(np.sum((f_approx - grad(x0, *args)) ** 2))

    return diff


def check_grad_element_wise(func, grad, x0, *args, **kwargs):
    """
    Verify the gradient of a scalar multivariate function with respect to one
    element.

    Parameters
    ----------
    func : callable func(x0, *args)
        Function whose derivative is to be checked.
    grad : callable grad(x0, *args)
        Gradient of func.
    x0 : ndarray
        Points to check grad against symmetric difference approximation of grad
        using func.
    args : *args, optional
        Extra arguments passed to func and grad.
    epsilon : float, optional
        Step size used for the finite difference approximation. It defaults to
        sqrt(numpy.finfo(float).eps), which is approximately 1.49e-08.
    element : int, optional
        If given, the gradient will be calculated only with respect to this
        element of x0. If not given, the gradient check will be calculated for
        all each element separately and the maximum difference will be returnd.

    Returns
    -------
    err : float
        The square maximal difference between grad(x0, *args) and the finite
        difference approximation of grad using func at the "elemnt" point x0 or
        all points calculated one after the other.
    """

    _epsilon = np.sqrt(np.finfo(float).eps)
    epsilon = kwargs.pop('epsilon', _epsilon)

    _element = None
    element = kwargs.pop('element', _element)

    if element is not None:
        i = element
        ei = np.zeros(len(x0))
        ei[element] = 1
        epsi = epsilon * ei

        f_ = grad(x0, element, *args)

        f1 = func(x0 - 0.5 * epsi, *args)
        f2 = func(x0 + 0.5 * epsi, *args)
        f_approx = (f2 - f1) / (epsilon)

        diff = np.sqrt(np.sum((f_approx - f_) ** 2))
        return diff

    else:
        maxv = 0.0
        for i in range(len(x0)):
            ei = np.zeros(len(x0))
            ei[i] = 1
            epsi = epsilon * ei

            f_ = grad(x0, i, *args)

            f1 = func(x0 - 0.5 * epsi, *args)
            f2 = func(x0 + 0.5 * epsi, *args)
            f_approx = (f2 - f1) / (epsilon)

            diff = np.sqrt(np.sum((f_approx - f_) ** 2))
            maxv = max(maxv, diff)

        return maxv
