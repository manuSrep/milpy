#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Optimizer to complete scipy.optimize.minimize methods.
:author: Manuel Tuschen
:date: 06.07.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import numpy as np
import scipy as sci
from scipy.optimize import OptimizeResult, approx_fprime, minimize

epsilon = np.sqrt(np.finfo(float).eps)

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}


def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper





def approx_fhess(x0, fun, *args, epsilon=epsilon, f1=None):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum)
    """
    # Calculate the first derivative
    if f1 is None:
        f1 = approx_fprime(x0, fun, epsilon, *args)
    print(f1)
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = np.zeros((n, n))



    # The next loop fills in the matrix
    N = x.size
    h = np.zeros((N, N))
    df_0 = approx_fprime(x, fun, epsilon, *args)
    print(df_0)
    for i in range(n):
        xx0 = 1.*x[i]
        x[i] = xx0 + epsilon
        df_1 = approx_fprime(x, fun, epsilon, *args)
        print(df_1)
        h[i,:] = (df_1 - df_0)/epsilon
        x[i] = xx0
    return h


def gradient_descent(fun, x0, args=(), jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, disp=False, gtol=1e-5, norm=np.inf, maxiter=None, return_all=False, eps=None, alpha=0.001):
    """
    Perform simple gradient descent with a fixed leraning rate.

    Options
    -------
    fun : callable
        The function to be minimized.
    x0 : numpy.array
        The variable to minimize with respect to.
    args : tuple
        Further function arguments.
    jac : bool or callable, optional
        Jacobian (gradient) of objective function. If jac is a Boolean and is
        True, fun is assumed to return the gradient along with the objective
        function. If False, the gradient will be estimated numerically. jac can
        also be a callable returning the gradient of the objective. In this
        case, it must accept the same arguments as fun.
    hess, hessp : callable, optional
        Hessian (matrix of second-order derivatives) of objective function or
        Hessian of objective function times an arbitrary vector p.  If hess is
        provided, then hessp will be ignored. If neither hess nor hessp is
        provided, then the Hessian product will be approximated using finite
        differences on jac. hessp must compute the Hessian times an arbitrary
        vector. Not needed for gradient descent.
    bounds : sequence, optional
        Bounds for variables (min, max) pairs for each element in x, defining
        the bounds on that parameter. Use None for one of min or max when
        there is no bound in that direction. Not needed for gradient descent.
    constraints : dict or sequence of dict, optional
        Constraints definition. Each constraint is defined in a dictionary with fields:
            type : str
                Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of fun.
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.
    Equality constraint means that the constraint function result is to be zero
    whereas inequality means that it is to be non-negative. Not needed for gradient descent.
    tol : float, optional
        Tolerance for termination. For detailed control,
        use solver-specific options.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the current
        parameter vector.
    options : mapping
        A dictionary of solver options.

    Options
    -------
    disp : bool, optional
        Set to True to print convergence messages.
    gtol : float, optional
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min).
    maxiter : int, optional
        Maximum number of iterations to perform.
    return_all : bool, optional
        If True, return values of the parameter vector after every iteration.
    eps : float or ndarray, optional
        If `jac` is approximated, use this value for the step size.
    alpha : float, optional
        The lerning rate.

    Returns
    -------
    result : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution array, success a Boolean flag
        indicating if the optimizer exited successfully and message which
        describes the cause of the termination. See OptimizeResult for a
        description of other attributes.
    """

    x0 = np.asarray(x0).flatten()

    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)
    if maxiter is None:
        maxiter = len(x0) * 10e100

    func_calls, fun = wrap_function(fun, args)
    if jac is None:
        grad_calls, myjac = wrap_function(approx_fprime, (fun, eps))
    else:
        grad_calls, myjac = wrap_function(jac, args)

    gft = myjac(x0)
    xt = x0
    t = 0

    if return_all:
        allvecs = [xt]
    warnflag = 0
    gnorm = np.linalg.norm(gft, ord=norm)

    while (gnorm > gtol) and (t < maxiter):
        xt -= alpha * gft
        if return_all:
            allvecs.append(xt)

        if callback is not None:
            callback(xt)

        gft= myjac(xt)
        gnorm = np.linalg.norm(gft, ord=norm)
        t += 1

    if t >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fun(xt))
            print("         Iterations: %d" % t)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fun(xt))
            print("         Iterations: %d" % t)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    result = OptimizeResult(fun=fun(xt), jac=gft, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xt,
                            nit=t)
    if return_all:
        result['allvecs'] = allvecs

    return result


def adam(fun, x0, args=(), jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, disp=False, gtol=1e-5, norm=np.inf, maxiter=None, eps=None, return_all=False, alpha=0.001, beta1=0.9, beta2=0.999, tau=1e-8):
    """
    Perform minimization using ADAM. For more details see [1].

    Options
    -------
    fun : callable
        The function to be minimized.
    x0 : numpy.array
        The variable to minimize with respect to.
    args : tuple
        Further function arguments.
    jac : bool or callable, optional
        Jacobian (gradient) of objective function. If jac is a Boolean and is
        True, fun is assumed to return the gradient along with the objective
        function. If False, the gradient will be estimated numerically. jac can
        also be a callable returning the gradient of the objective. In this
        case, it must accept the same arguments as fun.
    hess, hessp : callable, optional
        Hessian (matrix of second-order derivatives) of objective function or
        Hessian of objective function times an arbitrary vector p.  If hess is
        provided, then hessp will be ignored. If neither hess nor hessp is
        provided, then the Hessian product will be approximated using finite
        differences on jac. hessp must compute the Hessian times an arbitrary
        vector. Not needed for adam.
    bounds : sequence, optional
        Bounds for variables (min, max) pairs for each element in x, defining
        the bounds on that parameter. Use None for one of min or max when
        there is no bound in that direction. Not needed for adam.
    constraints : dict or sequence of dict, optional
        Constraints definition. Each constraint is defined in a dictionary with fields:
            type : str
                Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of fun.
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.
    Equality constraint means that the constraint function result is to be zero
    whereas inequality means that it is to be non-negative. Not needed for adam.
    tol : float, optional
        Tolerance for termination. For detailed control,
        use solver-specific options.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the current
        parameter vector.
    options : mapping
        A dictionary of solver options.

    Options
    -------
    disp : bool, optional
        Set to True to print convergence messages.
    gtol : float, optional
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min).
    maxiter : int, optional
        Maximum number of iterations to perform.
    return_all : bool, optional
        If True, return values of the parameter vector after every iteration.
    eps : float or ndarray, optional
        If `jac` is approximated, use this value for the step size.
    alpha : float, optional
        The lerning rate.
    beta1 : float, optional
        The beta1 parameter.
    beta2 : float, optional
        The beta2 parameter.
    tau=1e-8 : float, optinal.
        The tau parameter. Originally named epsilon.

    Returns
    -------
    result : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution array, success a Boolean flag
        indicating if the optimizer exited successfully and message which
        describes the cause of the termination. See OptimizeResult for a
        description of other attributes.


    References
    ----------
    .. [1] Diederik Kingma, Jimmy Ba, "Adam: A Method for Stochastic
        Optimization", 2014
    """

    x0 = np.asarray(x0).flatten()

    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)
    if maxiter is None:
        maxiter = len(x0) * 10e100

    func_calls, fun = wrap_function(fun, args)
    if jac is None:
        grad_calls, myjac = wrap_function(approx_fprime, (fun, eps))
    else:
        grad_calls, myjac = wrap_function(jac, args)

    gft = myjac(x0)
    xt = x0
    mt = np.mean(xt)
    vt = np.var(xt)
    t = 1

    if return_all:
        allvecs = [xt]
    warnflag = 0
    gnorm = np.linalg.norm(gft, ord=norm)

    while (gnorm > gtol) and (t < maxiter):

        mt = beta1 * mt + (1.0 - beta1) * gft
        vt = beta2 * vt + (1.0 - beta2) * gft**2
        mt_ = mt / (1.0 - beta1**t)
        vt_ = vt / (1.0 - beta2**t)
        xt -= alpha * mt_ / (np.sqrt(vt_) + tau)
        if return_all:
            allvecs.append(xt)

        if callback is not None:
            callback(xt)

        gft = myjac(xt)
        gnorm = np.linalg.norm(gft, ord=norm)
        t += 1

    if t >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fun(xt))
            print("         Iterations: %d" % t)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fun(xt))
            print("         Iterations: %d" % t)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    result = OptimizeResult(fun=fun(xt), jac=gft, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xt,
                            nit=t)
    if return_all:
        result['allvecs'] = allvecs

    return result


def newton_raphson(fun, x0, args=(), jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, maxiter=None, eps=1e-8, disp=False, return_all=False, gtol=1e-5, norm=np.inf):
    """
    Perform minimization with newton raphson.

    Options
    -------
    fun : callable
        The function to be minimized.
    x0 : numpy.array
        The variable to minimize with respect to.
    args : tuple
        Further function arguments.
    jac : bool or callable, optional
        Jacobian (gradient) of objective function. If jac is a Boolean and is
        True, fun is assumed to return the gradient along with the objective
        function. If False, the gradient will be estimated numerically. jac can
        also be a callable returning the gradient of the objective. In this
        case, it must accept the same arguments as fun.
    hess, hessp : callable, optional
        Hessian (matrix of second-order derivatives) of objective function or
        Hessian of objective function times an arbitrary vector p.  If hess is
        provided, then hessp will be ignored. If neither hess nor hessp is
        provided, then the Hessian product will be approximated using finite
        differences on jac. hessp must compute the Hessian times an arbitrary
        vector. At the moment the hessianp is not taken into account.
    bounds : sequence, optional
        Bounds for variables (min, max) pairs for each element in x, defining
        the bounds on that parameter. Use None for one of min or max when
        there is no bound in that direction. Not needed for newton-raphson.
    constraints : dict or sequence of dict, optional
        Constraints definition. Each constraint is defined in a dictionary with fields:
            type : str
                Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of fun.
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.
    Equality constraint means that the constraint function result is to be zero
    whereas inequality means that it is to be non-negative. Not needed for
    newton-raphson.
    tol : float, optional
        Tolerance for termination. For detailed control,
        use solver-specific options.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the current
        parameter vector.
    options : mapping
        A dictionary of solver options.

    Options
    -------
    disp : bool, optional
        Set to True to print convergence messages.
    gtol : float, optional
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min).
    maxiter : int, optional
        Maximum number of iterations to perform.
    return_all : bool, optional
        If True, return values of the parameter vector after every iteration.
    eps : float or ndarray, optional
        If `jac` is approximated, use this value for the step size.
    alpha : float, optional
        The lerning rate.

    Returns
    -------
    result : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution array, success a Boolean flag
        indicating if the optimizer exited successfully and message which
        describes the cause of the termination. See OptimizeResult for a
        description of other attributes.
    """
    x0 = np.asarray(x0).flatten()

    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)
    if maxiter is None:
        maxiter = len(x0) * 10e100

    func_calls, fun = wrap_function(fun, args)
    if jac is None:
        grad_calls, myjac = wrap_function(approx_fprime, (fun, eps))
    else:
        grad_calls, myjac = wrap_function(jac, args)

    if hess is None:
        hess_calls, myhess = wrap_function(approx_fhess, (myjac, eps))
    else:
        hess_calls, myhess = wrap_function(hess, args)

    gft = myjac(x0)
    hft = myhess(x0)
    hft_LL = sci.linalg.cho_factor(hft)
    xt = x0
    t = 1

    if return_all:
        allvecs = [xt]
    warnflag = 0
    gnorm = np.linalg.norm(gft, ord=norm)

    while (gnorm > gtol) and (t < maxiter):

        xt -= sci.linalg.cho_solve(hft_LL,gft)
        if return_all:
            allvecs.append(xt)

        if callback is not None:
            callback(xt)

        gft = myjac(xt)
        hft = myhess(xt)
        hft_LL = sci.linalg.cho_factor(hft)

        gnorm = np.linalg.norm(gft, ord=norm)
        t += 1

    if t >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fun(xt))
            print("         Iterations: %d" % t)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])
            print("         Hessian evaluations: %d" % hess_calls[0])

    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fun(xt))
            print("         Iterations: %d" % t)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])
            print("         Hessian evaluations: %d" % hess_calls[0])


    result = OptimizeResult(fun=fun(xt), jac=gft, nfev=func_calls[0],
                            njev=grad_calls[0], nhev=hess_calls[0],status=warnflag,
                            success=(warnflag == 0), message=msg, x=xt,
                            nit=t)
    if return_all:
        result['allvecs'] = allvecs

    return result




if __name__ == "__main__":

    # A test function
    def powell(x):
        return (x[0]+10.0*x[1])**2 + 5.0*(x[2]-x[3])**2 + (x[1]-2.0*x[2])**4 + 10.0*(x[0]-x[3])**4

    def powellJac(x):
        return np.array([2.0*(x[0]+10.0*x[1])+40.0*(x[0]-x[3])**3, 20.0*(x[0]+10.0*x[1])+4.0*(x[1]-2.0*x[2])**3, 10.0*(x[2]-x[3])-8.0*(x[1]-2.0*x[2])**3, -10.0*(x[2]-x[3])-40.0*(x[0]-x[3])**3])

    def powellHess(x):
        return np.array([[2+120*(x[0]-x[3])**2, 20, 0, -120*(x[0]-x[3])**2],
                         [20, 200+12*(x[1]-2*x[2])**2, -24*(x[1]-2*x[2])**2, 0],
                         [0, -24*(x[1]-2*x[2])**2,10+48*(x[1]-2*x[2])**2, -10],
                         [-120*(x[0]-x[3])**2, 0, -10, 10+120*(x[0]-x[3])**2]])

    x = np.array([3, -1, 0, 1])
    #print(approx_fhess(np.array([3, -1, 0, 1]), powell, epsilon=np.sqrt(np.finfo(float).eps)))
    #print(powellHess(np.array([3, -1, 0, 1])))

    print(powell(x))

    print(powellHess(x))

    print(approx_fhess(x, powell))


    #print(minimize(powell, np.array([3, -1, 0, 1]), method=gradient_descent, jac=None))
    #print(minimize(powell, np.array([3, -1, 0, 1]), method=adam, jac=powellJac))
    #print(minimize(powell, np.array([3,-1,0,1]), method=newton_raphson, jac=powellJac, hess=powellHess))
    #print(minimize(powell, np.array([3, -1, 0, 1]), method=newton_raphson, jac=powellJac, hess=None))
