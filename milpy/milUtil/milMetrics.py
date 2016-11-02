#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Metrics module for MIL datasets. The API is tried to keep consistent with
sklearn. Several parts of the code are also copied directly from sklearn and are
therefore released under their license.

:author: Manuel Tuschen
:date: 04.07.2016
:license: BSD
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import numpy as np
import sklearn as skl
from sklearn import metrics


__all__ = ["accuracy_score","f1_score","fbeta_score","negative_predictive_value","positive_predictive_value","roc_auc_score","true_positive_rate","true_negative_rate"]


################################################################################
#                                                                              #
#                       Classification metrics                                 #
#                                                                              #
################################################################################

# We start with those scores relevant for binary class prediction


def accuracy_score(z_true, z_pred, y_true=None, y_pred=None):
    """
    Accuracy classification score for bag level and optionally for instance
    level.

    Parameters:
    -----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.
    z_pred : 1d array-like
         Predicted labels, as returned by a classifier for the bags.
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.
    y_pred : 1d array-like
       Predicted labels, as returned by a classifier for the instances.

    Returns:
    --------
    score_z : float
       The correctly classified samples for the bags. The best performance is 1.
    score_y : float
       The correctly classified samples for the instances. The best performance
       is 1.
    """
    score_z = skl.metrics.accuracy_score(z_true, z_pred, normalize=True, sample_weight=None)
    if y_true is not None and y_pred is not None:
        score_y = skl.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    else:
        score_y = None
    return score_z, score_y

def f1_score(z_true, z_pred, y_true=None, y_pred=None, pos_label=1):
    """
    Compute the F1 score, also known as balanced F-score or F-measure
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    In the multi-class and multi-label case, this is the weighted average of
    the F1 score of each class.

    Parameters:
    ----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.
    z_pred : 1d array-like
         Predicted labels, as returned by a classifier for the bags.
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.
    y_pred : 1d array-like
       Predicted labels, as returned by a classifier for the instances.
    pos_label : str or int, 1 by default
        The class to report.

    Returns:
    --------
    f1_score_z : float
        F1 score of the positive bags.
    f1_score_y : float
        F1 score for the positive instances.
    """
    f1_score_z = skl.metrics.f1_score(z_true, z_pred, labels=None, pos_label=pos_label, average='binary', sample_weight=None)
    if y_true is not None and y_pred is not None:
        f1_score_y = skl.metrics.f1_score(y_true, y_pred, labels=None, pos_label=pos_label, average='binary', sample_weight=None)
    else:
        f1_score_y = None
    return f1_score_z, f1_score_y

def fbeta_score(z_true, z_pred, y_true=None, beta=1, y_pred=None, pos_label=1):
    """
    Compute the F-beta score
    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.
    The `beta` parameter determines the weight of precision in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> inf``
    only recall).

    Parameters
    ----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.
    z_pred : 1d array-like
        Predicted labels, as returned by a classifier for the bags.
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.
    y_pred : 1d array-like
        Predicted labels, as returned by a classifier for the instances.
    beta: float, optional
        Weight of precision in harmonic mean.
    pos_label : str or int, 1 by default
        The class to report.

   Returns
   -------
   fbeta_score_z : float
       F-beta score of the positive class for the
       bags.
   fbeta_score_y : float
       F-beta score of the positive class for the
       instances.
    """

    fbeta_score_z = skl.metrics.fbeta_score(z_true, z_pred, beta=beta, labels=None, pos_label=pos_label, average='binary', sample_weight=None)
    if y_true is not None and y_pred is not None:
        fbeta_score_y = skl.metrics.fbeta_score(y_true, y_pred, beta=beta, labels=None, pos_label=pos_label, average='binary', sample_weight=None)
    else:
        fbeta_score_y = None
    return fbeta_score_z, fbeta_score_y

def negative_predictive_value(z_true, z_pred, y_true=None, y_pred=None, pos_label=1):
    """
    Calculate negative predictive value (negative equivalent to precision).

    Parameters:
    ----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.
    z_pred : 1d array-like
         Predicted labels, as returned by a classifier for the bags.
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.
    y_pred : 1d array-like
       Predicted labels, as returned by a classifier for the instances.
    pos_label : str or int, 1 by default
        The class to report.

    Return:
    -------
    npv_z: float
        The negative predictive value for the bags.
    npv_y: float
        The negative predictive value for the instances.
    """
    z_true1 = np.array(z_true) == pos_label
    z_true0 = np.array(z_true) != pos_label
    z_pred1 = np.array(z_pred) == pos_label
    z_pred0 = np.array(z_pred) != pos_label

    # True positive count and True negative count
    tp = np.sum(np.logical_and(z_pred1, z_true1))  # correct positives
    tn = np.sum(np.logical_and(z_pred0, z_true0))  # correct negatives

    # False positive count and False negative count for calculation of npv
    fp = np.sum(np.logical_and(z_pred1, z_true0))  # false positives
    fn = np.sum(np.logical_and(z_pred0, z_true1))  # false negatives

    if (tn + fn) == 0:
        npv_z = np.nan
    else:
        npv_z = tn / (tn + fn)

    if y_true is not None and y_pred is not None:
        y_true1 = np.array(y_true) == pos_label
        y_true0 = np.array(y_true) != pos_label
        y_pred1 = np.array(y_pred) == pos_label
        y_pred0 = np.array(y_pred) != pos_label

        # True positive count and True negative count
        tp = np.sum(np.logical_and(y_pred1, y_true1))  # correct positives
        tn = np.sum(np.logical_and(y_pred0, y_true0))  # correct negatives

        # False positive count and False negative count for calculation of npv
        fp = np.sum(np.logical_and(y_pred1, y_true0))  # false positives
        fn = np.sum(np.logical_and(y_pred0, y_true1))  # false negatives

        if (tn + fn) == 0:
            npv_y = np.nan
        else:
            npv_y = tn / (tn + fn)
    else:
        npv_y = None

    return npv_z, npv_y

def positive_predictive_value(z_true, z_pred, y_true=None, y_pred=None, pos_label=1):
    """
    Calculate positive predictive value (precision)

    Parameters:
    ----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.
    z_pred : 1d array-like
         Predicted labels, as returned by a classifier for the bags.
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.
    y_pred : 1d array-like
       Predicted labels, as returned by a classifier for the instances.
    pos_label : str or int, 1 by default
        The class to report.

    Return:
    ------
    ppv_z : float
        The positive predictive value for the bags.
    ppv_y : float
        The positive predictive value for the instances.
    """

    z_true1 = np.array(z_true) == pos_label
    z_true0 = np.array(z_true) != pos_label
    z_pred1 = np.array(z_pred) == pos_label
    z_pred0 = np.array(z_pred) != pos_label

    # True positive count and True negative count
    tp = np.sum(np.logical_and(z_pred1, z_true1))  # correct positives
    tn = np.sum(np.logical_and(z_pred0, z_true0))  # correct negatives

    # False positive count and False negative count for calculation of npv
    fp = np.sum(np.logical_and(z_pred1, z_true0))  # false positives
    fn = np.sum(np.logical_and(z_pred0, z_true1))  # false negatives

    if (tp + fp) == 0:
        ppv_z = np.nan
    else:
        ppv_z = tp / (tp + fp)

    if y_true is not None and y_pred is not None:
        y_true1 = np.array(y_true) == pos_label
        y_true0 = np.array(y_true) != pos_label
        y_pred1 = np.array(y_pred) == pos_label
        y_pred0 = np.array(y_pred) != pos_label

        # True positive count and True negative count
        tp = np.sum(np.logical_and(y_pred1, y_true1))  # correct positives
        tn = np.sum(np.logical_and(y_pred0, y_true0))  # correct negatives

        # False positive count and False negative count for calculation of npv
        fp = np.sum(np.logical_and(y_pred1, y_true0))  # false positives
        fn = np.sum(np.logical_and(y_pred0, y_true1))  # false negatives

        if (tp + fp) == 0:
            ppv_y = np.nan
        else:
            ppv_y = tp / (tp + fp)
    else:
        ppv_y = None

    return ppv_z, ppv_y

def roc_auc_score(z_true, z_score, y_true=None, y_score=None, average="macro"):
    """
    Compute Area Under the Curve (AUC) from prediction scores
    Note: this implementation is restricted to the binary classification task

    Parameters:
    -----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.
    z_score : 1d array-like
        Bag target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.
    y_score : 1d array-like
        Instance target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    average : string, [None, 'micro', 'macro' (default), 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:
        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

    Returns:
    --------
    auc_z : float
    auc_y : float
    """
    auc_z = skl.metrics.roc_auc_score(z_true, z_score, average=average, sample_weight=None)
    if y_true is not None and y_score is not None:
        auc_y = skl.metrics.roc_auc_score(y_true, y_score, average=average,
            sample_weight=None)
    else:
        auc_y = None
    return auc_z, auc_y

def true_positive_rate(z_true, z_pred, y_true=None, y_pred=None, pos_label=1):
    """
    Calculate true positive rate (sensitivity, recall).
    True positive rate = 1 - false negative rate

    Parameters:
    -----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.
    z_pred : 1d array-like
         Predicted labels, as returned by a classifier for the bags.
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.
    y_pred : 1d array-like
       Predicted labels, as returned by a classifier for the instances.
    pos_label : str or int, 1 by default
        The class to report.

    Return:
    ------
    tpr_z : float
        The true positive rate for the bags.
    tpr_y : float
        The true positive rate for the instances.
    """
    z_true1 = np.array(z_true) == pos_label
    z_true0 = np.array(z_true) != pos_label
    z_pred1 = np.array(z_pred) == pos_label
    z_pred0 = np.array(z_pred) != pos_label

    # True positive count and True negative count
    tp = np.sum(np.logical_and(z_pred1, z_true1))  # correct positives
    tn = np.sum(np.logical_and(z_pred0, z_true0))  # correct negatives

    # False positive count and False negative count for calculation of npv
    fp = np.sum(np.logical_and(z_pred1, z_true0))  # false positives
    fn = np.sum(np.logical_and(z_pred0, z_true1))  # false negatives

    if tp + fn == 0:
        tpr_z = np.nan
    else:
        tpr_z = tp / (tp + fn)

    if y_true is not None and y_pred is not None:
        y_true1 = np.array(y_true) == pos_label
        y_true0 = np.array(y_true) != pos_label
        y_pred1 = np.array(y_pred) == pos_label
        y_pred0 = np.array(y_pred) != pos_label

        # True positive count and True negative count
        tp = np.sum(np.logical_and(y_pred1, y_true1))  # correct positives
        tn = np.sum(np.logical_and(y_pred0, y_true0))  # correct negatives

        # False positive count and False negative count for calculation of npv
        fp = np.sum(np.logical_and(y_pred1, y_true0))  # false positives
        fn = np.sum(np.logical_and(y_pred0, y_true1))  # false negatives

        if tp + fn == 0:
            tpr_y = np.nan
        else:
            tpr_y = tp / (tp + fn)
    else:
        tpr_y = None

    return tpr_z, tpr_y

def true_negative_rate(z_true, z_pred, y_true=None, y_pred=None, pos_label=1):
    """
    Calculate true negative rate.
    True negative rate = 1 - false positive rate

    Parameters:
    ----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.
    z_pred : 1d array-like
         Predicted labels, as returned by a classifier for the bags.
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.
    y_pred : 1d array-like
       Predicted labels, as returned by a classifier for the instances.
    pos_label : str or int, 1 by default
        The class to report.

    Return:
    ------
    tnr_z : float
        The true negative rate for the bags.
    tnr_y : float
        The true negative rate for the instances.
    """
    z_true1 = np.array(z_true) == pos_label
    z_true0 = np.array(z_true) != pos_label
    z_pred1 = np.array(z_pred) == pos_label
    z_pred0 = np.array(z_pred) != pos_label

    # True positive count and True negative count
    tp = np.sum(np.logical_and(z_pred1, z_true1))  # correct positives
    tn = np.sum(np.logical_and(z_pred0, z_true0))  # correct negatives

    # False positive count and False negative count for calculation of npv
    fp = np.sum(np.logical_and(z_pred1, z_true0))  # false positives
    fn = np.sum(np.logical_and(z_pred0, z_true1))  # false negatives

    if (tn + fp) == 0:
        tnr_z = np.nan
    else:
        tnr_z = tn / (tn + fp)


    if y_true is not None and y_pred is not None:
        y_true1 = np.array(y_true) == pos_label
        y_true0 = np.array(y_true) != pos_label
        y_pred1 = np.array(y_pred) == pos_label
        y_pred0 = np.array(y_pred) != pos_label

        # True positive count and True negative count
        tp = np.sum(np.logical_and(y_pred1, y_true1))  # correct positives
        tn = np.sum(np.logical_and(y_pred0, y_true0))  # correct negatives

        # False positive count and False negative count for calculation of npv
        fp = np.sum(np.logical_and(y_pred1, y_true0))  # false positives
        fn = np.sum(np.logical_and(y_pred0, y_true1))  # false negatives

        if (tn + fp) == 0:
            tnr_y = np.nan
        else:
            tnr_y = tn / (tn + fp)

    else:
        tnr_y = None

    return tnr_z, tnr_y


################################################################################
#                                                                              #
#                                   Scorere                                    #
#                                                                              #
################################################################################

# We need a classification scorerer as in skleran for cross-validation

def make_scorer(score_func, greater_is_better=True, needs_proba=False,
        needs_threshold=False, **kwargs):
    """
    Make a scorer from a performance metric or loss function.
    This factory function wraps scoring functions for use in GridSearchCV
    and cross_val_score. It takes a score function, such as ``accuracy_score``,
    or ``average_precision`` and returns a callable that scores an estimator's
    output.

    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(z_true, z_pred, y_true, y_pred, **kwargs)``.
    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.
    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.
        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True,"
                         " but not both.")
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)

class _PredictScorer():

    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign


    def __call__(self, estimator, milData):
        """
        Evaluate predicted target values for milData relative to ground truth.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        milData : milData object
            Test data that will be fed to estimator.predict.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on milData.
        """
        z_pred, y_pred = estimator.predict(milData)
        return self._sign * self._score_func(milData.z, z_pred, milData.y, y_pred, **self._kwargs)


class _ProbaScorer():

    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign


    def __call__(self, estimator, milData):
        """
        Evaluate predicted target values for milData relative to ground truth.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        milData : milData object
            Test data that will be fed to estimator.predict.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on milData.
        """
        z_pred, y_pred = estimator.predict_proba(milData)
        return self._sign * self._score_func(milData.z, z_pred, milData.y, y_pred, **self._kwargs)


class _ThresholdScorer():

    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign


    def __call__(self, estimator, milData):
        """
        Evaluate predicted target values for milData relative to ground truth.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        milData : milData object
            Test data that will be fed to estimator.predict.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on milData.
        """
        z_pred, y_pred = estimator.decision_function(milData)
        return self._sign * self._score_func(milData.z, z_pred, milData.y, y_pred, **self._kwargs)





