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


__all__ = ["make_scorer","get_scorer","accuracy_score","average_precision_score","f1_score","fbeta_score","negative_predictive_value","positive_predictive_value","precision_score","precision_recall_curve" "roc_auc_score","roc_curve","true_positive_rate","true_negative_rate"]


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
    Note: this implementation is restricted to the binary classification task.

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

def average_precision_score(z_true, z_score, y_true=None, y_score=None):
    """
    Compute average precision (AP) from prediction scores
    This score corresponds to the area under the precision-recall curve.
    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
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

    Returns
    -------
    average_precision_z : float
        Average precision score for the bags.
    average_precision_Y : float
        Average precision score for the instances.
    """
    ap_z = skl.metrics.average_precision_score(z_true, z_score, average="macro", sample_weight=None)
    if y_true is not None and y_score is not None:
        ap_y = skl.metrics.average_precision_score(y_true, y_score, average="macro",
            sample_weight=None)
    else:
        ap_y = None
    return ap_z, ap_y

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
    Note: this implementation is restricted to the binary classification task.

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
    Note: this implementation is restricted to the binary classification task.

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
       F-beta score for the positive bags.
   fbeta_score_y : float
       F-beta score for the positive instances.
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
    Note: this implementation is restricted to the binary classification task.

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
    Note: this implementation is restricted to the binary classification task.

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

def precision_recall_curve(z_true, z_score, y_true=None, y_score=None, pos_label=1):
    """
    Compute precision-recall pairs for different probability thresholds
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold.  This ensures that the graph starts on the
    x axis.
    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.  If labels are not
        binary, pos_label should be explicitly given.
    z_score: 1d array-like
        Target scores for the bags, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.  If labels are not
        binary, pos_label should be explicitly given.
    y_score: 1d array-like
        Target scores for the instances, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    pos_label : str or int
        Label considered as positive and others are considered negative.

    Returns
    -------
    precision_z : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall_z : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds_z : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    precision_y : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall_y : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds_y : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    pre_z, rec_z, thresholds_z = skl.metrics.precison_recall_curve(z_true, z_score,
        pos_label=pos_label, sample_weight=None)
    if y_true is not None and y_score is not None:
        pre_y, rec_y, thresholds_y = skl.metrics.precison_recall_curve(z_true, z_score,
            pos_label=pos_label, sample_weight=None)
    else:
        fpre_y, rec_y, thresholds_y = None
    return pre_z, rec_z, thresholds_z, pre_y, rec_y, thresholds_y

def precision_score(z_true, z_pred, y_true=None, y_pred=None, pos_label=1):
    """Compute the precision
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The best value is 1 and the worst value is 0.
    Note: this implementation is restricted to the binary classification task.

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
    pos_label : str or int, 1 by default
        The class to report.

    Returns
    -------
    precision_z : float
        Precision of the positive bags.
    precision_y : float
        Precision of the positive instances.
    """
    pre_z = skl.metrics.precision_score(z_true, z_pred, labels=None, pos_label=pos_label, average='binary', sample_weight=None)
    if y_true is not None and y_pred is not None:
        pre_y = skl.metrics.precision_score(y_true, y_pred, labels=None, pos_label=pos_label, average='binary', sample_weight=None)
    else:
        pre_y = None
    return pre_z, pre_y

def recall_score(z_true, z_pred, y_true=None, y_pred=None, pos_label=1):
    """
    Compute the recall
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    Note: this implementation is restricted to the binary classification task.

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
    pos_label : str or int, 1 by default
        The class to report.

    Returns
    -------
    recall_z : float
        Recall of the positive class for the bags.
    recall_y : float
        Recall of the positive class for the instances.
    """
    rec_z = skl.metrics.precision_score(z_true, z_pred, labels=None, pos_label=pos_label, average='binary', sample_weight=None)
    if y_true is not None and y_pred is not None:
        rec_y = skl.metrics.recall_score(y_true, y_pred, labels=None, pos_label=pos_label, average='binary', sample_weight=None)
    else:
        rec_y = None
    return rec_z, rec_y

def roc_auc_score(z_true, z_score, y_true=None, y_score=None):
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

    Returns:
    --------
    auc_z : float
        ROC auc for the bags.
    auc_y : float
        ROC auc for the instances.
    """
    auc_z = skl.metrics.roc_auc_score(z_true, z_score, average="macro", sample_weight=None)
    if y_true is not None and y_score is not None:
        auc_y = skl.metrics.roc_auc_score(y_true, y_score, average="macro", sample_weight=None)
    else:
        auc_y = None
    return auc_z, auc_y

def roc_curve(z_true, z_score, y_true=None, y_score=None, pos_label=1, drop_intermediate=True):
    """
    Compute Receiver operating characteristic (ROC)
    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    z_true : 1d array-like
        Ground truth (correct) labels for the bags.  If labels are not
        binary, pos_label should be explicitly given.
    z_score: 1d array-like
        Target scores for the bags, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    y_true : 1d array-like
        Ground truth (correct) labels for the instances.  If labels are not
        binary, pos_label should be explicitly given.
    y_score: 1d array-like
        Target scores for the instances, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    pos_label : str or int
        Label considered as positive and others are considered negative.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

    Returns
    -------
    fpr_z : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
    tpr_z : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
    thresholds_z : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no bags being predicted
        and is arbitrarily set to `max(y_score) + 1`.
    fpr_y : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
    tpr_y : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
    thresholds_y : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.
    """
    fpr_z, tpr_z, thresholds_z  = skl.metrics.roc_curve(z_true, z_score, pos_label=pos_label, sample_weight=None, drop_intermediate=drop_intermediate)
    if y_true is not None and y_score is not None:
        fpr_y, tpr_y, thresholds_y = skl.metrics.roc_curve(z_true, z_score, pos_label=pos_label, sample_weight=None, drop_intermediate=drop_intermediate)
    else:
        fpr_y = tpr_y = thresholds_y = None
    return fpr_z, tpr_z, thresholds_z, fpr_y, tpr_y, thresholds_y

def true_positive_rate(z_true, z_pred, y_true=None, y_pred=None, pos_label=1):
    """
    Calculate true positive rate (sensitivity, recall).
    True positive rate = 1 - false negative rate
    Note: this implementation is restricted to the binary classification task.

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
    Note: this implementation is restricted to the binary classification task.

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

def get_scorer(scoring):
    """
    Get a scorer named after the standard scores defined in here using default
    arguments.

    Parameters
    ----------
    scoring : str
        The name of the scorere

    Return
    ------
    scorer : scorer
        The requested scorer

    """
    if isinstance(scoring, str):
        try:
            scorer = SCORERS[scoring]
        except KeyError:
            scorers = [scorer for scorer in SCORERS
                       if SCORERS[scorer]._deprecation_msg is None]
            raise ValueError('%r is not a valid scoring value. '
                             'Valid options are %s'
                             % (scoring, sorted(scorers)))
    else:
        scorer = scoring
    return scorer

class _PredictScorer():

    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign


    def __call__(self, estimator, data):
        """
        Evaluate predicted target values for milData relative to ground truth.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        data : milData object
            Test data that will be fed to estimator.predict.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on milData.
        """
        z_pred, y_pred = estimator.predict(data)
        return self._sign * self._score_func(data.z, z_pred, data.y, y_pred, **self._kwargs)


class _ProbaScorer():

    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign


    def __call__(self, estimator, data):
        """
        Evaluate predicted target values for milData relative to ground truth.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        data : milData object
            Test data that will be fed to estimator.predict.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on milData.
        """
        z_pred, y_pred = estimator.predict_proba(data)
        return self._sign * self._score_func(data.z, z_pred, data.y, y_pred, **self._kwargs)


class _ThresholdScorer():

    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign


    def __call__(self, estimator, data):
        """
        Evaluate predicted target values for milData relative to ground truth.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        data : milData object
            Test data that will be fed to estimator.predict.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on milData.
        """
        z_pred, y_pred = estimator.decision_function(data)
        return self._sign * self._score_func(data.z, z_pred, data.y, y_pred, **self._kwargs)


# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score)
fbeta_scorer = make_scorer(f1_score)
negative_predictive_scorer = make_scorer(negative_predictive_value)
positive_predictive_scorer = make_scorer(positive_predictive_value)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)
true_positive_scorer = make_scorer(true_positive_rate)
true_negative_scorer = make_scorer(true_negative_rate)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score, needs_threshold=True)


SCORERS = dict(accuracy_score=accuracy_scorer,
    average_precision_score=average_precision_scorer,
    f1_score=f1_score,
    fbeta_score=fbeta_scorer,
    negative_predictive_value=negative_predictive_scorer,
    positive_predictive_value=positive_predictive_scorer,
    precision_score=precision_scorer,
    recall_score=recall_scorer,
    roc_auc_score=roc_auc_scorer, true_positive_rate=true_positive_scorer,
    true_negative_rate=true_negative_scorer)


