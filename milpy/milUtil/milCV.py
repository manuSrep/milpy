#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Model selection module for MIL datasets. API is tried to keep consistent with
sklearn. Several parts of the code are also copied directly from sklearn and are
therefore released under their license.

:author: Manuel Tuschen
:date: 04.07.2016
:license: BSD
"""

from __future__ import division, absolute_import, unicode_literals, print_function

from copy import deepcopy

import numpy as np
import sklearn as skl
from sklearn import model_selection
from joblib import Parallel, delayed

__all__ = ["KFold","train_test_split", "cross_val_predict"]


################################################################################
#                                                                              #
#                       Splitter Classes                                       #
#                                                                              #
################################################################################


class KFold():
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state


    def split(self, milData):
        """
        Generate keys to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_samples = milData.N_B

        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_mask = np.zeros(milData.N_B, dtype=np.bool)
            test_mask[indices[start:stop]] = True
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            yield np.array(milData.keys)[train_index], np.array(milData.keys)[test_index]
            current = stop


################################################################################
#                                                                              #
#                       Splitter Functions                                     #
#                                                                              #
################################################################################

def train_test_split(data, test_size=None, train_size=None, random_state=None):
    """
    Split a mil dataset into test and training set.

    Parameters
    ----------
    data : milData
        The dataset to split
    test_size : float, int, or None (default is None), optional
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is automatically
        set to the complement of the train size. If train size is also None,
        test size is set to 0.25.
    train_size : float, int, or None (default is None), optional
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the train split. If int, represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.
    random_state : int, optional
        Pseudo-random number generator state used for random sampling.

    Returns
    --------
    data_train : milData
        The training set.
    data_test : milData
        The test set.
    """
    keys_train, keys_test, _, _ = skl.model_selection.train_test_split(data.keys, data.keys, test_size=test_size, train_size=train_size, random_state=random_state)

    data_train = deepcopy(data)
    for key in keys_test:
        data_train.del_B(key)

    data_test = deepcopy(data)
    for key in keys_train:
        data_test.del_B(key)

    return data_train, data_test


################################################################################
#                                                                              #
#                       Model validation                                       #
#                                                                              #
################################################################################

def cross_val_score(estimator, milData, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs'):

    """
    Evaluate a score by cross-validation
    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    milData : milData object
        The data to fit.
    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : integer, optional
        The verbosity level.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score)(deepcopy(estimator), milData, scoring, train, test, verbose, None, fit_params) for train, test in cv.split(milData))
    return np.array(scores)[:, 0]


def cross_val_predict(estimator, milData, groups=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', method='predict'):
    """
    Generate cross-validated estimates for each input data point

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    milData : milData object
        The data to fit.
    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : integer, optional
        The verbosity level.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator.

    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``
    """
    # Ensure the estimator has implemented the passed decision function
    if not callable(getattr(estimator, method)):
        raise AttributeError('{} not implemented in estimator'.format(method))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(deepcopy(estimator), milData, train, test, verbose, fit_params, method)for train, test in cv.split(milData))

    # Concatenate the predictions
    predictions_z = [pred_block_z for pred_block_z, _, _ in prediction_blocks]
    predictions_y = [pred_block_y for _, pred_block_y, _ in prediction_blocks]

    test_keys = np.concatenate([key for _, _, key in prediction_blocks])
    test_indices = [list(test_keys).index(key) for key in milData.keys]
    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices), dtype=int)

    predictions_z = np.concatenate(predictions_z)
    if not None in predictions_y:
        predictions_y = np.concatenate(predictions_y)
        return predictions_z[inv_test_indices], predictions_y[inv_test_indices]
    else:
        return predictions_z[inv_test_indices], None


def _fit_and_score(estimator, milData, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score='raise'):
    """Fit estimator and compute scores for a given dataset split.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape at least 2D
        The data to fit.
    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    train : array-like, shape (n_train_samples,)
        Indices of training samples.
    test : array-like, shape (n_test_samples,)
        Indices of test samples.
    verbose : integer
        The verbosity level.
    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : boolean, optional, default: False
        Compute and return score on training set.
    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    Returns
    -------
    train_score : float, optional
        Score on training set, returned only if `return_train_score` is `True`.
    test_score : float
        Score on test set.
    n_test_samples : int
        Number of test samples.
    fit_time : float
        Time spent for fitting in seconds.
    score_time : float
        Time spent for scoring in seconds.
    parameters : dict or None, optional
        The parameters that have been evaluated.
    """

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, v) for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    data_train = _safe_split(milData, train)
    data_test = _safe_split(milData, test)

    estimator.fit(data_train, **fit_params)
    test_score = scorer(estimator, data_test)

    if return_train_score:
        train_score = scorer(estimator, data_train)

    ret = [train_score, test_score] if return_train_score else [test_score]
    return ret


def _fit_and_predict(estimator,milData, train, test, verbose, fit_params,
                     method):
    """
    Fit estimator and predict values for a given dataset split.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    milData : milData object
        The data to fit.
    train : array-like, shape (n_train_samples,)
        Indices of training samples.
    test : array-like, shape (n_test_samples,)
        Indices of test samples.
    verbose : integer
        The verbosity level.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    method : string
        Invokes the passed method name of the passed estimator.
    Returns
    -------
    predictions : sequence
        Result of calling 'estimator.method'
    test : array-like
        This is the value of the test parameter
    """
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, v) for k, v in fit_params.items()])

    data_train = _safe_split(milData, train)
    data_test = _safe_split(milData, test)

    estimator.fit(data_train, **fit_params)
    func = getattr(estimator, method)
    predictions_z, predictions_y = func(data_test)
    return predictions_z, predictions_y, test


def _safe_split(milData, keys):
    """
    Create milData subset.
    """
    new = deepcopy(milData)

    for key in milData.keys:
        if key not in keys:
            new.del_B(key)

    return new



'''
def corss_val_scores(y_true, y_pred, y_prob):
    """
   Caclulate scores from CV results.

    Parameters
    ----------
    y_true : ndaray
        The ground truth labels for each CV run. Must be 0 or 1.
    y_pred : ndaray
        The predicted labels for each CV run. Must be 0 or 1.
    y_prob : ndarray, optional
        The predicted probabilities for each CV run for class 1.

    Returns
    ----------
    scores : dictionary
        The calculated scores.
    """
    if y_prob is not None and np.size(y_prob) == 0:
        y_prob = None
        raise Warning("Empty probability array was given and will not be used.")


    scores = {"y_true":[], "y_pred":[], "y_prob":[], "accuracy":[], "tpr":[], "tnr":[], "ppv":[],  "npv":[], "precision":[], "recall":[], "sensitivity":[], "F1":[], "ROC_auc":[], "ROC":[], "precRec":[] }



    for i in range(len(y_true)):
        if len(y_true[i]) != len(y_pred[i]):
            raise ValueError("The number of predictions does not match the ground truth")


        y_true[i][y_true[i] <= 0] = 0
        y_true[i][y_true[i] > 0] = 1
        y_pred[i][y_pred[i] <= 0] = 0
        y_pred[i][y_pred[i] > 0] = 1

        # raw values
        scores["y_true"].append(y_true[i])
        scores["y_pred"].append(y_pred[i])
        if y_prob is not None:
            scores["y_prob"].append(y_prob[i])

        # Accuracy
        scores["accuracy"].append(accuracy_score(y_true[i], y_pred[i]))

        # True positive rate = sensitivity = recall
        # True positive rate = 1 - false negative rate
        scores["tpr"].append(true_positive_rate(y_true[i], y_pred[i]))
        scores["sensitivity"].append(true_positive_rate(y_true[i], y_pred[i]))
        scores["recall"].append(true_positive_rate(y_true[i], y_pred[i]))

        # True negative rate = specificity
        # True negative rate = 1 - false positive rate
        scores["tnr"].append(true_negative_rate(y_true[i], y_pred[i]))

        # positive predictive value = precision
        scores["precision"].append(positive_predictive_value(y_true[i], y_pred[i]))

        # Negative predictive value (negative equivalent to precision)
        scores["npv"].append(negative_predictive_value(y_true[i], y_pred[i]))

        # F1 Score
        scores["F1"].append(F1_score(y_true[i], y_pred[i]))

        if y_prob is not None:
            try:
                # ROC AUC score
                scores["ROC_auc"].append(roc_auc_score(y_true[i], y_prob[i]))
            except(ValueError, ZeroDivisionError):
                scores["ROC_auc"].append(np.nan)

            try:
                # Precision, Recall Curve
                pre, rec, thr = precision_recall_curve(y_true[i], y_prob[i][:,1], pos_label=1)
                scores["precRec"].append([pre, rec, thr])

                # ROC
                roc_fpr, roc_tpr, thr = roc_curve(y_true[i], y_prob[i][:,1], pos_label=1)
                scores["ROC"].append([roc_fpr, roc_tpr, thr])
            except:
                continue
    return scores

def cross_val_predict(estimator, data, k=10, n=1, random_state=None, fit_params={}, pred_params={}):
    """
    Function for n times k-fold cross-validation.

    Parameters
    ----------
    estimator : milEstimator
        The prediction class with a "fit" and a "predict" method.
    data : milData
        The mil dataset
    k : int, optional
        The number of CV to perform.
    n : int, optional
        The times k-fold CV will be performed on shuffled dataset.
    random_state : int or random state, optional
        The random state used to shuffle the datasets during n times CV.
    fit_params : mapping, optional
        The fit parameters for the fit method. Only if "probability":True is
        included, the prediction probabilities can be computed.
    pred_params : mapping, optional
        The prediction parameters for the fit method.

    Returns
    -------
    z_gt_CV : ndaray
        The true bag labels.
    z_pred_CV : ndaray
        The predicted bag labels. If prediction was not possible, the array
        will be empty.
    z_prob_CV : ndarray
        The predicted bag probabilities. If prediction was not enabled or
        possible, the array will be empty.
    y_gt_CV : ndaray
        The true instance labels. If prediction of y was not possible, the array
        will be empty.
    y_pred_CV : ndaray
        The predicted instance labels. If prediction was not possible, the array
        will be empty.
    y_prob_CV : ndarray
        The predicted instance probabilities. If prediction was not enabled or
        possible, the array will be empty.
    """
    # define  output parameters
    z_gt_CV = []
    z_pred_CV = []
    z_prob_CV = []
    y_gt_CV = []
    y_pred_CV = []
    y_prob_CV = []

    widgets = ['Cross Validation: ', Percentage(), ' ', Bar(), ' ', ETA(), ' ']
    progress = ProgressBar(widgets=widgets, maxval=n*k).start()
    prog_val = 0

    np.random.seed(random_state)
    keys_i = data.keys.copy()

    for i in range(n):
        np.random.shuffle(keys_i)  # we shuffle the keys

        #print(keys_i)

        CVindex = KFold(data.N_B, n_folds=k, shuffle=False)

        #print(CVindex)

        for train_index, test_index in CVindex:

            #print(train_index, test_index)

            #continue
            keys_train = np.array(keys_i)[train_index]
            keys_test = np.array(keys_i)[test_index]

            data_train = deepcopy(data)
            for key in keys_test:
                data_train.del_B(key)
            data_test = deepcopy(data)
            for key in keys_train:
                data_test.del_B(key)


            # do the training:
            estimator.fit(data_train, **fit_params)

            # do the prediction
            z_pred, y_pred = estimator.predict(data_test, **pred_params)

            z_gt_CV.append(data_test.z)
            z_pred_CV.append(z_pred)

            if y_pred is not None:
                y_gt_CV.append(data_test.y)
                y_pred_CV.append(y_pred)

            # if wanted and possible predict probabilities
            probability = fit_params.pop("probability", False)
            fit_params["probability"] = probability
            if probability:
                try:
                    z_prob, y_prob = estimator.predict_proba(data_test, **pred_params)
                    z_prob_CV.append(z_prob)
                    if y_prob is not None:
                        y_prob_CV.append(y_prob)
                except:
                    raise Warning("Probabilities could not be computed using the selected method.")


            prog_val += 1
            progress.update(prog_val)

    return z_gt_CV, z_pred_CV, z_prob_CV, y_gt_CV, y_pred_CV, y_prob_CV

def permutation_test_score():
    pass

def leraning_curve():
    pass

def validation_curve():
    pass




def loadCV(file, path=None):
    """
    Load CV scores from file.

    Parameters
    ----------
    file : string
        The file to load.
    path : string, optional
        The path where to load the file. If None the current working directory
        is assumed

    Returns
    ----------
    scores : dictionary
        The caclulated scores.
    """
    fname = prepareLoading(file, path, ".json")
    scores = pd.read_json(fname, typ='series').to_dict()
    return scores


def saveCV(file, path, scores):
    """
    Save CV scores to file.

    Parameters
    ----------
    file : string
        The file to save the data in.
    path : string, optional
        The path where to save the file. If None the current working directory
        is assumed
    scores : dictionary
        The caclulated scores.
    """

    fname = prepareSaving(file ,path, ".json")
    scores = pd.Series(scores)
    scores.to_json(fname)



def plotPrecRec(precRec, name="PrecisionRecall_CV", result_dir=None):
    """
    Plot Precision Recall Curve from CV results

    Parameters
    ----------
    precRec : array_like
        The precRec data.
    name : string, optional
        filename of the plot.
    result_dir : string, optional
        The path where to save the plot. If None the current working directory
        is assumed
    """
    fname = prepareSaving(name, result_dir, ".png")

    # Compute Precision-Recall and plot curve

    k = len(precRec)

    curves = np.zeros((k, 1000))
    interpol = np.linspace(0, 1., 1000)
    aucs = np.zeros((k))

    to_delete = []
    for i, entry in enumerate(precRec):
        try:
            pre, rec, thr = entry
            curves[i] = interp(interpol, pre, rec)
            aucs[i] = auc(rec, pre)
        except(TypeError, ValueError):
            to_delete.append(i)
            continue

    curves = np.delete(curves, to_delete, axis=0)
    aucs = np.delete(aucs, to_delete, axis=0)

    curves[:, 0] = 1
    curves[:, -1] = 0
    pre_mean = np.mean(curves, axis=0)
    pre_std = np.std(curves, axis=0)
    rec_mean = interpol
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    # plot
    plt.figure()
    plt.plot(rec_mean, pre_mean,label=u'AUC = %0.3f ± %0.3f ' % (auc_mean, auc_std))
    plt.errorbar(rec_mean[::100], pre_mean[::100], yerr=pre_std[::100],fmt='o', color='black', ecolor='black')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.savefig(fname)
    plt.close()

def plotROC(ROC, name="ROC_CV", result_dir=None):
    """
    Plot Receiver Operator Curve from CV results

    Parameters
    ----------
    ROC : array_like
        The ROC data.
    name : string, optional
        filename of the plot.
    result_dir : string, optional
        The path where to save the plot. If None the current working directory
        is assumed

    """
    fname = prepareSaving(name, result_dir, ".png")

    # Compute ROC and plot curve

    k = len(ROC)

    curves = np.zeros((k, 1000))
    interpol = np.linspace(0, 1., 1000)
    aucs = np.zeros((k))

    to_delete = []
    for i, entry in enumerate(ROC):
        try:
            fpr, tpr, thr = entry
            curves[i] = interp(interpol, fpr, tpr)
            aucs[i] = auc(fpr, tpr)
        except(TypeError, ValueError):
            to_delete.append(i)
            continue

    curves = np.delete(curves, to_delete, axis=0)
    aucs = np.delete(aucs, to_delete, axis=0)

    curves[:, 0] = 0
    curves[:, -1] = 1
    tpr_mean = np.mean(curves, axis=0)
    tpr_std = np.std(curves, axis=0)
    fpr_mean = interpol
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    # plot
    plt.figure()
    plt.plot(fpr_mean, tpr_mean,label=u'AUC = %0.3f ± %0.3f ' % (auc_mean, auc_std))

    plt.errorbar(fpr_mean[::100], tpr_mean[::100], yerr=tpr_std[::100],fmt='o', color='black', ecolor='black')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(fname)
    plt.close()
'''