#!/usr/bin/python
# -*- coding: utf8 -*-

"""
IDAPR MIL classifier.

:author: Manuel Tuschen
:date: 20.08.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import sys
sys.path.append("../milUtil")

from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator

from milUtil._milData import milData, MilError




class IDAPR(BaseEstimator):
    """
    Iterative Discriminant Axis Parallel Rectangel mil algorithm.

    The algorithm tries to find the upper bounds and lower bounds of
    relevant features which constitutes a hyper axis-parallel rectangle over
    positive instances.
    For every new instance, using this bounds it can be predicted whether it is
    positive or not.
    """

    def __init__(self, thr=100, tau=0.999, eps=0.01, step=0.05):
        self.thr = thr
        self.tau = tau
        self.eps = eps
        self.step = step


    def fit(self, data):

        self.pos_train = deepcopy(data) # extract positive data
        for key in self.pos_train.keys:
            if self.pos_train.get_B(key)[1] <= 0:
                self.pos_train.del_B(key)
        self.n_pos = self.pos_train.N_B
        self.neg_train = deepcopy(data)
        for key in self.neg_train.keys: # extract negative data
            if self.neg_train.get_B(key)[1] > 0:
                self.neg_train.del_B(key)
        self.n_neg = self.neg_train.N_B

        conv = False
        iter = 0
        relevant = np.full(data.N_D, True ) # mark feature as relevant
        while not conv: # iteratively grow and discriminate
            iter += iter + 1

            lbs, ubs = grow(self.pos_train, relevant);

            relevant, convergence = discrim(self.neg_train, lbs, ubs, relevant, self.thr)

        self.lbs, self.ubs = expand(self.pos_train, lbs, ubs, relevant, self.tau, self.eps, self.step)


    def predict(self, data):
        pass





################################################################################
#                                                                              #
#                           helper functions                                   #
#                                                                              #
################################################################################



def grow(pos_data, relevant):
    """
    Find smallest APR that covers at least one instance of all positive bags.
    """
    if not np.any(relevant):
        raise MilError('No relevant features')

    # create a dataset with only relevant features
    relevant_data = deepcopy(pos_data)
    relevant_data.del_feat(np.where(relevant == False)[0])


    # const minimax APR to find initial seed
    for key in relevant_data.keys: # go through all bags
        min_bag = np.concatenate((min_bag, np.min(relevant_data.get_B(key)[0], axis=1)), axis=1)
        max_bag = np.concatenate((max_bag, np.max(relevant_data.get_B(key)[0], axis=1)), axis=1)

    min_bound = np.min(max_bag, axis=1)
    max_bound = np.max(min_bag, axis=1)
    minmax = np.concatenate((min_bound,max_bound), axis=1)
    center = np.mean(minmax, axis=1)


    # choosing initial seed as those instance closest to minimax APR
    start_inst = np.argmin(cdist(relevant_data.X, center))
    start_bag = relevant_data.pos_x[start_inst][0] # key of starting bag


    chosen = np.zeros((relevant_data.N_B, relevant_data.N_D))
    chosen[0] = PBags{starting_bags, 1}(starting_inst,:);
    usage = zeros(1, num_bags);
    usage(1, starting_bags) = 1;
    pointer_bags = zeros(1, num_bags);
    pointer_instances = zeros(1, num_bags);
    pointer_bags(1, 1) = starting_bags;
    pointer_instances(1, 1) = starting_inst;

    for i in range(pos_data.N_B):
        # greedy step, add instance which least increases APR
        curAPR = minmax(chosen(1:(i - 1),:).T)
        curSize = sum(curAPR(:, 2)-curAPR(:, 1))
        incremental = 1e20;
        for j = 1:num_bags
            if (usage(1, j) == 0)
                tempsize = size(PBags{j, 1});
            for k=1:tempsize(1)
                tempAPR = minmax([curAPR';PBags{j,1}(k,:)]');
                temp_increment = sum(tempAPR(:, 2)-tempAPR(:, 1))-curSize;
                if (temp_increment < incremental)
                    incremental = temp_increment;
                    pointer_bags(1, i) = j;
                    pointer_instances(1, i) = k;
                    chosen(i,:)=PBags{j, 1}(k,:);
        usage(1, pointer_bags(1, i)) = 1;

        # backfitting
        changed = 1;
        while (changed == 1):
            changed = 0;
            for m=1:i
                tempAPR = [chosen(1:(m - 1),:);chosen((m + 1):i,:)];
                tempAPR = minmax(tempAPR
                ');
                tempsize = sum(tempAPR(:, 2)-tempAPR(:, 1));
                incremental = 1e20;
                cur_instance = pointer_instances(1, m);
                cur_bag = pointer_bags(1, m);
                size_cur_bag = size(PBags
                {cur_bag, 1});
                for n=1:size_cur_bag(1)
                    tempAPR1 = minmax([tempAPR';PBags{cur_bag,1}(n,:)]');
                    temp_increment = sum(tempAPR1(:, 2)-tempAPR1(:, 1))-tempsize;
                    if (temp_increment < incremental)
                        incremental=temp_increment;
                        pointer_instances(1, m)=n;
                        chosen(m,:)=PBags
                        {cur_bag, 1}(n,:);

                if (pointer_instances(1, m)~ = cur_instance)
                    changed = 1;


    resultAPR = minmax(chosen.T);
    lbs = resultAPR(:, 1).T;
    ubs = resultAPR(:, 2).T;

    return lbs, ubs


def discrim(self, NBags,lbs,ubs,relevant,threshold):

    size1 =len(NBags)
    neg_instances = []
    for i in range(size1):
        neg_instances.append(NBags[i])

    num_instances = len(neg_instances)
    dimension = neg_instances.shape[1]
    lowerbounds = np.zeros(dimension)
    upperbounds = np.zeros(dimension)
    pointer = 0

    for i in range(dimension):
        if relevant[i] == 1:
        pointer = pointer + 1;
        lowerbounds[i] = lbs[pointer]
        upperbounds[i] = ubs[pointer]

    count = 0
    discrimed = np.zeros(num_instances)
    under_consider = relevant
    result = np.zeros(dimension)

    while (~((count == num_instances) or (np.sum(under_consider) == 0))):
        discrimlist = cell(dimension, 1)
        for i in range(num_instances):
            if discrimed[i] == 0:
                outdistance = np.zeros(dimension)
            for j in range(dimension):
                if under_consider[j] == 1:
                    if neg_instances[j] < lowerbounds[j]:
                        outdistance[j] = np.abs(neg_instances[i,j]- lowerbounds[j])
                    if outdistance[j] >= threshold:
                        discrimlist[j] = [discrimlist[j], i]
                else:
                    if neg_instances[i,j] > upperbounds[j]:
                        outdistance[j] = np.abs(neg_instances[i,j] - upperbounds[j])
                        if outdistance[j] >= threshold:
                            discrimlist[j] = [discrimlist[j], i]

        [maximum, index] = np.max(outdistance)
        if maximum == 0:
            discrimed[i] = 1
            count += 1
        else:
            if outdistance[index] < threshold:
                discrimlist[index] = [discrimlist[index], i]

        discrim_num = np.zeros(dimension)
        for k in range(dimension):
            tempsize = size(discrimlist{k, 1})
            discrim_num(1, k) = tempsize(2)


        [maximum, index] = np.max(discrim_num)
        if maximum != 0:
            for m in range(maximum):
                discrimed(discrimlist {index, 1}(1, m), 1)=1;
                count = count + 1;

        under_consider[index] = 0
        result[index] = 1

    if np.sum(result != relevant) == 0:
        convergence = 1
    else:
        convergence = 0


def expand(PBags, lbs, ubs, relevant, tau, epsilon, step):

    if np.sum(relevant) == 0:
        raise MilError('no relevant features')

    size1 = size(PBags);
    num_bags = size1(1);
    size2 = size(relevant);
    dimension = size2(2);

    temp_pbags = cell(num_bags, 1);
    for i=1:num_bags
        count = 0;
    for j=1:dimension
        if (relevant(1, j) == 1)
            count = count + 1;
            temp_pbags{i, 1}(:, count)=PBags {i, 1}(:, j);

    positives = []
    for i in range(num_bags):
        positives = [positives,temp_pbags{i, 1}]

    size2 = size(positives)
    num_instances = size2(1);
    for i in rangesum(relevant)
        sigma = ((lbs(1, i) - ubs(1, i)) / 2) / norminv((1 - tau) / 2);
        cur_dimension = [];
        for j in range(num_instances):
            if ((positives(j, i) >= lbs(1, i)) & (positives(j, i) <= ubs(1, i))):
                cur_dimension = [cur_dimension, positives(j, i)];
        size3 = size(cur_dimension);
        coeff = 1 / size3(2)


        templb = lbs(1, i);
        tempprob = coeff * sum(normcdf(templb, cur_dimension, sigma))
        while (tempprob > epsilon / 2):
            templb = templb - step;
            tempprob = coeff * sum(normcdf(templb, cur_dimension, sigma))

        tempub = ubs(1, i);
        tempprob = coeff * sum(normcdf(tempub, cur_dimension, sigma))
        while (tempprob <= (1 - epsilon / 2)):
            tempub = tempub + step;
            tempprob = coeff * sum(normcdf(tempub, cur_dimension, sigma))

        lbounds(1, i) = templb
        ubounds(1, i) = tempub

