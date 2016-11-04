################################################################################
#                                                                              #
# The following code was released under the following conditions by Gary Doran #
#                                                                              #
################################################################################

"""
Copyright (c) 2013, Case Western Reserve University, Gary Doran
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of the owner nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
Utility functions and classes
"""
import numpy as np
import scipy.sparse as sp
from itertools import chain
from random import uniform


def rand_convex(n):
    rand = np.matrix([uniform(0.0, 1.0) for i in range(n)])
    return rand / np.sum(rand)


def spdiag(x):
    n = len(x)
    return sp.spdiags(x.flat, [0], n, n)


def partition(items, group_sizes):
    """
    Partition a sequence of items
    into groups of the given sizes
    """
    i = 0
    for group in group_sizes:
        yield items[i: i + group]
        i += group


def slices(groups):
    """
    Generate slices to select
    groups of the given sizes
    within a list/matrix
    """
    i = 0
    for group in groups:
        yield i, i + group
        i += group


class BagSplitter(object):
    def __init__(self, bags, classes):
        self.bags = bags
        self.classes = classes

    def __getattr__(self, name):
        if name == 'pos_bags':
            self.pos_bags = [bag for bag, cls in zip(self.bags, self.classes) if cls > 0.0]
            return self.pos_bags
        elif name == 'neg_bags':
            self.neg_bags = [bag for bag, cls in zip(self.bags, self.classes) if cls <= 0.0]
            return self.neg_bags
        elif name == 'neg_instances':
            self.neg_instances = np.vstack(self.neg_bags)
            return self.neg_instances
        elif name == 'pos_instances':
            self.pos_instances = np.vstack(self.pos_bags)
            return self.pos_instances
        elif name == 'instances':
            self.instances = np.vstack([self.neg_instances, self.pos_instances])
            return self.instances
        elif name == 'inst_classes':
            self.inst_classes = np.vstack([-np.ones((self.L_n, 1)), np.ones((self.L_p, 1))])
            return self.inst_classes
        elif name == 'pos_groups':
            self.pos_groups = [len(bag) for bag in self.pos_bags]
            return self.pos_groups
        elif name == 'neg_groups':
            self.neg_groups = [len(bag) for bag in self.neg_bags]
            return self.neg_groups
        elif name == 'L_n':
            self.L_n = len(self.neg_instances)
            return self.L_n
        elif name == 'L_p':
            self.L_p = len(self.pos_instances)
            return self.L_p
        elif name == 'L':
            self.L = self.L_p + self.L_n
            return self.L
        elif name == 'X_n':
            self.X_n = len(self.neg_bags)
            return self.X_n
        elif name == 'X_p':
            self.X_p = len(self.pos_bags)
            return self.X_p
        elif name == 'X':
            self.X = self.X_p + self.X_n
            return self.X
        elif name == 'neg_inst_as_bags':
            self.neg_inst_as_bags = [inst for inst in chain(*self.neg_bags)]
            return self.neg_inst_as_bags
        elif name == 'pos_inst_as_bags':
            self.pos_inst_as_bags = [inst for inst in chain(*self.pos_bags)]
            return self.pos_inst_as_bags
        else:
            raise AttributeError('No "%s" attribute.' % name)
        raise Exception("Unreachable %s" % name)
