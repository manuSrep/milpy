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
Implements sMIL
"""
import numpy as np

from ._quadprog import quadprog
from ._kernel import by_name as kernel_by_name
from ._util import BagSplitter
from .nsk import NSK


class sMIL(NSK):
    """
    Sparse MIL (Bunescu & Mooney, 2007)
    """

    def __init__(self, *args, **kwargs):
        """
        @param kernel : the desired kernel function; can be linear, quadratic,
                        polynomial, or rbf [default: linear]
                        (by default, no normalization is used; to use averaging
                        or feature space normalization, append either '_av' or
                        '_fs' to the kernel name, as in 'rbf_av'; averaging
                        normalization is used in the original formulation)
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param scale_C : if True [default], scale C by the number of examples
        @param p : polynomial degree when a 'polynomial' kernel is used
                   [default: 3]
        @param gamma : RBF scale parameter when an 'rbf' kernel is used
                      [default: 1.0]
        @param verbose : print optimization status messages [default: True]
        @param sv_cutoff : the numerical cutoff for an example to be considered
                           a support vector [default: 1e-7]
        """
        super(sMIL, self).__init__(*args, **kwargs)

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        bs = BagSplitter(map(np.asmatrix, bags),
                         np.asmatrix(y).reshape((-1, 1)))
        self._bags = bs.neg_inst_as_bags + bs.pos_bags
        self._y = np.matrix(np.vstack([-np.ones((bs.L_n, 1)),
                                       np.ones((bs.X_p, 1))]))
        if self.scale_C:
            iC = float(self.C) / bs.L_n
            bC = float(self.C) / bs.X_p
        else:
            iC = self.C
            bC = self.C
        C = np.vstack([iC * np.ones((bs.L_n, 1)),
                       bC * np.ones((bs.X_p, 1))])

        if self.verbose:
            print('Setup QP...')
        K, H, f, A, b, lb, ub = self._setup_svm(self._bags, self._y, C)

        # Adjust f with balancing terms
        factors = np.vstack([np.matrix(np.ones((bs.L_n, 1))),
                             np.matrix([2.0 / len(bag) - 1.0
                                        for bag in bs.pos_bags]).T])
        f = np.multiply(f, factors)

        if self.verbose:
            print('Solving QP...')
        self._alphas, self._objective = quadprog(H, f, A, b, lb, ub,
                                                 self.verbose)
        self._compute_separator(K)

        # Recompute predictions for full bags
        self._bag_predictions = self.predict(bags)
