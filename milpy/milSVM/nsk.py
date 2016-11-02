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
Implements the Normalized Set Kernel
of Gartner et al.
"""
import numpy as np

from ._quadprog import quadprog
from ._kernel import by_name as kernel_by_name
from ._util import spdiag
from .svm import SVM


class NSK(SVM):
    """
    Normalized set kernel of Gaertner, et al. (2002)
    """

    def __init__(self, *args, **kwargs):
        """
        @param kernel : the desired kernel function; can be linear, quadratic,
                        polynomial, or rbf [default: linear]
                        (by default, no normalization is used; to use averaging
                        or feature space normalization, append either '_av' or
                        '_fs' to the kernel name, as in 'rbf_av')
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
        super(NSK, self).__init__(*args, **kwargs)
        self._bags = None
        self._sv_bags = None
        self._bag_predictions = None

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._bags = map(np.asmatrix, bags)
        self._y = np.asmatrix(y).reshape((-1, 1))
        if self.scale_C:
            C = self.C / float(len(self._bags))
        else:
            C = self.C

        if self.verbose:
            print('Setup QP...')
        K, H, f, A, b, lb, ub = self._setup_svm(self._bags, self._y, C)

        # Solve QP
        if self.verbose:
            print('Solving QP...')
        self._alphas, self._objective = quadprog(H, f, A, b, lb, ub,
                                                 self.verbose)
        self._compute_separator(K)

    def _compute_separator(self, K):

        self._sv = np.nonzero(self._alphas.flat > self.sv_cutoff)
        self._sv_alphas = self._alphas[self._sv]
        self._sv_bags = [self._bags[i] for i in self._sv[0]]
        self._sv_y = self._y[self._sv]

        n = len(self._sv_bags)
        if n == 0:
            self._b = 0.0
            self._bag_predictions = np.zeros(len(self._bags))
        else:
            _sv_all_K = K[self._sv]
            _sv_K = _sv_all_K.T[self._sv].T
            e = np.matrix(np.ones((n, 1)))
            D = spdiag(self._sv_y)
            self._b = float(e.T * D * e - self._sv_alphas.T * D * _sv_K * e) / n
            self._bag_predictions = np.array(self._b
                                             + self._sv_alphas.T * D * _sv_all_K).reshape((-1,))

    def predict(self, bags):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @return : an array of length n containing real-valued label predictions
                  (threshold at zero to produce binary predictions)
        """
        if self._sv_bags is None or len(self._sv_bags) == 0:
            return np.zeros(len(bags))
        else:
            kernel = kernel_by_name(self.kernel, p=self.p, gamma=self.gamma)
            K = kernel(map(np.asmatrix, bags), self._sv_bags)
            return np.array(self._b + K * spdiag(self._sv_y) * self._sv_alphas).reshape((-1,))
