###############################################################################
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
Implements standard code for problems that
require the Concave-Convex Procedure (CCCP),
or similar iteration.
"""
from sys import stderr


class CCCP(object):
    """
    Encapsulates the CCCP
    """
    TOLERANCE = 1e-6

    def __init__(self, verbose=True, max_iters=50, **kwargs):
        self.verbose = verbose
        self.max_iters = (max_iters + 1)
        self.kwargs = kwargs

    def mention(self, message):
        if self.verbose:
            print(message)

    def solve(self):
        """
        Called to solve the CCCP problem
        """
        for i in range(1, self.max_iters):
            self.mention('\nIteration %d...' % i)
            try:
                self.kwargs, solution = self.iterate(**self.kwargs)
            except Exception as e:
                if self.verbose:
                    print(stderr, 'Warning: Bailing due to error: %s' % e)
                return self.bailout(**self.kwargs)
            if solution is not None:
                return solution

        if self.verbose:
            print('Warning: Max iterations exceeded')
        return self.bailout(**self.kwargs)

    def iterate(self, **kwargs):
        """
        Should perform an iteration of the CCCP,
        using values in kwargs, and returning the
        kwargs for the next iteration.

        If the CCCP should terminate, also return the
        solution; otherwise, return 'None'
        """
        pass

    def bailout(self, **kwargs):
        """
        Return a solution in the case that the
        maximum allowed iterations was exceeded.
        """
        pass

    def check_tolerance(self, last_obj, new_obj=0.0):
        """
        Compares objective values, or takes the first
        value as delta if no second argument is given.
        """
        if last_obj is not None:
            delta_obj = abs(float(new_obj) - float(last_obj))
            self.mention('delta obj ratio: %.2e' % (delta_obj / self.TOLERANCE))
            return delta_obj < self.TOLERANCE
        return False
