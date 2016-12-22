#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from .mica import *
from .misssvm import *
from .misvm import *
from .nsk import *
from .sbmil import *
from .sil import *
from .smil import *
from .stk import *
from .stmil import *
from .svm import *

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
MISVM: An implementation of multiple-instance support vector machines

The following algorithms are implemented:

  SVM     : a standard supervised SVM
  SIL     : trains a standard SVM classifier after applying bag labels to each
            instance
  MISVM   : the MI-SVM algorithm of Andrews, Tsochantaridis, & Hofmann (2002)
  miSVM   : the mi-SVM algorithm of Andrews, Tsochantaridis, & Hofmann (2002)
  NSK     : the normalized set kernel of Gaertner, et al. (2002)
  STK     : the statistics kernel of Gaertner, et al. (2002)
  MissSVM : the semi-supervised learning approach of Zhou & Xu (2007)
  MICA    : the MI classification algorithm of Mangasarian & Wild (2008)
  sMIL    : sparse MIL (Bunescu & Mooney, 2007)
  stMIL   : sparse, transductive  MIL (Bunescu & Mooney, 2007)
  sbMIL   : sparse, balanced MIL (Bunescu & Mooney, 2007)

__name__ = 'milSVM'
__version__ = '1.0'


from .svm import SVM
from .sil import SIL
from .nsk import NSK
from .smil import sMIL
from .misvm import miSVM, MISVM
from .stk import STK
from .stmil import stMIL
from .sbmil import sbMIL
from .mica import MICA
from .misssvm import MissSVM
"""
