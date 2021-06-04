"""4D hypothesis (3D position and coarse time).

Created on Tue Jun  9 15:32:04 2020

@author: Jonas Beuchert
"""
import numpy as np


class Hypothesis:
    """4D hypothesis (3D position and coarse time).

    p - 3D receiver position
    t - Coarse time
    lmin - Pseudo-likelihood
    lmax - Estimated upper bound on pseudo-likelihoods in search space
    searchSpaceP - Diameter of position uncertainty cube
    searchSpaceT - Length of time uncertainty interval
    """

    def __init__(self, p=np.array([]), t=np.array([]), lmin=np.array([]),
                 lmax=np.array([]), searchSpaceP=np.array([]),
                 searchSpaceT=np.array([])):
        """Instantiate 4D hypothesis (3D position and coarse time).

        Inputs:
          p            - [Optional] 3D receiver position
          t            - [Optional] Coarse time
          lmin         - [Optional] Pseudo-likelihood
          lmax         - [Optional] Estimated upper bound on pseudo-
                         likelihoods in search space
          searchSpaceP - [Optional] Diameter of position uncertainty cube
          searchSpaceT - [Optional] Length of time uncertainty interval
        """
        self.p = p
        self.t = t
        self.lmin = lmin
        self.lmax = lmax
        self.searchSpaceP = searchSpaceP
        self.searchSpaceT = searchSpaceT

    def __lt__(self, other):
        """Compare two hypotheses.

        Return that this hypothesis has a smaller value, i.e., a higher
        priority if maximum achievable likelihood is larger.
        """
        return self.lmax > other.lmax
