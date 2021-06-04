"""Set for holding the n most likely hypotheses.

Created on Tue Jun  9 16:17:22 2020

@author: Jonas Beuchert
"""
import numpy as np
from hypothesis import Hypothesis


class Set:
    """Set for holding the n most likely hypotheses.

    Author: Jonas Beuchert

    Attributes
    ----------
        data            - Array of hypotheses
        minLmin         - Minimum likelihood of any hypothesis in set
        size            - Current size of set
        maxSize         - Maximum size of set
        minInd          - Index of hypothesis with minimum likelihood
    """

    def __init__(self, n):
        """Create empty set for given maximum number of elements.

        Input:
          n - Maximum number of hypotheses in set
        """
        # Create array with n empty hypotheses
        self.data = [Hypothesis() for _ in range(n)]
        # Initialize minimum likelihood
        self.minLmin = -np.inf
        # Initialize current and maximum size of set
        self.size = 0
        self.maxSize = n

    def add(self, h):
        """Add a hypothesis to the set.

        Adds a hypothesis to the set if its likelihood is larger than the
        smallest likelihood of any hypothesis in the set.
        Input:
           h - Hypothesis to add
        """
        # Check if set holds less than n elements
        if self.size < self.maxSize:
            # Add hypothesis to end of array
            self.size = self.size + 1
            self.data[self.size - 1] = h
            # Check if set is full now
            if self.size == self.maxSize:
                # Find smallest likelihood
                lmin = [_.lmin for _ in self.data]
                self.minInd = np.argmin(lmin)
                self.minLmin = lmin[self.minInd]
        # Check if likelihood is larger than smallest likelihood in set
        elif h.lmin > self.minLmin:
            # Replace hypothesis with smallest likelihood
            self.data[self.minInd] = h
            # Find new smallest likelihood
            lmin = [_.lmin for _ in self.data]
            self.minInd = np.argmin(lmin)
            self.minLmin = lmin[self.minInd]
