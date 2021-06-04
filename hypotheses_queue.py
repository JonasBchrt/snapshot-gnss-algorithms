"""Priority queue for hypotheses.

Created on Tue Jun  9 16:05:03 2020

@author: Jonas Beuchert
"""
import heapq


class Queue:
    """Priority queue for hypotheses.

    Key element is the estimated upper-bound on the likelihood of the
    hypothesis.

    Author: Jonas Beuchert
    """

    def __init__(self):
        """Create empty priority queue."""
        self.q = []

    def add(self, h):
        """Insert hypothesis into priority queue.

        Input:
           h - Hypothesis to add
        """
        heapq.heappush(self.q, h)

    def hasElement(self):
        """Check if priority contains an element.

        Output:
           out - Logical value indicating if queue holds element
        """
        return len(self.q) > 0

    def popMostLikely(self):
        """Pop element with largest upper bound on  psuedo-likelihood.

        Return and remove element from priority queue.
        Output:
           out - Hypothesis
        """
        return heapq.heappop(self.q)
