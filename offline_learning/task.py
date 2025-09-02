from copy import deepcopy

from offline_learning.sample import Sample
from offline_learning.scalarization_methods import ScalarizationFunction

"""
Define a MOPG task, which is a pair of an algorithm and a scalarization weight.
"""


class Task:
    """
    A task consists of a Sample to start training from, and a scalarization that indicates the training direction.
    """

    def __init__(self, sample: Sample, scalarization: ScalarizationFunction, copy=True):
        if copy:
            self.sample = Sample.copy_from(sample)
        else:
            self.sample = sample
        self.scalarization = deepcopy(scalarization)
