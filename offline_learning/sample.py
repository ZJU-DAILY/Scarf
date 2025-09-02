import itertools
from copy import deepcopy
from typing import Optional

import numpy as np

from model.sac import MOSAC


class Sample:
    """
    Each Sample is a snapshot of algorithm with its achieved objectives, indexed by a unique optgraph_id.
    The MORL process can pick any sample to resume its training process or train with another optimization direction
    through those information.
    """

    # auto-increment id
    id_counter = itertools.count()

    def __init__(
        self,
        algo: MOSAC,
        objs: np.ndarray,
        optgraph_id=None,
    ):
        # thread-safe
        self.id = next(Sample.id_counter)
        self.algo = algo
        self.objs = objs
        self.optgraph_id = optgraph_id

    @classmethod
    def copy_from(cls, sample):
        algo = deepcopy(sample.algo)
        objs = deepcopy(sample.objs)
        optgraph_id = sample.optgraph_id
        return cls(algo, objs, optgraph_id)

    def __expr__(self):
        return f"Sample(id={self.id}, algo={self.algo}, objs={self.objs}, optgraph_id={self.optgraph_id})"

    def __str__(self):
        return self.__expr__()
