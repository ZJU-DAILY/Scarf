import random
from abc import ABC, abstractmethod
from typing import Any, Iterator

from flink.knob import KnobDef


class KnobValueSampler(ABC):
    @abstractmethod
    def generate(self) -> Iterator[dict[str, Any]]:
        """
        Generate a set of knob values for knob selection.
        :return:
        """
        pass


class LHSKnobValueSampler(KnobValueSampler):
    """Sample knob values using Latin Hypercube Sampling."""

    def __init__(self, knobs: list[KnobDef], num_samples: int, include_default: bool):
        """
        :param knobs: list of knobs
        :param num_samples: number of samples
        """
        assert num_samples > 0
        self.knobs = knobs
        self.num_samples = num_samples
        self.include_default = include_default

    def generate(self) -> Iterator[dict[str, Any]]:
        from scipy.stats._qmc import LatinHypercube

        lhs = LatinHypercube(d=len(self.knobs))

        if self.include_default:
            # Default values
            values = dict()
            for knob in self.knobs:
                values[knob.name] = knob.default
            yield values
            samples: list[list[float]] = lhs.random(self.num_samples - 1).tolist()
        else:
            samples: list[list[float]] = lhs.random(self.num_samples).tolist()

        for sample in samples:
            values = dict()
            for i, knob in enumerate(self.knobs):
                values[knob.name] = knob.get_value_from_percentile(sample[i])
            yield values


class RandomKnobValueSampler(KnobValueSampler):
    """Sample knob values randomly."""

    def __init__(self, knobs: list[KnobDef], num_samples: int, include_default: bool):
        """
        :param knobs: list of knobs
        :param num_samples: number of samples
        """
        assert num_samples > 0
        self.knobs = knobs
        self.num_samples = num_samples
        self.include_default = include_default

    def generate(self) -> Iterator[dict[str, Any]]:
        if self.include_default:
            # Default values
            values = dict()
            for knob in self.knobs:
                values[knob.name] = knob.default
            yield values
            remaining = self.num_samples - 1
        else:
            remaining = self.num_samples

        for _ in range(remaining):
            values = dict()
            for i, knob in enumerate(self.knobs):
                values[knob.name] = knob.get_value_from_percentile(random.random())
            yield values
