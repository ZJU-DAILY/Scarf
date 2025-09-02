from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, NamedTuple, NotRequired, TypedDict

import numpy as np
import torch as th

from utils.config import RootConfig

DEFAULT_PARALLELISM_KNOB_NAME = "parallelism.default"


class KnobUnit(ABC):
    @property
    @abstractmethod
    def UNITS(self) -> dict[str, int]:
        pass

    def __init__(self, value: int, unit: str):
        self._value = value
        self._unit = unit.lower()
        if self._unit not in self.UNITS:
            raise ValueError(f"Unknown unit: {self._unit}")

    @staticmethod
    @abstractmethod
    def from_str(value: str):
        pass

    def value(self):
        return self._value

    def unit(self):
        return self._unit

    def to_base_unit(self) -> int:
        return self._value * self.UNITS[self._unit]

    def __repr__(self):
        return f"{self._value} {self._unit}"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, KnobUnit):
            return False
        return self._value == other._value and self._unit == other._unit

    def __hash__(self):
        return hash((self._value, self._unit))


class StorageUnit(KnobUnit):
    UNITS = {
        "b": 1,
        "kb": 1024,
        "mb": 1024 * 1024,
        "gb": 1024 * 1024 * 1024 * 1024,
        "tb": 1024 * 1024 * 1024 * 1024 * 1024,
    }

    @staticmethod
    def from_str(value: str):
        if value is None:
            raise ValueError("value is null")
        parts = value.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid unit: {value}")
        value, unit = parts
        return StorageUnit(int(value), unit)


class TimeUnit(KnobUnit):
    UNITS = {
        "ms": 1,
        "s": 1000,
        "min": 1000 * 60,
        "h": 1000 * 60 * 60,
        "d": 1000 * 60 * 60 * 24,
    }

    @staticmethod
    def from_str(value: str):
        if value is None:
            raise ValueError("value is null")
        parts = value.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid unit: {value}")
        value, unit = parts
        return TimeUnit(int(value), unit)


class ObsDict(TypedDict):
    """
    A type alias for the observation space.
    """

    node_features: np.ndarray  # (num_nodes, num_feature_dim)
    adjacency_matrix: np.ndarray  # (num_nodes, num_nodes)


class MetricStatDict(TypedDict):
    """
    A type alias for the metrics statistics.
    """

    min: float
    max: float
    avg: float
    sum: float
    skew: float


type MetricStatKeyType = Literal["min", "max", "avg", "sum", "skew"]


class JobPlanNodeInputDict(TypedDict):
    """
    A type alias for the job plan node input.
    """

    exchange: str
    id: str
    num: int
    ship_strategy: str


class JobPlanNodeDict(TypedDict):
    """
    A type alias for the job plan node.
    """

    description: str
    id: str
    inputs: NotRequired[list[JobPlanNodeInputDict]]
    operator: str
    parallelism: int


class VertexDict(TypedDict):
    """
    A type alias for the vertex.
    """

    id: str
    name: str
    parallelism: int
    status: str


class JobPlanDict(TypedDict):
    """
    A type alias for the job plan.
    """

    jid: str
    name: str
    nodes: list[JobPlanNodeDict]
    type: str


class SparseObsDict(TypedDict):
    """
    A type alias for the sparse observation.
    """

    node_features: th.Tensor  # (num_nodes, num_feature_dim)
    edge_indices: th.Tensor  # (2, total_num_edges)
    batch_vector: th.Tensor  # (num_nodes,)


class GraphDictReplayBufferSamples(NamedTuple):
    observations: SparseObsDict
    actions: th.Tensor
    next_observations: SparseObsDict
    dones: th.Tensor
    rewards: th.Tensor


class JobState(Enum):
    UNDER_PROVISIONED = 0
    OPTIMAL = 1
    OVER_PROVISIONED = 2


class ParallelismKnobType(Enum):
    GROUP = "group"
    CLUSTER = "cluster"


@dataclass
class ParallelismKnob:
    name: str
    operator: str
    type: ParallelismKnobType


class ParallelismKnobSet:
    def __init__(self, conf: RootConfig):
        # If conf.knobs.operator_names is not empty, this is a DataStream job, and we set the parallelisms
        # of individual operators; otherwise, this is a Table API job, and we set the parallelism.default
        # parameter.
        if conf.knobs.operator_names:
            if "parallelism" not in [k.name for k in conf.knobs.operator_knobs]:
                raise ValueError("Missing parallelism knob in operator knobs")
            self.knobs = [
                ParallelismKnob(
                    name=name + ".parallelism",
                    operator=name,
                    type=ParallelismKnobType.GROUP,
                )
                for name in conf.knobs.operator_names
            ]
            self.default_parallelisms = [
                int(k.default) for k in conf.knobs.operator_knobs
            ] * len(self.knobs)
        else:
            if DEFAULT_PARALLELISM_KNOB_NAME not in [
                k.name for k in conf.knobs.cluster_knobs
            ]:
                raise ValueError("Missing parallelism knob in cluster knobs")

            self.knobs = [
                ParallelismKnob(
                    name=DEFAULT_PARALLELISM_KNOB_NAME,
                    operator="job",
                    type=ParallelismKnobType.CLUSTER,
                )
            ]
            self.default_parallelisms = [
                int(k.default)
                for k in conf.knobs.cluster_knobs
                if k.name == DEFAULT_PARALLELISM_KNOB_NAME
            ]
            if len(self.default_parallelisms) != 1:
                raise ValueError("Multiple parallelism knobs found in cluster knobs")

    def is_sql_job(self) -> bool:
        """Check if this is a SQL job."""
        return (
            len(self.knobs) == 1 and self.knobs[0].type == ParallelismKnobType.CLUSTER
        )

    @property
    def size(self) -> int:
        return len(self.knobs)

    def get_params_and_knobs(
        self, parallelisms: list[int]
    ) -> tuple[str, dict[str, Any]]:
        if len(parallelisms) != self.size:
            raise ValueError(
                f"Expected {self.size} parallelisms, but got {len(parallelisms)}"
            )

        params: list[str] = []
        knobs: dict[str, Any] = {}
        for i, parallelism in enumerate(parallelisms):
            knob = self.knobs[i]
            if knob.type == ParallelismKnobType.GROUP:
                # The parallelism of each operator is given in the job param like:
                # --opname1.parallelism=4 --opname2.parallelism=4 ...
                params.append(f"--{knob.name} {parallelism}")
            else:
                knobs[knob.name] = parallelism

        return " ".join(params), knobs

    def __iter__(self):
        """Iterate over the knobs."""
        return iter(self.knobs)
