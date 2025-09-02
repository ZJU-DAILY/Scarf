from dataclasses import dataclass
from enum import Enum

from utils.config import MetricConfig
from utils.types import MetricStatKeyType


class MetricScope(Enum):
    TM = "tm"
    TASK = "task"
    OPERATOR = "operator"


@dataclass
class MetricDef:
    name: str
    metric_scope: MetricScope
    min: float
    max: float

    def is_tm_metric(self):
        return self.metric_scope == MetricScope.TM

    def is_task_metric(self):
        return self.metric_scope == MetricScope.TASK

    def is_operator_metric(self):
        return self.metric_scope == MetricScope.OPERATOR

    def normalize(self, val):
        val = float(val)
        if self.min == self.max:
            return val
        if val < self.min:
            return 0
        if val > self.max:
            return 1
        normalized = (val - self.min) / (self.max - self.min)  # (0, 1)
        # normalized = normalized * 2 - 1  # (-1, 1)
        return normalized

    @staticmethod
    def of_tm(name: str, min_value: float, max_value: float):
        return MetricDef(name, MetricScope.TM, min_value, max_value)

    @staticmethod
    def of_task(name: str, min_value: float, max_value: float):
        return MetricDef(name, MetricScope.TASK, min_value, max_value)

    @staticmethod
    def of_operator(name: str, min_value: float, max_value: float):
        return MetricDef(name, MetricScope.OPERATOR, min_value, max_value)


def parse_metric_def(metric_config: list[MetricConfig]) -> list[MetricDef]:
    metric_defs = []
    for metric in metric_config:
        if metric.scope == MetricScope.TM.value:
            metric_defs.append(MetricDef.of_tm(metric.name, metric.min, metric.max))
        elif metric.scope == MetricScope.TASK.value:
            metric_defs.append(MetricDef.of_task(metric.name, metric.min, metric.max))
        elif metric.scope == MetricScope.OPERATOR.value:
            metric_defs.append(
                MetricDef.of_operator(metric.name, metric.min, metric.max)
            )
        else:
            raise ValueError(
                f"Metric with unknown scope: {metric}, scope is {metric.scope}"
            )
    return metric_defs


def parse_included_stats(included_stats: list[str]) -> list[MetricStatKeyType]:
    """
    Parse the included statistics from the configuration.
    """
    parsed_stats: list[MetricStatKeyType] = []
    for stat in included_stats:
        if stat not in ["avg", "min", "max", "sum", "skew"]:
            raise ValueError(f"Unknown metric statistic: {stat}")
        parsed_stats.append(stat)  # type: ignore
    return parsed_stats
