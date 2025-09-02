import copy
import logging
from typing import Dict, List, Optional

from flink.connector import FlinkConnector, FlinkJob
from utils.config import RootConfig
from utils.types import (
    JobPlanDict,
    JobPlanNodeDict,
    JobPlanNodeInputDict,
    MetricStatDict,
    VertexDict,
)


"""For testing only."""
class MockFlinkConnector(FlinkConnector):
    _DEFAULT_CORES_PER_TASK = 1
    _MINIMUM_MEM_PER_TASK_MB = 2048
    _PROCESSING_ABILITY = [
        1500,
        120000,
        4500,
        1500,
        1500,
        6000,
        9000,
        120000,
        400,
        300,
        200,
    ]

    def __init__(self, operators: List[str]):
        self._job_plan: Optional[JobPlanDict] = None
        self._running = False
        self._job_id_cnt = 0
        self._last_knobs = {}
        self._last_params = ""
        self._source_rate = 0
        self._operators = operators
        self._physical_plan = None
        self.logger = logging.getLogger("connector")

    def _parse_knobs_and_params(self, knobs: Dict[str, str], params: str):
        operator_chaining = (
            knobs.get("pipeline.operator-chaining.enabled", "false").lower() == "true"
        )
        object_reuse = knobs.get("pipeline.object-reuse", "false").lower() == "true"
        memory_mb = int(
            knobs.get("taskmanager.memory.process.size", "2048 mb").split()[0]
        )

        # params: "--<node-name1>.parallelism <value1> --<node-name2>.parallelism <value2> ..."
        parallelisms = [0] * len(self._operators)
        parts = params.split("--")
        for part in parts:
            if part.strip():
                key, value = part.strip().split(" ", 1)
                key = key.strip()
                value = value.strip()
                if key.endswith(".parallelism"):
                    node_name = key[: -len(".parallelism")]
                    if node_name in self._operators:
                        parallelisms[self._operators.index(node_name)] = int(value)

        return parallelisms, operator_chaining, object_reuse, memory_mb

    def get_vertices(self) -> List[Dict]:
        return [dict(id=v, name=v) for v in self._operators]

    def get_execution_plan_id(
        self, job: FlinkJob, params: str, knob_values: Dict[str, str]
    ) -> str:
        return "physicalplan"

    def submit(
        self,
        job: FlinkJob,
        params: Optional[str],
        knob_values: Dict[str, str],
        source_rate: int,
        savepoint_dir: Optional[str] = None,
    ) -> JobPlanDict:
        self._job_id_cnt += 1
        self._running = True
        self._last_knobs = knob_values
        self._last_params = "" if params is None else params
        self._source_rate = source_rate

        parallelisms, operator_chaining, object_reuse, memory_mb = (
            self._parse_knobs_and_params(knob_values, params)
        )
        rates = self._vertex_process_rates(knob_values, params)
        rates_arr = [rates[v] for v in self._operators]

        if not self._physical_plan:
            self._physical_plan = "physicalplan"
            self.logger.info(
                "New physical plan %s. Creating savepoint.", self._physical_plan
            )
        else:
            self.logger.info(
                "Using existing savepoint for plan %s", self._physical_plan
            )

        self.logger.info("Submit command: flink run fake job")
        self.logger.info(
            "Mock job: parallelisms=%s, operator_chaining=%s, object_reuse=%s, memory_mb=%d, abilities=%s",
            parallelisms,
            operator_chaining,
            object_reuse,
            memory_mb,
            rates_arr,
        )
        nodes: list[JobPlanNodeDict] = []
        for i in range(len(self._operators)):
            node: JobPlanNodeDict = {
                "id": self._operators[i],
                "description": self._operators[i],
                "operator": "",
                "operator_strategy": "",
                "optimizer_properties": {},
                "parallelism": parallelisms[i],
            }
            if i > 0:
                node["inputs"] = [
                    JobPlanNodeInputDict(
                        exchange="pipelined_bounded",
                        id=self._operators[i - 1],
                        num=0,
                        ship_strategy="FORWARD",
                    )
                ]
            nodes.append(node)

        self._job_plan = {
            "jid": "mock",
            "name": "mock_job",
            "type": "STREAMING",
            "nodes": nodes,
        }

        if memory_mb < self._MINIMUM_MEM_PER_TASK_MB:
            self.reset()

        return self._job_plan

    def submit_and_wait_until_stable(
        self,
        job: FlinkJob,
        params: str,
        knob_values: Dict[str, str],
        source_rate: int,
        savepoint_dir: str | None,
        config: RootConfig,
    ) -> JobPlanDict:
        return self.submit(job, params, knob_values, source_rate, savepoint_dir)

    def is_running(self) -> bool:
        return self._running

    def is_ended(self) -> bool:
        return not self._running

    def is_all_subtasks_running(self) -> bool:
        return self._running

    def get_running_job_plan(self) -> JobPlanDict:
        if self._job_plan is None:
            raise RuntimeError("No job running.")
        return copy.deepcopy(self._job_plan)

    def is_backpressured(self, plan: JobPlanDict) -> bool:
        return False

    def _vertex_process_rates(
        self, knobs: Dict[str, str], params: str
    ) -> Dict[str, float]:
        parallelisms, operator_chaining, object_reuse, memory_mb = (
            self._parse_knobs_and_params(knobs, params)
        )

        process_rates = {}
        bonus = 1
        if operator_chaining:
            bonus *= 2
        if object_reuse:
            bonus *= 2
        for i, operator in enumerate(self._operators):
            process_rates[operator] = (
                self._PROCESSING_ABILITY[i] * parallelisms[i] * bonus
            )
        return process_rates

    def _actual_throughput(self, max_thps: Dict[str, float]) -> float:
        return min(self._source_rate, min(max_thps.values()))

    def observe_task_metrics(
        self, plan: JobPlanDict, metric_names: List[str]
    ) -> Dict[str, Dict[str, MetricStatDict]]:
        max_thps = self._vertex_process_rates(self._last_knobs, self._last_params)
        actual_thp = self._actual_throughput(max_thps)

        min_vertex = None
        for k, v in max_thps.items():
            if v == actual_thp:
                min_vertex = k
                break
        assert min_vertex is not None

        job_backpressured = actual_thp < self._source_rate
        job_bottleneck_busy_frac = actual_thp / self._source_rate

        metrics_out = {}
        prev_thp = actual_thp
        parallelisms, _, _, _ = self._parse_knobs_and_params(
            self._last_knobs, self._last_params
        )

        for idx, v_id in enumerate(self._operators):
            max_thp = max_thps[v_id]
            busy_frac = actual_thp / max_thp if max_thp > 0 else 0.0
            if v_id == min_vertex:
                idle = 1 - busy_frac
                backpressured = 0.0
            elif self._operators.index(v_id) < self._operators.index(min_vertex):
                if job_backpressured:
                    idle = 0
                    backpressured = 1 - busy_frac
                else:
                    idle = 1 - busy_frac
                    backpressured = 0.0
            else:
                # 在瓶颈后，空闲
                idle = 1 - busy_frac
                backpressured = 0.0

            val_map = {
                "busyTimeMsPerSecond": 1000 * busy_frac,
                "idleTimeMsPerSecond": 1000 * idle,
                "backPressuredTimeMsPerSecond": 1000 * backpressured,
                "numRecordsInPerSecond": prev_thp, 
                "numRecordsOutPerSecond": actual_thp, 
            }
            # 每个 vertex 有 N 个 subtask，指标一样
            metrics = {}
            for mn in val_map:
                v = val_map[mn]
                metrics[mn] = {
                    "min": v,
                    "max": v,
                    "avg": v,
                    "sum": v * parallelisms[idx],
                    "skew": 0.0,
                }
            metrics_out[v_id] = metrics
            prev_thp = actual_thp

        return metrics_out

    def get_throughput(self, plan: JobPlanDict) -> List[float]:
        rates = self._vertex_process_rates(self._last_knobs, self._last_params)
        actual_thp = self._actual_throughput(rates)
        return [actual_thp]

    def get_total_processed(self, plan: JobPlanDict) -> int:
        return 1000000000

    def get_core_usage(self) -> int:
        parallelisms, operator_chaining, _, _ = self._parse_knobs_and_params(
            self._last_knobs, self._last_params
        )

        return sum(parallelisms)

    def get_memory_usage(self) -> int:
        parallelisms, operator_chaining, _, memory_mb = self._parse_knobs_and_params(
            self._last_knobs, self._last_params
        )
        if operator_chaining:
            return memory_mb * max(parallelisms)
        else:
            return memory_mb * sum(parallelisms)

    def stop(self, with_savepoint: bool) -> Optional[str]:
        self._running = False
        if with_savepoint:
            return f"/tmp/mock_savepoint_{self._job_id_cnt}"
        else:
            return None

    def reset_base_time(self) -> None:
        return

    def reset(self) -> None:
        self._job_plan = None
        self._running = False
        self._source_rate = 0
