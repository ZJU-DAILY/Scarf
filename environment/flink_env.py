import json
import logging
import math
import pprint
import re
import time
from tracemalloc import Snapshot
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces, envs
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

from flink.connector import FlinkConnector, FlinkJob
from flink.knob import KnobDef
from flink.metrics import MetricDef
from flink.sql_plan import SavepointStore, generate_plan_hash, get_job_plan_hash
from utils.common import is_throughput_stable
from utils.types import JobPlanDict, MetricStatDict, MetricStatKeyType, ObsDict


# match expressions like numRecordsInPerSecond / busyTimeMsPerSecond
OBS_EXPRESSION_PATTERN = r"(\w+)\s*([+\-*/])\s*(\w+)"


def register_flink_env(config: dict):
    """
    Register the Flink environment in OpenAI Gym.
    """
    gym.envs.registration.register(
        id="FlinkEnv-v1",
        entry_point="environment.flink_env:FlinkEnv",
        max_episode_steps=None,
        kwargs={
            "render_mode": None,
            **config,
        },
        disable_env_checker=True,
    )


def calculate_expression(
    dict_1: MetricStatDict, dict_2: MetricStatDict, op: str
) -> MetricStatDict:
    if op == "+":
        return {
            "min": dict_1["min"] + dict_2["min"],
            "max": dict_1["max"] + dict_2["max"],
            "avg": dict_1["avg"] + dict_2["avg"],
            "sum": dict_1["sum"] + dict_2["sum"],
            "skew": max(dict_1["skew"], dict_2["skew"]),
        }
    elif op == "-":
        return {
            "min": dict_1["min"] - dict_2["max"],
            "max": dict_1["max"] - dict_2["min"],
            "avg": dict_1["avg"] - dict_2["avg"],
            "sum": dict_1["sum"] - dict_2["sum"],
            "skew": max(dict_1["skew"], dict_2["skew"]),
        }
    elif op == "*":
        return {
            "min": dict_1["min"] * dict_2["min"],
            "max": dict_1["max"] * dict_2["max"],
            "avg": dict_1["avg"] * dict_2["avg"],
            "sum": dict_1["sum"] * dict_2["sum"],
            "skew": max(dict_1["skew"], dict_2["skew"]),
        }
    elif op == "/":
        return {
            "min": dict_1["min"] / dict_2["max"] if dict_2["max"] != 0 else 0,
            "max": dict_1["max"] / dict_2["min"] if dict_2["min"] != 0 else 0,
            "avg": dict_1["avg"] / dict_2["avg"] if dict_2["avg"] != 0 else 0,
            "sum": dict_1["sum"] / dict_2["sum"] if dict_2["sum"] != 0 else 0,
            "skew": max(dict_1["skew"], dict_2["skew"]),
        }
    else:
        raise ValueError("Unrecognized operator: %s" % op)


class ActionSpec:
    """
    A class to represent the action specification.
    """

    def __init__(
        self,
        operator_names: list[str],
        operator_knobs: list[KnobDef],
        cluster_knobs: list[KnobDef],
    ):
        self.operator_names = operator_names
        self.operator_knobs = operator_knobs
        self.cluster_knobs = cluster_knobs

    @property
    def action_space(self):
        return len(self.operator_names) * len(self.operator_knobs) + len(
            self.cluster_knobs
        )

    def get_params_and_knobs(self, action: np.ndarray) -> tuple[str, dict[str, Any]]:
        """
        Get the parameters and knobs from the action.

        :param action: the action output from the model.
        :return:  a tuple of (parameters, knobs).
          - Parameters indicate the operator knobs. It is a string of the form:
            "--opname1.knob1=value1 --opname2.knob2=value2 ..."
          - Knobs include cluster configurations chosen in the knob selection. It is a
            dictionary of the form: {"knob1": value1, "knob2": value2 ...}
        """
        self._check_action(action)
        action = action.flatten()

        params = []
        knobs = {}
        index = 0
        # Action: [[op_1 knobs], [op_2 knobs], ..., [op_n knobs], [cluster knobs]]
        for i, group in enumerate(self.operator_names):
            for j, knob in enumerate(self.operator_knobs):
                # [-1, 1] -> [0, 1]
                percentile = (action[index].item() + 1) / 2
                val = knob.format_value(knob.get_value_from_percentile(percentile))
                params.append(f"--{group}.{knob.name} {val}")
                index += 1
        for i, knob in enumerate(self.cluster_knobs):
            percentile = (action[index].item() + 1) / 2
            val = knob.format_value(knob.get_value_from_percentile(percentile))
            knobs[knob.name] = val
            index += 1
        return " ".join(params), knobs

    def get_default_params_and_knobs(self):
        """
        Get the default parameters and knobs.

        :return: a tuple of (parameters, knobs).
          - Parameters indicate the Operator knobs. It is a string of the form:
            "--group1.knob1=value1 --group2.knob2=value2 ..."
          - Knobs include cluster configurations chosen in the knob selection. It is a
            dictionary of the form: {"knob1": value1, "knob2": value2 ...}
        """
        params = []
        knobs = {}
        for i, group in enumerate(self.operator_names):
            for j, knob in enumerate(self.operator_knobs):
                params.append(f"--{group}.{knob.name} {knob.default}")
        for i, knob in enumerate(self.cluster_knobs):
            knobs[knob.name] = knob.format_value(knob.default)
        return " ".join(params), knobs

    def _check_action(self, action: np.ndarray):
        # Requirements: only the last dimension should have size > 1, and each element is between [0, 1]
        for dim in action.shape[:-1]:
            if dim != 1:
                raise ValueError(f"Invalid action shape: {action.shape}")
        if action.shape[-1] != self.action_space:
            raise ValueError(f"Invalid action shape: {action.shape}")
        if not np.all((action >= -1) & (action <= 1)):
            raise ValueError(f"Action values should be in [0, 1], but found {action}")

    def __repr__(self):
        return (
            f"ActionSpec(operator_names={self.operator_names}, "
            f"operator_knobs={self.operator_knobs}, "
            f"global_knobs={self.cluster_knobs})"
        )


class ObsSpec:
    """
    A class to represent the observation specification.
    """

    def __init__(
        self, task_metrics: list[MetricDef], included_stat: list[MetricStatKeyType]
    ):
        """
        Initialize the observation specification.

        :param task_metrics: metrics to be included in the observation
        :param included_stat: included statistics, can be ["min", "max", "avg", "sum", "skew"]
        """
        # for now, only support task metrics
        for metric in task_metrics:
            if not metric.is_task_metric():
                raise ValueError(f"Invalid metric: {metric.name} is not a task metric")
        self.task_metrics = task_metrics
        self.included_stat = included_stat

    def __repr__(self):
        return f"ObsSpec(operator_metrics={self.task_metrics})"

    def get_metric_names(self) -> list[str]:
        return [metric.name for metric in self.task_metrics]

    @property
    def feature_dim(self) -> int:
        """
        Get the feature dimension of the observation.

        :return: the feature dimension
        """
        return len(self.task_metrics) * len(self.included_stat)

    def get_obs(
        self,
        plan: JobPlanDict,
        metrics: Optional[dict[str, list[MetricStatDict]]],
    ) -> ObsDict:
        """
        Get the observation from the plan.

        :param plan: the plan output from the connector
        :param metrics: the metrics output from the connector, a dictionary mapping vertex ID to a list of metric values
            (one per metric name), each metric value a dict containing min, max, avg, sum, skew.
        :param ranges: the min and max values for each metric
        :return: the observation of type ObsDict
        """
        # Check metrics shape
        if metrics is not None:
            for vertex in metrics:
                if len(metrics[vertex]) != len(self.task_metrics):
                    raise ValueError(
                        f"Metrics shape mismatch: {len(metrics[vertex])} != {len(self.task_metrics)}"
                    )

        node_features: list[list[float]] = [[] for _ in range(len(plan["nodes"]))]
        adjacency_matrix = np.zeros(
            (len(plan["nodes"]), len(plan["nodes"])), dtype=np.float64
        )

        # First obtain node indexes
        node_indexes = dict[str, int]()
        index = 0
        for node in plan["nodes"]:
            node_indexes[node["id"]] = index
            index += 1

        # Fill in node features and adjacency matrix
        for i, node in enumerate(plan["nodes"]):
            node_index = node_indexes[node["id"]]
            for j, metric in enumerate(self.task_metrics):
                for stat in self.included_stat:
                    if metrics is not None:
                        raw_val: float | str = metrics[node["id"]][j][stat]
                        # NaN -> 0
                        if isinstance(raw_val, str):
                            try:
                                raw_val = float(raw_val)
                            except:
                                raw_val = float("nan")
                        if math.isnan(raw_val):
                            node_features[node_index].append(0.0)
                        else:
                            node_features[node_index].append(metric.normalize(raw_val))
                    else:
                        # If metrics is None, return zero values
                        node_features[node_index].append(0.0)

            if "inputs" in node:
                for input_node in node["inputs"]:
                    input_node_index = node_indexes[input_node["id"]]
                    # For now, we use undirected graph
                    adjacency_matrix[node_index][input_node_index] = 1.0
                    adjacency_matrix[input_node_index][node_index] = 1.0

        return {
            "node_features": np.array(node_features, dtype=np.float64),
            "adjacency_matrix": adjacency_matrix,
        }


class FlinkEnv(gym.Env):

    PENALTY_OBJS = [-1.0, -1.0]

    def __init__(self, **kwargs):
        self.logger = logging.getLogger("env")
        self.logger.setLevel(kwargs["log_level"])
        self.action_spec: ActionSpec = kwargs["action_spec"]
        self.obs_spec: ObsSpec = kwargs["obs_spec"]
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.action_spec.action_space,), dtype=np.float64
        )
        # 0 refers to the number of nodes
        node_features_space = spaces.Box(
            0,
            1,
            shape=(
                0,
                len(self.obs_spec.task_metrics) * len(self.obs_spec.included_stat),
            ),
            dtype=np.float64,
        )
        # edge indices: (2, num_edges)
        edge_indices_space = spaces.Box(0, 1, shape=(2, 0), dtype=np.float64)
        # batch vector: (num_nodes)
        batch_vector_space = spaces.Box(0, 1, shape=(0,), dtype=np.float64)

        self.observation_space = spaces.Dict(
            {
                "node_features": node_features_space,
                "edge_indices": edge_indices_space,
                "batch_vector": batch_vector_space,
            }
        )
        self.flink_connector: FlinkConnector = kwargs["flink_connector"]
        self.flink_job: FlinkJob = kwargs["flink_job"]
        self.job_warmup_sec: float = kwargs["job_warmup_sec"]
        self.monitor_interval_sec: float = kwargs["monitor_interval_sec"]  # seconds
        self.max_wait_attempts: int = kwargs["max_wait_attempts"]
        self.max_monitor_attempts: int = kwargs["max_monitor_attempts"]
        self.stable_window_size: int = kwargs["stable_window_size"]
        self.throughput_weight: float = kwargs["throughput_weight"]
        self.resource_weight: float = kwargs["resource_weight"]
        self.max_core_usage: int = kwargs["max_core_usage"]
        self.max_memory_usage: int = kwargs["max_memory_usage"]
        self.savepoint_size: bool = kwargs["savepoint_size"]
        self.source_rate: int = kwargs["source_rate"]
        self.random_seed = 42
        self.current_step = 0
        self.render_mode = kwargs["render_mode"]
        self.savepoint_store: str = kwargs["savepoint_store"]
        # physical plan id -> savepoint path

    def step(self, action):
        self.current_step += 1

        self.logger.debug("Env step %d", self.current_step)
        if isinstance(action, np.ndarray):
            action = action.clip(-1, 1)
        elif isinstance(action, torch.Tensor):
            action = action.clamp(-1, 1)
        else:
            raise ValueError("Action type not supported: %s" % type(action))

        self.logger.info("Action: %s", action)

        params, knob_values = self.action_spec.get_params_and_knobs(action)
        # self.logger.debug("Params: %s", params)
        # self.logger.debug("Knob values: %s", knob_values)

        obs, objs, info = self._get_obs_objs_and_info(params, knob_values)
        self.logger.debug("Obs:\n%s", pprint.pformat(obs))
        self.logger.info("Objs: %s", objs)

        done = False

        return (
            obs,
            0.0,
            done,
            False,
            {"objs": np.array(objs), **info},
        )

    def _get_penalty_obs_objs_and_info(
        self, core_usage=None, memory=None
    ) -> tuple[ObsDict, list[float], dict[str, Any]]:
        n_operators = len(self.action_spec.operator_names)
        if n_operators == 0:
            n_operators = 1
        obs: ObsDict = {
            "node_features": np.zeros(
                (n_operators, self.obs_spec.feature_dim), dtype=np.float64
            ),
            "adjacency_matrix": np.zeros((n_operators, n_operators), dtype=np.int64),
        }
        objs = self.PENALTY_OBJS
        throughput = [0.0]
        if core_usage is None:
            core_usage = self.max_core_usage * 2
        if memory is None:
            memory = self.max_memory_usage * 2
        info = {"throughput": throughput, "core": core_usage, "memory": memory}
        self.logger.info(
            "Throughput: %.1f, Core usage: %.1f, Memory usage: %.1f MB",
            np.average(throughput),
            core_usage,
            memory,
        )
        return obs, objs, info

    def _check_metrics(
        self,
        metrics: dict[str, list[MetricStatDict]],
        throughput: list[float],
        core_usage: float,
        memory_usage: float,
    ) -> bool:
        # Check if all metrics are valid
        for metric in metrics.values():
            if not metric:
                self.logger.warning("Metrics are empty")
                return False
        # Check if throughput is valid
        if not throughput or any(t < 0 for t in throughput):
            self.logger.warning("Throughput is empty or negative")
            return False
        # Check if core usage and memory usage are valid
        if core_usage < 0:
            self.logger.warning("Core usage is negative")
            return False
        if memory_usage < 0:
            self.logger.warning("Memory usage is negative")
            return False
        return True

    def _get_savepoint(self, plan_hash: str) -> Optional[str]:
        store = SavepointStore(self.savepoint_store)
        return store.get(plan_hash)

    def _save_savepoint(self, plan_hash: str, savepoint_dir: str) -> None:
        store = SavepointStore(self.savepoint_store)
        store.save(plan_hash, savepoint_dir)

    def _create_savepoint(self, params, knob_values, plan_hash: str):
        self.flink_connector.reset()
        try:
            plan = self.flink_connector.submit(
                self.flink_job, params, knob_values, self.source_rate
            )
        except Exception as e:
            self.logger.error(
                "Error occurred during job submission for savepoint creation: %s",
                e,
                exc_info=True,
            )
            raise

        # All jobs should be running within 1 minute
        attempt_time = 0
        wait_interval = 10
        while attempt_time < 6:
            attempt_time += 1
            if self.flink_connector.is_ended():
                self.logger.error(
                    "Job has ended prematurely, returning penalty. Attempt %d/%d",
                    attempt_time,
                    6,
                )
                raise RuntimeError("Job ended prematurely")
            if self.flink_connector.is_all_subtasks_running():
                break
            self.logger.debug(
                "Attempt %d/6: Waiting for all subtasks to run. Retrying in %d s...",
                attempt_time,
                wait_interval,
            )
            time.sleep(wait_interval)

        # Then wait for the job to process
        wait_interval = 30
        while True:
            if self.flink_connector.is_ended():
                self.logger.error(
                    "Job has ended prematurely, returning penalty. Attempt %d/20",
                    attempt_time,
                )
                raise RuntimeError("Job ended prematurely")
            processed = self.flink_connector.get_total_processed(plan)
            self.logger.info(
                "Total input processed: %d/%d", processed, self.savepoint_size
            )
            if processed >= self.savepoint_size:
                break

            time.sleep(wait_interval)

        # Create savepoint
        try:
            savepoint_dir = self.flink_connector.stop(with_savepoint=True)
        except Exception as e:
            self.logger.error(
                "Error occurred during stopping with savepoint: %s", e, exc_info=True
            )
            raise

        if savepoint_dir is None:
            self.logger.error("Savepoint creation failed: savepoint_dir is None")
            raise RuntimeError("Savepoint creation failed: savepoint_dir is None")

        self.logger.info(
            "Savepoint for plan %s created at %s", plan_hash, savepoint_dir
        )
        self._save_savepoint(plan_hash, savepoint_dir)

    def _get_obs_objs_and_info(self, params, knob_values):
        """
        Get the observation, objectives and info from the Flink job.

        The process:

        - Submit the job and wait for all subtasks to run. If the job fails or the subtasks are not all running within 3
          minutes, return penalty.
        - When all subtasks are running, wait for `job_warmup_sec` seconds.
        - Collect core usage and memory usage from the YARN resource manager.
        - Every `monitor_interval_sec` seconds, collect the throughput and metrics from the Flink job manager. If the
          total number of monitor attempts exceeds `max_monitor_attempts`, or the last monitored results have
          stabilized, stop monitoring.
        - Calculate the observations, objectives and info.

        :param params: job params
        :param knob_values: cluster knob values
        :return: obs, objectives and info
        """
        if self.savepoint_size > 0:
            plan_hash = self.flink_connector.get_execution_plan_id(
                self.flink_job, params, knob_values
            )
            savepoint_dir = self._get_savepoint(plan_hash)
            if savepoint_dir is None:
                self.logger.info("New physical plan %s. Creating savepoint.", plan_hash)
                try:
                    self._create_savepoint(params, knob_values, plan_hash)
                except Exception as e:
                    self.logger.error(
                        "Error occurred during savepoint creation, returning penalty: %s",
                        e,
                        exc_info=True,
                    )
                    return self._get_penalty_obs_objs_and_info()
                savepoint_dir = self._get_savepoint(plan_hash)
            else:
                self.logger.info("Using existing savepoint for plan %s", plan_hash)

            self.logger.info("Savepoint directory: %s", savepoint_dir)
        else:
            plan_hash = self.flink_connector.get_execution_plan_id(
                self.flink_job, params, knob_values
            )
            self.logger.info("physical plan %s.", plan_hash)
            savepoint_dir = None

        # Job submission
        self.flink_connector.reset()
        try:
            plan = self.flink_connector.submit(
                self.flink_job, params, knob_values, self.source_rate, savepoint_dir
            )
        except Exception as e:
            self.logger.error(
                "Error occurred during job submission, returning penalty: %s",
                e,
                exc_info=True,
            )
            return self._get_penalty_obs_objs_and_info()

        attempt_time = 0
        all_subtasks_running = False
        wait_interval = 10

        while attempt_time < self.max_wait_attempts:
            attempt_time += 1
            if self.flink_connector.is_ended():
                self.logger.error(
                    "Job has ended prematurely, returning penalty. Attempt %d/%d",
                    attempt_time,
                    self.max_wait_attempts,
                )
                return self._get_penalty_obs_objs_and_info()
            if self.flink_connector.is_all_subtasks_running():
                all_subtasks_running = True
                break
            self.logger.debug(
                "Attempt %d/%d: Waiting for all subtasks to run. Retrying in %d s...",
                attempt_time,
                self.max_wait_attempts,
                wait_interval,
            )
            time.sleep(wait_interval)

        if not all_subtasks_running:
            self.logger.error(
                "All subtasks are not running after %s seconds, returning penalty.", 180
            )
            return self._get_penalty_obs_objs_and_info()

        total_warmup_sec = 0
        while total_warmup_sec < self.job_warmup_sec:
            if self.flink_connector.is_ended():
                self.logger.error(
                    "Job has ended prematurely during warmup, returning penalty."
                )
                return self._get_penalty_obs_objs_and_info()
            time.sleep(wait_interval)
            total_warmup_sec += wait_interval

        # Get job plan
        try:
            job_plan_hash = get_job_plan_hash(plan["nodes"])
            self.logger.info("Job plan hash: %s", job_plan_hash)
        except:
            self.logger.error(
                "Error occurred during job plan hash generation, returning penalty.",
                exc_info=True,
            )
            return self._get_penalty_obs_objs_and_info()

        # Collect core usage and memory usage
        try:
            core_usage = self.flink_connector.get_core_usage()
            memory_usage = self.flink_connector.get_memory_usage()
        except Exception as e:
            self.logger.error(
                "Error occurred during core/memory usage collection, returning penalty: %s",
                e,
                exc_info=True,
            )
            return self._get_penalty_obs_objs_and_info()

        # Monitor throughput and metrics
        throughput_list: list[list[float]] = []
        metrics_list: list[dict[str, list[MetricStatDict]]] = []
        metrics_stable = False

        for attempt in range(self.max_monitor_attempts):
            if self.flink_connector.is_ended():
                self.logger.warning(
                    "Attempt %s/%s: Job has ended before metrics are stable. Returning penalty.",
                    attempt + 1,
                    self.max_monitor_attempts,
                )
                return self._get_penalty_obs_objs_and_info()

            try:
                metric_names = self.obs_spec.get_metric_names()
                standard_metric_names = set()
                for metric in metric_names:
                    # Extract expressions like "numRecordsInPerSecond / busyTimeMsPerSecond"
                    expression_match = re.search(OBS_EXPRESSION_PATTERN, metric)
                    if expression_match:
                        op_1, op_2 = expression_match.group(1), expression_match.group(
                            3
                        )
                        standard_metric_names.add(op_1)
                        standard_metric_names.add(op_2)
                    else:
                        standard_metric_names.add(metric)
                standard_metric_names = list(standard_metric_names)

                standard_metrics = self.flink_connector.observe_task_metrics(
                    plan, standard_metric_names
                )

                metrics: dict[str, list[MetricStatDict]] = {}
                for vertex_id, vertex_metrics in standard_metrics.items():
                    metrics[vertex_id] = []
                    for metric in metric_names:
                        # Extract expressions like "numRecordsInPerSecond / busyTimeMsPerSecond"
                        expression_match = re.search(OBS_EXPRESSION_PATTERN, metric)
                        if expression_match:
                            op_1, op, op_2 = expression_match.groups()
                            dict_1 = vertex_metrics[op_1]
                            dict_2 = vertex_metrics[op_2]
                            metrics[vertex_id].append(
                                calculate_expression(dict_1, dict_2, op)
                            )
                        else:
                            metrics[vertex_id].append(vertex_metrics[metric])
                self.logger.debug("Collected metrics: %s", metrics)

                throughput = self.flink_connector.get_throughput(plan)
            except Exception as e:
                self.logger.error(
                    "Error occurred during metrics collection: %s", e, exc_info=True
                )
                time.sleep(self.monitor_interval_sec)
                continue

            if not self._check_metrics(metrics, throughput, core_usage, memory_usage):
                time.sleep(self.monitor_interval_sec)
                continue

            throughput_list.append(throughput)
            metrics_list.append(metrics)

            if not self._is_stable(throughput_list):
                self.logger.debug(
                    "Attempt %s/%s: Throughput is not stable: %s",
                    attempt + 1,
                    self.max_monitor_attempts,
                    throughput,
                )
                time.sleep(self.monitor_interval_sec)
                continue

            self.logger.debug(
                "Attempt %s/%s: Throughput is stable: %s",
                attempt + 1,
                self.max_monitor_attempts,
                throughput,
            )
            metrics_stable = True
            break

        if not metrics_stable and len(throughput_list) > 0:
            self.logger.warning(
                "Metrics are not stable after %d attempts. Using unstable metrics anyway.",
                self.max_monitor_attempts,
            )

        if len(throughput_list) > 0:
            counted = min(len(throughput_list), self.stable_window_size)

            throughput_list = throughput_list[-counted:]
            metrics_list = metrics_list[-counted:]

            throughput = np.average(np.sum(throughput_list, axis=1))
            metrics = [self.obs_spec.get_obs(plan, m) for m in metrics_list]
            self.logger.debug("Metrics: %s", metrics)
            # Make sure the adjacency matrix is the same
            adjacency_matrix = metrics[0]["adjacency_matrix"]
            for i in range(1, len(metrics)):
                if not np.array_equal(adjacency_matrix, metrics[i]["adjacency_matrix"]):
                    self.logger.error(
                        "Adjacency matrix is not the same across all metrics"
                    )
                    raise ValueError
            node_features = np.average(
                np.array([m["node_features"] for m in metrics]), axis=0
            )
            obs = {
                "node_features": node_features,
                "adjacency_matrix": adjacency_matrix,
            }

            self.logger.info(
                "Throughput: %.1f, Core usage: %.1f, Memory usage: %.1f MB",
                np.average(throughput),
                core_usage,
                memory_usage,
            )
            objs = [
                self.throughput_weight * np.average(throughput),
                self.resource_weight
                * (
                    2
                    - (
                        core_usage / self.max_core_usage
                        + memory_usage / self.max_memory_usage
                    )
                ),
            ]
            info = {
                "throughput": throughput,
                "core": core_usage,
                "memory": memory_usage,
            }
            return obs, objs, info
        else:
            self.logger.error("No metrics collected, returning penalty.", exc_info=True)
            return self._get_penalty_obs_objs_and_info(core_usage, memory_usage)

    def _is_stable(self, throughput: list[list[float]]):
        return is_throughput_stable(throughput, self.stable_window_size)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsDict, dict[str, Any]]:
        self.current_step = 0
        self.flink_connector.reset()
        obs, objs, info = self._get_penalty_obs_objs_and_info()
        return (
            obs,
            {"objs": np.array(objs), **info},
        )
