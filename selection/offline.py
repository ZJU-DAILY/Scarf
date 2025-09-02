import logging
import random
import socket
from logging import Logger
from typing import Any

import numpy as np
import torch

from environment.flink_env import FlinkEnv
from flink.connector import FlinkJob
from flink.knob import parse_knob_def
from flink.sql_plan import SavepointStore
from flink.yarn_connector import YarnFlinkConnector
from offline_learning.run import get_flink_connector_and_job
from selection.sampler import KnobValueSampler, LHSKnobValueSampler
from utils.config import (
    RootConfig,
    load_full_knob_root_config,
)
import gymnasium as gym

import os
import json

DEFAULT_PARALLELISM = "8"

logger = logging.getLogger("selection")


def load_observations(load_dir: str | None) -> Any:
    if load_dir is not None:
        if os.path.exists(load_dir) and os.path.isdir(load_dir):
            filename = os.path.join(load_dir, "observations.json")
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    observations = json.load(f)
                    return observations
    return None


def sample_observation(
    connector: YarnFlinkConnector,
    job: FlinkJob,
    knob_values: dict[str, Any],
    conf: RootConfig,
    source_rate: int,
):
    connector.reset()
    connector.reset_base_time()
    plan = connector.submit_and_wait_until_stable(
        job, "", knob_values, source_rate, None, conf
    )
    throughput = connector.get_throughput(plan)
    core_usage = connector.get_core_usage()
    memory_usage = connector.get_memory_usage()

    logger.info(
        "Throughput: %.1f, Core usage: %.1f, Memory usage: %.1f MB",
        np.average(throughput),
        core_usage,
        memory_usage,
    )

    sample = (knob_values, float(np.average(throughput)), float(core_usage), float(memory_usage))
    return sample


def sample_observations(
    conf: RootConfig,
    knob_sampler: KnobValueSampler,
    logger: Logger,
    use_default_parallelism: bool = True,
):
    knob_value_set = knob_sampler.generate()
    connector: YarnFlinkConnector
    connector, job = get_flink_connector_and_job(conf)  # type: ignore
    offline_rate = conf.job.offline_source_rate
    if use_default_parallelism:
        connector.default_cluster_knobs["parallelism.default"] = DEFAULT_PARALLELISM

    valid_sample_count = 0
    # list of (knob_values, throughput, core, memory)
    observations: list[tuple[dict[str, Any], float, float, float]] = []
    total_trials = 0
    for knob_values in knob_value_set:
        logger.info(
            "Trial %d, next sample to collect: %d", total_trials, valid_sample_count + 1
        )
        logger.info("Knob values: %s", knob_values)
        source_rate = offline_rate
        try:
            sample = sample_observation(
                connector,
                job,
                knob_values,
                conf,
                source_rate
            )
            observations.append(sample)
            valid_sample_count += 1
        except Exception as e:
            logger.error(f"Error submitting job: {e}", exc_info=True)
            throughput = 0.0
            core_usage = -1
            memory_usage = -1
            sample = (knob_values, throughput, core_usage, memory_usage)
            logger.info("Sample %d: %s", total_trials + 1, sample)
            observations.append(sample)
        total_trials += 1

    logger.info(
        "Collected %d observations, %d valid", len(observations), valid_sample_count
    )
    return observations


def save_observations(observations: Any, save_dir: str):
    """
    Save observations to a JSON file.
    :param observations: list of tuples (knob values, source rate, throughput)
    :param save_dir: directory to save observations to
    """
    import json
    import os

    def custom_ser(obj):
        return str(obj)

    with open(os.path.join(save_dir, "observations.json"), "w") as f:
        json.dump(observations, f, default=custom_ser)


def run(conf: RootConfig):

    offline_rate = conf.job.offline_source_rate

    knob_selection_conf = load_full_knob_root_config(
        "/home/User/code/stream-tuning/flink-tuner/config/full_params.yaml"
    )
    full_knobs = parse_knob_def(
        knob_selection_conf.knobs, conf.knobs.excluded_cluster_knob_prefixes
    )
    num_samples = knob_selection_conf.num_samples
    logger.info("Knob definitions: %s", full_knobs)
    logger.info("%d knobs parsed", len(full_knobs))
    logger.info("%d samples to collect", num_samples)

    sampler = LHSKnobValueSampler(full_knobs, num_samples, True)

    observations = sample_observations(conf, sampler, logger)
    save_observations(observations, conf.tuner.save_dir)
