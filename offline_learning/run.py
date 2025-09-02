import logging
import os
import socket
import sys
from datetime import datetime
from logging import Formatter

import numpy as np
import torch

from environment.flink_env import ActionSpec, ObsSpec, register_flink_env
from flink.connector import FlinkConnector, FlinkJob
from flink.sql_plan import SavepointStore
from flink.yarn_connector import YarnFlinkConnector
from flink.mock_connector import MockFlinkConnector
from flink.knob import parse_default_knob_def, parse_knob_def
from flink.metrics import parse_included_stats, parse_metric_def
from offline_learning import morl, sorl
from utils.config import RootConfig, load_tuning_root_config, save_to_yaml


def get_flink_connector_and_job(
    conf: RootConfig,
) -> tuple[FlinkConnector, FlinkJob]:
    # Prepare Flink job
    flink_job = FlinkJob(
        conf.job.jar_path,
        conf.job.main_class,
        conf.job.job_name,
        conf.job.default_job_params,
        conf.job.base_time_param_name,
        conf.job.rate_param_name,
    )

    # Prepare Flink connector
    if conf.mode.connector == "mock":
        operators = conf.knobs.operator_names
        if not operators:
            # SQL jobs
            operators = ["all"]
        flink_connector = MockFlinkConnector(operators)
    else:
        default_cluster_knobs = parse_default_knob_def(conf.flink.default_cluster_knobs)
        flink_connector = YarnFlinkConnector(
            conf.flink.flink_home,
            conf.flink.hadoop_home,
            conf.flink.yarn_rm_http_address,
            conf.flink.savepoint_dir,
            default_cluster_knobs,
        )

    return flink_connector, flink_job


def main(conf: RootConfig):
    if conf.mode.algo == "sorl":
        sorl.run(conf)
    elif conf.mode.algo == "morl":
        morl.run(conf)
    else:
        raise ValueError("Unsupported algorithm " + conf.mode.algo)
