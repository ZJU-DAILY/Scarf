import logging
import os
import socket
import sys

from argparse import ArgumentParser
from datetime import datetime
from logging import Formatter

import numpy as np
import torch

from environment.flink_env import ActionSpec, ObsSpec, register_flink_env
from flink.knob import parse_knob_def
from flink.metrics import parse_included_stats, parse_metric_def
from flink.mock_connector import MockFlinkConnector
from offline_learning.run import get_flink_connector_and_job
from scripts.plot import training_visualize_2d
from selection.offline import load_observations
from utils.config import (
    RootConfig,
    load_full_knob_root_config,
    load_tuning_root_config,
    save_to_yaml,
)


def init_loggers(conf: RootConfig):

    loggers = conf.log.levels.__dict__
    root_level = (None,)
    file_handler = logging.FileHandler(os.path.join(conf.tuner.save_dir, "log.txt"))
    file_handler.setFormatter(
        Formatter("%(asctime)s %(name)s [%(levelname)8s] %(message)s")
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(Formatter("%(asctime)s [%(levelname)8s] %(message)s"))

    for name, level in loggers.items():
        if name != "root":
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addHandler(file_handler)
            logger.addHandler(stdout_handler)
            logger.propagate = False
        else:
            root_level = level

    logging.basicConfig(
        level=root_level,
        handlers=[
            file_handler,
            stdout_handler,
        ],
    )


def register_environment(conf: RootConfig):
    if conf.mode.env.startswith("FlinkEnv"):
        # Prepare actions
        operator_names = conf.knobs.operator_names
        operator_knobs = parse_knob_def(conf.knobs.operator_knobs)
        cluster_knobs = parse_knob_def(conf.knobs.cluster_knobs)

        action_spec = ActionSpec(
            operator_names=operator_names,
            operator_knobs=operator_knobs,
            cluster_knobs=cluster_knobs,
        )

        # Prepare observations
        metric_def = parse_metric_def(conf.metrics.observed)
        included_stats = parse_included_stats(conf.metrics.included_stats)
        obs_spec = ObsSpec(metric_def, included_stats)

        flink_connector, flink_job = get_flink_connector_and_job(conf)
        flink_connector.logger.setLevel(conf.log.levels.connector)
        if conf.mode.connector == "mock":
            operator_names = conf.knobs.operator_names
            if not operator_names:
                # SQL jobs
                operator_names = ["all"]
            flink_connector = MockFlinkConnector(operator_names)
            job_warmup_sec = 0
            monitor_interval_sec = 0
        elif conf.mode.connector == "yarn":
            job_warmup_sec = conf.env.job_warmup_sec
            monitor_interval_sec = conf.env.monitor_interval_sec
        else:
            raise ValueError(f"Unknown connector: {conf.mode.connector}")

        register_flink_env(
            {
                "log_level": conf.log.levels.env,
                "action_spec": action_spec,
                "obs_spec": obs_spec,
                "flink_connector": flink_connector,
                "flink_job": flink_job,
                "job_warmup_sec": job_warmup_sec,
                "monitor_interval_sec": monitor_interval_sec,
                "max_wait_attempts": conf.env.max_wait_attempts,
                "max_monitor_attempts": conf.env.max_monitor_attempts,
                "stable_window_size": conf.env.stable_window_size,
                "throughput_weight": conf.env.throughput_weight,
                "resource_weight": conf.env.resource_weight,
                "max_core_usage": conf.env.max_core_usage,
                "max_memory_usage": conf.env.max_memory_m_bytes,
                "savepoint_size": conf.env.savepoint_size,
                "savepoint_store": conf.env.savepoint_store,
                "source_rate": conf.job.offline_source_rate,
            }
        )
    else:
        raise ValueError(f"Unknown environment: {conf.mode.env}")


if __name__ == "__main__":

    # Set print options
    # Do not use scientific mode
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
    torch.set_default_dtype(torch.float64)

    parser = ArgumentParser()
    # selection, offline, online
    parser.add_argument(
        "-o", "--mode", type=str, default="FlinkEnv", help="Mode to use"
    ) 
    # coldstart, analysis, cluster
    parser.add_argument("-s", "--stage", type=str, default="coldstart", help="stage")
    parser.add_argument("-c", "--config", type=str, help="Path to config file")
    parser.add_argument(
        "-l",
        "--load-dir",
        default="",
        type=str,
        help="Path to load directory. This will override tuner.load_dir in the config file.",
    )
    args, _ = parser.parse_known_args()

    conf_file = args.config if args.config else f"config/{socket.gethostname()}.yaml"
    print(f"Using config file: {conf_file}")
    conf = load_tuning_root_config(conf_file)

    if args.load_dir:
        conf.tuner.load_dir = args.load_dir
    if conf.tuner.load_dir and not os.path.exists(conf.tuner.load_dir):
        raise FileNotFoundError(f"Load directory {args.load_dir} does not exist.")

    # build saving folder
    conf.tuner.save_dir = os.path.join(
        conf.tuner.save_dir, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(conf.tuner.save_dir, exist_ok=True)
    os.makedirs(os.path.join(conf.tuner.save_dir, "log"), exist_ok=True)
    os.makedirs(os.path.join(conf.tuner.save_dir, "tensorboard"), exist_ok=True)
    # save arguments
    save_to_yaml(conf, os.path.join(conf.tuner.save_dir, "args.txt"))
    conf.env.savepoint_store = os.path.join(conf.tuner.save_dir, "savepoint_store.db")

    init_loggers(conf)
    register_environment(conf)

    if args.mode == "selection":
        if args.stage == "coldstart":
            from selection import offline

            offline.run(conf)
        elif args.stage == "analysis":
            from selection import selector

            selector.run(conf)
        elif args.stage == "cluster":
            from selection import speedup

            speedup.main(conf)
    elif args.mode == "offline":
        from offline_learning import run

        run.main(conf)
    elif args.mode == "online":
        from online_tuning import online

        online.main(conf)
