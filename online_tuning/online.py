from argparse import ArgumentParser
import time

import numpy as np
from flink.connector import FlinkConnector, FlinkJob
from flink.sql_plan import get_job_plan_hash
from offline_learning.run import get_flink_connector_and_job
from utils.common import is_throughput_stable
from utils.config import RootConfig, load_tuning_root_config
from utils.log_analyzer import read_command_and_metrics
import logging


logger = logging.getLogger("eval")


def should_online_tune(connector: FlinkConnector) -> bool:
    # Tune when backpressure kicks in, or when source rate becomes low
    STABLE_SIZE = 5

    assert connector.is_running()
    plan = connector.get_running_job_plan()
    throughputs = []

    while True:
        time.sleep(10)
        if connector.is_backpressured(plan):
            return True

        t = connector.get_throughput(plan)
        throughputs.append(t)

        if len(throughputs) > STABLE_SIZE:
            if t < np.average(throughputs) * 0.8:
                return True
            throughputs = throughputs[1:]


def online_tune(
    command: str,
    savepoint_dir: str,
    connector: FlinkConnector,
    job: FlinkJob,
    target_throughput: float,
    conf: RootConfig,
):
    parts = command.split()
    for i in range(len(parts) - 1):
        if parts[i] == "--event-rates":
            parts[i + 1] = str(target_throughput)

    command = " ".join(parts)
    # print(f"Submit command: {command}")
    connector.reset()
    plan = connector.submit_and_wait_until_stable(
        job=job,
        params="",
        knob_values={},
        source_rate=int(target_throughput),
        config=conf,
        exact_command=command,
        savepoint_dir=savepoint_dir,
    )

    # Get job plan
    try:
        job_plan_hash = get_job_plan_hash(plan["nodes"])
        logger.info("Job plan hash: %s", job_plan_hash)
    except:
        logger.error(
            "Error occurred during job plan hash generation.",
            exc_info=True,
        )
        return

    # Collect core usage and memory usage
    try:
        core_usage = connector.get_core_usage()
        memory_usage = connector.get_memory_usage()
        throughput = connector.get_throughput(plan)

    except Exception as e:
        logger.error(
            "Error occurred during throughput/core/memory usage collection, exiting: %s",
            e,
            exc_info=True,
        )
        return

    logger.info(
        "Throughput: %.1f, Core usage: %.1f, Memory usage: %.1f MB",
        np.average(throughput),
        core_usage,
        memory_usage,
    )


def get_best_command(results: list[tuple[str, float, float, float]], thr_target: float, conf: RootConfig) -> tuple[str, dict]:
    best_command = ""
    best_score = float("inf")
    best_metrics = {}

    average_cpu = np.average([r[2] for r in results])
    average_memory = np.average([r[3] for r in results])    

    for command, throughput, core_usage, memory_usage in results:
        # Normalize core and memory usage
        norm_core = core_usage / average_cpu  # Assuming max 16 cores
        norm_memory = memory_usage / average_memory  # Assuming max 64GB memory

        score = norm_core + norm_memory  # smaller -> better

        # Multiplied by 1.05 because we want to leave some processing capability
        # for catching up
        if float(throughput) > thr_target * 1.05 and score < best_score:
            best_score = score
            best_command = command
            best_metrics = {
                "throughput": throughput,
                "core_usage": core_usage,
                "memory_usage": memory_usage,
            }

    return best_command, best_metrics

def main(conf: RootConfig):
    conf.env.stable_window_size = 12

    connector, job = get_flink_connector_and_job(conf)

    results = read_command_and_metrics(
        f"/home/User/code/stream-tuning/saved-results/{conf.tuner.load_dir}/log.txt"
    )

    # Run with default params
    connector.submit_and_wait_until_stable(
        job=job,
        params="",
        knob_values={},
        source_rate=",".join([str(r) for r in conf.job.online_source_rates]),
        savepoint_dir=None,
        config=conf,
    )

    while should_online_tune(connector):
        throughput = connector.get_throughput(connector.get_running_job_plan())
        command, metrics = get_best_command(results, float(np.average(throughput)), conf)

        savepoint_dir = connector.stop(with_savepoint=True)
        connector.submit_and_wait_until_stable(
            job=job,
            params="",
            knob_values={},
            source_rate=",".join([str(r) for r in conf.job.online_source_rates]),
            savepoint_dir=savepoint_dir,
            config=conf,
            exact_command=command,
        )
