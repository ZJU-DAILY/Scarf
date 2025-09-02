import logging
import socket
import time
from pprint import pprint
from unittest import TestCase

from flink.yarn_connector import YarnFlinkConnector
from flink.knob import parse_default_knob_def, parse_knob_def
from offline_learning.run import get_flink_connector_and_job
from utils.config import load_tuning_root_config


class TestYarnFlinkConnector(TestCase):

    def test_connector(self):

        # --- Configuration ---

        conf_file = f"config/{socket.gethostname()}.yaml"
        conf = load_tuning_root_config(conf_file)
        connector, my_job = get_flink_connector_and_job(conf)
        job_params = conf.job.default_job_params
        knob_defs = parse_knob_def(conf.knobs.cluster_knobs)
        knobs = {}
        for knob_def in knob_defs:
            knobs[knob_def.name] = knob_def.format_value(knob_def.default)
        default_knobs = parse_default_knob_def(conf.flink.default_cluster_knobs)
        knobs.update(default_knobs)

        try:
            connector.reset()
            # --- Submit Job ---
            print("\nSubmitting Flink job...")
            initial_plan = connector.submit(
                job=my_job, params=job_params, knob_values=knobs
            )
            print(
                f"\nJob submitted successfully. YARN App ID: {connector._current_yarn_app_id}"
            )
            print(f"Flink Job ID: {connector._current_job_id}")
            print(
                f"Initial Job Plan Nodes: {[node.get('id') for node in initial_plan.get('nodes', [])]}"
            )

            def check():
                # --- Check Job Status ---
                print("\nChecking job status...")
                running = connector.is_running()
                print("Job is running:", running)
                if not running:
                    raise RuntimeError("Job is not running.")

                # --- Monitor Job ---
                print("\nChecking resource usage...")
                cores = connector.get_core_usage()
                memory = connector.get_memory_usage()
                print(f"Allocated vCores: {cores}")
                print(f"Allocated Memory (MB): {memory}")

                print("\nChecking throughput...")
                throughputs = connector.get_throughput(initial_plan)
                if throughputs:
                    print(f"Source Throughput (records/sec per subtask): {throughputs}")
                    print(
                        f"Aggregate Source Throughput: {sum(throughputs):.2f} records/sec"
                    )
                else:
                    print("Could not retrieve throughput metrics.")

                print("\nObserving specific metrics...")
                observed_metrics = [metric.name for metric in conf.metrics.observed]
                # Note: Metric names depend on your Flink version and job
                busy_time_metrics = connector.observe_task_metrics(
                    initial_plan, observed_metrics
                )
                print("Metrics per vertex:")
                for vertex_id, values in busy_time_metrics.items():
                    print(f"  Vertex {vertex_id}: {[value['avg'] for value in values]}")

            for i in range(60):
                print(f"Round {i+1}")
                try:
                    check()
                except Exception as e:
                    print(f"Error during monitoring: {e}")
                time.sleep(10)

            print("\nStopping task in 60 s...")
            time.sleep(60)  # Allow some time before stopping

            # --- Stop Job with Savepoint ---
            print("\nStopping job with savepoint...")
            savepoint_path = connector.stop(with_savepoint=True)

            if savepoint_path:
                print(
                    f"\nJob stopped successfully. Savepoint created at: {savepoint_path}"
                )
                # Example: You could now potentially resubmit using this savepoint_path
                # print("\nResubmitting from savepoint (example)...")
                # connector.submit(job=my_job, params=JOB_PARAMS, knobs=job_knobs, savepoint_dir=savepoint_path)
                # print("Job resubmitted.")
            else:
                print("\nFailed to stop job with savepoint or confirm path.")

        except FileNotFoundError as e:
            print(f"\nError: {e}. Please check FLINK_HOME and JOB_JAR paths.")
        except (RuntimeError, ValueError, ConnectionError) as e:
            print(f"\nAn error occurred: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

        finally:
            # Example cleanup: If the script fails mid-run, you might want to kill the YARN app manually
            if connector._current_yarn_app_id:
                print(
                    f"\nWarning: Script finished, but a YARN application might still be running: {connector._current_yarn_app_id}"
                )
                print(
                    f"You may need to kill it manually: yarn application -kill {connector._current_yarn_app_id}"
                )

    def test_metrics(self):
        FLINK_RUN = "/home/User/code/stream-tuning/flink-2.0-preview1"
        connector = YarnFlinkConnector(
            flink_home=FLINK_RUN,
            hadoop_home="",
            yarn_rm_http_address="http://not.used",
            savepoint_dir="",
        )
        connector._current_flink_rest_url = "http://node11:4978"
        current_job_id = connector._find_running_job_id()
        if not current_job_id:
            print("No running job found.")
            return
        connector._current_job_id = current_job_id
        print("Current Job ID:", current_job_id)

        running = connector.is_all_subtasks_running()
        print("Running? " + str(running))

        plan = connector.get_running_job_plan()
        print("Job plan:")
        pprint(plan)

        busy_time_metrics = connector.observe_task_metrics(
            plan, ["busyTimeMsPerSecond"]
        )
        print("Busy Time Metrics (ms/sec per subtask per vertex):")
        for vertex_id, values in busy_time_metrics.items():
            print(f"  Vertex {vertex_id}: {values}")

        throughput = connector.get_throughput(plan)
        print(f"Throughput: {throughput}")

    def test_plan(self):

        # --- Configuration ---

        conf_file = f"config/{socket.gethostname()}.yaml"
        conf = load_tuning_root_config(conf_file)
        connector, my_job = get_flink_connector_and_job(conf)
        job_params = conf.job.default_job_params
        knob_defs = parse_knob_def(conf.knobs.cluster_knobs)
        knobs = {}
        for knob_def in knob_defs:
            knobs[knob_def.name] = knob_def.format_value(knob_def.default)
        default_knobs = parse_default_knob_def(conf.flink.default_cluster_knobs)
        knobs.update(default_knobs)

        plan = connector.get_execution_plan_id(
            job=my_job, params=job_params, knob_values=knobs
        )
        print(plan)
