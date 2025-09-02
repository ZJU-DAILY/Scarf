import json
import logging
import os
import re
import subprocess
import threading
import time
from functools import wraps
from typing import Any, Optional
import uuid

import numpy as np
import requests

from flink.connector import FlinkConnector, FlinkJob
from utils.common import get_sink_node_ids, get_source_node_ids
from utils.config import RootConfig
from utils.types import JobPlanDict, MetricStatDict, VertexDict


FLINK_SAVEPOINT_PATH = "/flink/savepoints"
FLINK_JAR_UPLOAD_PATH = "/user/User/.flink"


class CriticalError(Exception):
    """Custom exception for critical errors in Flink connector."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class InvalidStateError(Exception):
    """Custom exception for invalid state in Flink connector."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def check_valid_state(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_in_valid_state", False):
            raise InvalidStateError("Object is not in a valid state!")
        try:
            return method(self, *args, **kwargs)
        except CriticalError:
            self._in_valid_state = False
            raise

    return wrapper


class _FlinkJobMonitor:

    def __init__(self, connector: "YarnFlinkConnector", stop_event: threading.Event):
        """
        Initializes the FlinkJobMonitor with a FlinkConnector instance.

        Args:
            connector: An instance of a FlinkConnector.
        """
        self._connector = connector
        self._stop_event = stop_event

    def run(self):
        # Monitor throughput and watermark lag every 5 seconds
        while not self._stop_event.is_set():
            try:
                if not self._connector.is_running():
                    self._connector.logger.info(
                        "Monitor:  is not running, stopping monitor."
                    )
                    break

                plan = self._connector.get_running_job_plan()
                throughput = self._connector.get_throughput(plan)
                watermark = self._connector.get_sink_watermark(plan)
                total_processed = self._connector.get_total_processed(plan)
                self._connector.logger.info(
                    "Monitor: ts throughput watermark total_processed: %d %.1f %d %d",
                    int(time.time() * 1000),
                    np.average(throughput),
                    min(watermark),
                    total_processed,
                )
            except Exception as e:
                pass
                # self._connector.logger.error("Error in job monitor: %s", e)

            if self._stop_event.wait(timeout=5):
                break
        self._connector.logger.info("Monitor: exiting.")


class YarnFlinkConnector(FlinkConnector):
    """
    Interacts with a Flink cluster deployed on YARN via yarn-application mode.
    Handles job submission, monitoring, resource checking, and stopping jobs.
    """

    logger = logging.getLogger("connector")

    def __init__(
        self,
        flink_home: str,
        hadoop_home: str,
        yarn_rm_http_address: str,
        savepoint_dir: str,
        default_cluster_knobs: dict[str, str],
    ):
        """
        Initializes the connector.

        Args:
            flink_home: Flink home.
            yarn_rm_http_address: HTTP address of the YARN ResourceManager.
            default_knobs: Default configuration knobs for Flink jobs.
        """
        if not yarn_rm_http_address.startswith(("http://", "https://")):
            raise ValueError(
                "yarn_rm_http_address must include protocol (http:// or https://)"
            )
        self.flink_home: str = flink_home
        self.hadoop_home: str = hadoop_home
        self.yarn_rm_http_address: str = yarn_rm_http_address.rstrip("/")
        self.default_cluster_knobs: dict[str, str] = default_cluster_knobs or {}
        self.savepoint_dir: str = savepoint_dir

        self._current_yarn_app_id: Optional[str] = None
        self._current_flink_rest_url: Optional[str] = None
        self._current_job_id: Optional[str] = None
        self._current_monitor_stop_event: Optional[threading.Event] = None

        # This base time is used throughout the lifetime of this connector
        self._base_time: Optional[int] = None

        # Flag to indicate if the connector is in a bad state.
        self._in_valid_state = True

    def _run_command(self, command: list[str]) -> str:
        """Executes a shell command and returns stdout."""
        try:
            # self.logger.debug("Running command: %s", " ".join(command))
            # Use list of strings for command to avoid shell injection issues
            result = subprocess.run(
                ["/bin/bash", "-c", " ".join(command)],
                capture_output=True,
                text=True,
                env=os.environ,
                check=True,
            )
            # self.logger.debug("Command stdout: %s", result.stdout)
            # self.logger.debug("Command stderr: %s", result.stderr)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error("Command failed: %s", e)
            self.logger.error("Stderr: %s", e.stderr)
            self.logger.error("Stdout: %s", e.stdout)
            raise RuntimeError(f"Command execution failed: {e}") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"Flink executable not found at: {self.flink_home}")

    @staticmethod
    def _make_request(
        url: str,
        method: str = "GET",
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Makes an HTTP request and returns the JSON response."""
        try:
            response = requests.request(
                method, url, params=params, json=data, timeout=30
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            if response.content:
                return response.json()
            return None  # Handle cases like 202 Accepted with no body
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API request failed for {method} {url}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to decode JSON response from {url}: {e}\nResponse text: {response.text}"
            ) from e

    def _find_running_job_id(
        self, job_name: Optional[str] = None, max_retries: int = 10, delay: float = 10.0
    ) -> Optional[str]:
        """Polls the Flink REST API to find the ID of the running job."""
        if not self._current_flink_rest_url:
            return None

        url = f"{self._current_flink_rest_url}/jobs/overview"
        for attempt in range(max_retries):
            try:
                response = self._make_request(url)
                if response and "jobs" in response:
                    running_jobs = [
                        job
                        for job in response["jobs"]
                        if job.get("state") == "RUNNING"
                        and (not job_name or job.get("name") == job_name)
                    ]
                    if running_jobs:
                        if not job_name and len(running_jobs) > 1:
                            self.logger.warning(
                                "Found multiple RUNNING jobs %s: %s. Selecting the first one.",
                                (
                                    "matching job name"
                                    if job_name
                                    else "without providing a job name"
                                ),
                                [f"{j['jid']} ({j['name']})" for j in running_jobs],
                            )

                        return running_jobs[0]["jid"]
                    else:
                        self.logger.warning(
                            "Attempt %d/%d: No RUNNING jobs found%s. All jobs: %s. Retrying in %.0fs...",
                            attempt + 1,
                            max_retries,
                            " that matches the provided job name" if job_name else "",
                            [f"{j['jid']} ({j['name']})" for j in response["jobs"]],
                            delay,
                        )
            except ConnectionError as e:
                self.logger.error(
                    "Attempt %d/%d: Could not connect to Flink REST API (%s): %s. The job may have exited.",
                    attempt + 1,
                    max_retries,
                    url,
                    e,
                )
                return None
            except Exception as e:  # Catch other potential errors during parsing/access
                self.logger.error(
                    "Attempt %d/%d: Error fetching job overview: %s. Retrying in %.0fs...",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )

            time.sleep(delay)

        self.logger.error(
            "Could not find a RUNNING Flink job after %d attempts.", max_retries
        )
        return None

    @check_valid_state
    def get_execution_plan_id(
        self, job: FlinkJob, params: str, knob_values: dict[str, str]
    ) -> str:
        plan_related_knob_prefixes = ("pipeline.operator-chaining", "table.optimizer")
        values = [
            str(knob_values.get(k, None))
            for k in knob_values
            if k.startswith(plan_related_knob_prefixes)
        ]
        # hash
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, ",".join(values)))

    @check_valid_state
    def submit(
        self,
        job: FlinkJob,
        params: Optional[str],
        knob_values: dict[str, str],
        source_rate: int | str,
        savepoint_dir: Optional[str] = None,
        exact_command: Optional[str] = None,
    ) -> JobPlanDict:
        """
        Submits the FlinkJob to YARN in yarn-application mode.

        Args:
            job: The FlinkJob instance.
            params: Command-line parameters for the Flink job (excluding default ones)
            knob_values: List of dictionaries for Flink configuration (-D key=value) (excluding default ones)F
            source_rate: The source rate for the job.
            savepoint_dir: Optional path to restore from a savepoint.

        Returns:
            The job plan dictionary obtained from get_job_plan().

        Raises:
            RuntimeError: If submission fails or required IDs cannot be extracted.
            FileNotFoundError: If flink executable is not found.
            ConnectionError: If API requests fail.
            ValueError: If API responses are invalid or job ID is not found.
        """
        if self._current_yarn_app_id:
            raise RuntimeError(
                f"Another job ({self._current_yarn_app_id}) is already managed by this connector."
            )

        if not exact_command:
            command = [
                os.path.join(self.flink_home, "bin", "flink"),
                "run",
                "-t",
                "yarn-application",
                "-d",  # Run detached
            ]

            # Add default knobs
            for key, value in self.default_cluster_knobs.items():
                command.extend([f'"-D{key}={value}"'])

            # Add configuration knob_values
            for key, value in knob_values.items():
                command.extend([f'"-D{key}={value}"'])

            # Add JAR, main class, and parameters
            if job.main_class:
                command.extend(["-c", job.main_class])
            command.append(job.jar_path)
            if job.default_job_params:
                command.extend(job.default_job_params.split())
            if params:
                command.extend(params.split())

            # Add base time
            # if not self._base_time:
            #     self._base_time = int((time.time() + 5) * 1000) # 30s for job start
            self._base_time = 0
            command.extend([f"--{job.base_time_param_name}", str(self._base_time)])
            # Add source rate
            command.extend([f"--{job.rate_param_name}", str(source_rate)])
        else:
            # Only for debugging.
            command = exact_command.split()
        # Add savepoint if provided
        if savepoint_dir:
            command.extend(["-s", savepoint_dir])

        self.logger.debug("Submit command: " + " ".join(command))

        stdout = self._run_command(command)

        # Extract YARN Application ID
        yarn_match = re.search(r"Submitted application (application_\d+_\d+)", stdout)
        if not yarn_match:
            raise RuntimeError(
                "Could not extract YARN Application ID from Flink output."
            )
        self._current_yarn_app_id = yarn_match.group(1)
        self.logger.debug(
            "Extracted YARN Application ID: %s", self._current_yarn_app_id
        )

        # Extract Flink Web Interface URL
        # Handle potential variations in output format
        web_match = re.search(r"(?:Found Web Interface) ([^ \n]+)", stdout)

        if not web_match:
            # Fallback regex if the first one fails
            web_match = re.search(r"Web Interface:?\s+(https?://[^ \n]+)", stdout)

        if not web_match:
            self.logger.warning(
                "Could not automatically extract Flink Web Interface URL from output."
            )
            # Attempt to guess based on YARN app ID (less reliable)
            self.logger.warning(
                "Attempting to guess Flink REST URL based on YARN app ID."
            )
            self._current_flink_rest_url = (
                f"{self.yarn_rm_http_address}/proxy/{self._current_yarn_app_id}/"
            )

        else:
            raw_url = web_match.group(1)
            # Ensure protocol is present
            if not raw_url.startswith(("http://", "https://")):
                # Try common default http, but this might be wrong for https setups
                self._current_flink_rest_url = f"http://{raw_url}"
            else:
                self._current_flink_rest_url = raw_url
            self.logger.debug(
                f"Extracted Flink REST URL: {self._current_flink_rest_url}"
            )
            # Add API version
            self._current_flink_rest_url = (
                self._current_flink_rest_url.rstrip("/") + "/v1"
            )

        # Find the Flink Job ID using the REST API
        # self.logger.info("Waiting for Flink job to appear in RUNNING state...")
        time.sleep(10)  # Wait for the job to start
        self._current_job_id = self._find_running_job_id(job.name)
        if not self._current_job_id:
            self._current_yarn_app_id = None  # Reset state as we failed
            self._current_flink_rest_url = None
            raise ValueError(
                "Failed to find the Flink Job ID for the submitted application."
            )
        self.logger.debug("Found Flink Job ID: %s", self._current_job_id)

        # Get and return the job plan
        try:
            plan = self.get_running_job_plan()
            source_node_ids = get_source_node_ids(plan)
            self.logger.debug("Source node IDs: %s", source_node_ids)
        except Exception as e:
            # If getting the plan fails immediately after finding the job ID,
            # it might indicate a rapid job failure.
            self.logger.warning(
                "Successfully submitted job but failed to get initial plan: %s", e
            )
            # Decide whether to keep state or clear it. Keeping it might allow later inspection.
            # For now, let's keep the state but return an empty dict or re-raise.
            # Re-raising might be cleaner to signal the problem.
            raise RuntimeError(
                f"Job submitted ({self._current_job_id}) but failed to retrieve initial plan."
            ) from e

        # Start monitor
        self._current_monitor_stop_event = threading.Event()
        monitor = _FlinkJobMonitor(self, self._current_monitor_stop_event)
        monitor_thread = threading.Thread(target=monitor.run, daemon=True)
        monitor_thread.start()

        return plan

    def submit_and_wait_until_stable(
        self,
        job: FlinkJob,
        params: str,
        knob_values: dict[str, str],
        source_rate: int | str,  # Can be a comma delimited string for multiple rates
        savepoint_dir: Optional[str],
        config: RootConfig,
        exact_command: Optional[str] = None,
    ) -> JobPlanDict:
        """
        Submits the Flink job and waits until all subtasks are running and throughput is stable.

        Args:
            job: The FlinkJob instance.
            params: Command-line parameters for the Flink job (excluding default ones)
            knob_values: Flink configuration (-Dkey=value) (excluding default ones)
            wait_times: Max attempts to wait for the job to run.
            savepoint_dir: The savepoint to recover from, None if no savepoint.
            config: RootConfig

        Returns:
            The job plan dictionary.

        Raises:
            ValueError: If the job ends prematurely or does not become stable.
        """
        from utils.common import is_throughput_stable

        wait_times = config.env.max_wait_attempts
        wait_interval = config.env.monitor_interval_sec
        warmup_sec = config.env.job_warmup_sec
        check_stable_times = config.env.max_monitor_attempts
        check_stable_interval = config.env.monitor_interval_sec
        stable_window = config.env.stable_window_size

        plan = self.submit(
            job,
            params,
            knob_values,
            source_rate,
            savepoint_dir=savepoint_dir,
            exact_command=exact_command,
        )
        self.logger.debug("Job submitted. jid: %s", plan["jid"])

        attempt_time = 0
        all_subtasks_running = False
        while attempt_time < wait_times:
            attempt_time += 1
            if self.is_ended():
                raise ValueError("Job ends prematurely.")
            if self.is_all_subtasks_running():
                all_subtasks_running = True
                break
            self.logger.debug(
                "Attempt %d/%d: Waiting for all subtasks to run. Retrying in %d s...",
                attempt_time,
                wait_times,
                wait_interval,
            )
            time.sleep(wait_interval)

        if not all_subtasks_running:
            raise ValueError("Not all subtasks are running after max wait attempts.")

        # Wait for job warmup
        time.sleep(warmup_sec)
        if self.is_ended():
            raise ValueError("Job ends prematurely.")

        # Wait for job to be stable
        throughputs = []
        for t in range(check_stable_times):
            throughput = self.get_throughput(plan)
            throughputs.append(throughput)
            if (
                is_throughput_stable(throughputs, stable_window)
                and np.average(throughput) < 1.05 * source_rate
            ):
                self.logger.info(
                    "Attempt %d/%d: Throughput %s is stable.",
                    t + 1,
                    check_stable_times,
                    throughput,
                )
                break
            self.logger.debug(
                "Attempt %d/%d: Throughput %s is not stable.",
                t + 1,
                check_stable_times,
                throughput,
            )
            time.sleep(check_stable_interval)

        return plan

    def _get_job_status(self) -> Optional[str]:
        """
        Retrieves the status of the current Flink job.

        Returns:
            The status of the job as a string.
        """
        if not self._current_job_id or not self._current_flink_rest_url:
            raise ValueError

        url = f"{self._current_flink_rest_url}/jobs/{self._current_job_id}/status"
        response = self._make_request(url)
        if response and isinstance(response, dict):
            return response.get("status", None)
        else:
            self.logger.error("Unexpected response format for job status: %s", response)
            return None

    def is_running(self) -> bool:
        """
        Checks if the current Flink job is running.

        Returns:
            True if the job is running, False otherwise.
        """
        if not self._current_job_id or not self._current_flink_rest_url:
            return False
        try:
            return self._get_job_status() == "RUNNING"
        except Exception as e:
            self.logger.error("Error checking job status: %s", e)
            return False

    def is_ended(self) -> bool:
        """
        Checks if the current Flink job has ended.

        Returns:
            True if the job has ended, False otherwise.
        """
        if not self._current_job_id or not self._current_flink_rest_url:
            raise ValueError
        try:
            return self._get_job_status() in [
                None,
                "FAILED",
                "CANCELED",
                "FAILING",
                "CANCELLING",
                "FINISHED",
            ]
        except Exception as e:
            self.logger.error("Error checking job status: %s", e)
            return True

    def get_vertices(self) -> list[VertexDict]:
        if not self._current_job_id or not self._current_flink_rest_url:
            raise ValueError("No active Flink job is being tracked by this connector.")

        url = f"{self._current_flink_rest_url}/jobs/{self._current_job_id}"
        response = self._make_request(url)
        if (
            response
            and "vertices" in response
            and isinstance(response["vertices"], list)
        ):
            return response["vertices"]
        else:
            raise ValueError("Bad response: " + str(response))

    def is_all_subtasks_running(self) -> bool:
        """
        Checks if all subtasks of the current Flink job are running.

        Returns:
            True if all subtasks are running, False otherwise.
        """
        if not self._current_job_id or not self._current_flink_rest_url:
            return False

        url = f"{self._current_flink_rest_url}/jobs/{self._current_job_id}"
        try:
            vertices = self.get_vertices()
            for vertex in vertices:
                if vertex.get("status") != "RUNNING":
                    return False
            return True
        except ConnectionError as e:
            self.logger.error("Error checking subtasks status: %s", e)
            return False

    def get_running_job_plan(self) -> JobPlanDict:
        """
        Retrieves the job plan for the currently tracked running job.

        Returns:
            The dictionary representing the job plan's "plan" content.

        Raises:
            ValueError: If no job is currently tracked.
            ConnectionError: If the API request fails.
        """
        if not self._current_flink_rest_url or not self._current_job_id:
            raise ValueError("No active Flink job is being tracked by this connector.")

        url = f"{self._current_flink_rest_url}/jobs/{self._current_job_id}/plan"
        response = self._make_request(url)

        if response and "plan" in response:
            return response["plan"]
        else:
            raise ValueError(f"Invalid response structure for job plan from {url}")

    def is_backpressured(self, plan: JobPlanDict) -> bool:
        """
        Checks if the job is backpressured.

        Returns:
            True if the job is backpressured, False otherwise.
        """
        if not self._current_job_id or not self._current_flink_rest_url:
            raise ValueError("No active Flink job is being tracked by this connector.")

        # Check each vertex in the job plan
        if not plan or "nodes" not in plan:
            raise ValueError("Invalid job plan provided.")

        for node in plan["nodes"]:
            vertex_id = node["id"]

            # Check backpressure for this vertex
            url = f"{self._current_flink_rest_url}/jobs/{self._current_job_id}/vertices/{vertex_id}/subtasks/metrics?get=backPressuredTimeMsPerSecond"
            response = self._make_request(url)
            if response and isinstance(response, list):
                if len(response) == 1:
                    if (
                        isinstance(response[0], dict)
                        and response[0].get("id") == "backPressuredTimeMsPerSecond"
                    ):
                        if response[0].get("avg", 0) > 100:
                            return True
                    else:
                        raise ValueError("Invalid response: " + str(response))
                else:
                    self.logger.warning(
                        "Could not parse back pressure info for vertex %s: %s",
                        vertex_id,
                        response,
                    )
            else:
                raise ValueError("Invalid response: " + str(response))

        return False

    @check_valid_state
    def observe_task_metrics(
        self, plan: JobPlanDict, metric_names: list[str]
    ) -> dict[str, dict[str, MetricStatDict]]:
        """
        Collects specified task-scoped metrics for each vertex in the job plan.

        Args:
            plan: The job plan dictionary (output of get_job_plan).
            metric_names: A list of metric names to query (e.g., 'numRecordsInPerSecond').

        Returns:
            A dictionary mapping vertex ID to a list of metric values (one per metric name), each metric value a dict
            containing min, max, avg, sum, skew.

        Raises:
            ValueError: If no job is tracked or the plan is invalid.
            ConnectionError: If API requests fail.
        """
        if not self._current_flink_rest_url or not self._current_job_id:
            raise ValueError("No active Flink job is being tracked by this connector.")
        if not plan or "nodes" not in plan:
            raise ValueError("Invalid job plan provided.")

        all_metrics: dict[str, dict[str, MetricStatDict]] = {}
        metric_query = ",".join(metric_names)

        for node in plan["nodes"]:
            vertex_id = node.get("id")
            if not vertex_id:
                raise ValueError("Node in plan missing id")

            url = f"{self._current_flink_rest_url}/jobs/{self._current_job_id}/vertices/{vertex_id}/subtasks/metrics"
            params = {"get": metric_query}

            try:
                response = self._make_request(url, params=params)
                node_metrics: dict[str, MetricStatDict] = {}

                if isinstance(response, list):
                    subtask_values_found: dict[str, Optional[MetricStatDict]] = {
                        name: None for name in metric_names
                    }
                    for metric_info in response:
                        if not isinstance(metric_info, dict):
                            raise ValueError("metric info is not a dict")
                        metric_id = metric_info.get("id")
                        if metric_id in subtask_values_found:
                            del metric_info["id"]
                            if all(
                                [
                                    k in metric_info
                                    for k in ["min", "max", "avg", "sum", "skew"]
                                ]
                            ):
                                subtask_values_found[metric_id] = metric_info
                            else:
                                raise ValueError("metric info is missing keys")
                        else:
                            raise ValueError(
                                f"Metric info {metric_id} not in requested metrics"
                            )

                    for name in metric_names:
                        node_metrics[name] = subtask_values_found[name]

                else:
                    raise ValueError("Unexpected response format for metrics of vertex")

                all_metrics[vertex_id] = node_metrics

            except ConnectionError as e:
                # Maybe the job is not ready yet
                raise ValueError(f"Failed to fetch metrics for vertex {vertex_id}")
            except Exception as e:  # Catch other potential errors
                raise ValueError(f"Error when getting metrics for vertex {vertex_id}")
        return all_metrics

    @check_valid_state
    def get_throughput(self, plan: JobPlanDict) -> list[float]:
        """
        Calculates the overall job throughput based on source operators'
        numRecordsInPerSecond.

        Returns:
            A list containing numRecordsInPerSecond values from each subtask
            of each source operator. Returns empty list if sources cannot be
            determined or metrics fetched.

        Raises:
            ValueError: If no job is tracked.
            ConnectionError: If API requests fail.
        """
        source_node_ids = get_source_node_ids(plan)
        if not source_node_ids:
            raise ValueError("No source nodes found in the job plan.")
        try:
            source_metrics = self.observe_task_metrics(
                plan, ["numRecordsInPerSecond", "numRecordsIn"]
            )
        except (ValueError, ConnectionError) as e:
            raise ValueError("Failed to fetch metrics for source nodes.")

        throughput_values: list[float] = []
        for node_id in source_node_ids:
            if node_id in source_metrics:
                # We use the sum of all subtasks
                value = source_metrics[node_id]["numRecordsInPerSecond"]
                # Ensure the value is treated as float
                if isinstance(value["sum"], (int, float)):
                    throughput_values.append(float(value["sum"]))
                else:
                    self.logger.warning(
                        "Non-numeric sum value for node %s: %s. Defaulting to 0.",
                        node_id,
                        value["sum"],
                    )
                    throughput_values.append(0.0)  # Default to 0 if not numeric
            else:
                self.logger.warning(
                    "Metrics not found for source node %s. Skipping this source node.",
                    node_id,
                )

        return throughput_values

    def get_sink_watermark(self, plan: JobPlanDict) -> list[int]:
        """
        Retrieves the current watermark for each sink node in the job plan.
        Returns:
            A list of current watermark values for each sink node.
        Raises:
            ValueError: If no job is tracked or no sink nodes are found.
            ConnectionError: If API requests fail.
        """
        sink_node_ids = get_sink_node_ids(plan)
        if not sink_node_ids:
            raise ValueError("No sink nodes found in the job plan.")
        try:
            metrics = self.observe_task_metrics(plan, ["currentInputWatermark"])
        except (ValueError, ConnectionError) as e:
            raise ValueError("Failed to fetch metrics for sink nodes.")

        watermark_values: list[int] = []
        for node_id in sink_node_ids:
            if node_id in metrics:
                value = metrics[node_id]["currentInputWatermark"]
                # Ensure the value is treated as float
                if isinstance(value["min"], (int, float)):
                    watermark_values.append(int(value["min"]))
                else:
                    self.logger.warning(
                        "Non-numeric min value for node %s: %s. Defaulting to 0.",
                        node_id,
                        value["min"],
                    )
                    watermark_values.append(0)  # Default to 0 if not numeric
            else:
                self.logger.warning(
                    "Metrics not found for source node %s. Skipping this source node.",
                    node_id,
                )

        return watermark_values

    def get_total_processed(self, plan: JobPlanDict) -> int:
        """
        Calculates the total number of records processed by the job.

        Returns:
            The total number of records processed by the job.

        Raises:
            ValueError: If no job is tracked.
            ConnectionError: If API requests fail.
        """
        source_node_ids = get_source_node_ids(plan)
        if not source_node_ids:
            raise ValueError("No source nodes found in the job plan.")
        try:
            metrics = self.observe_task_metrics(plan, ["numRecordsIn"])
        except (ValueError, ConnectionError) as e:
            raise ValueError("Failed to fetch metrics for source nodes.")

        in_values: list[int] = []
        for node_id in source_node_ids:
            if node_id in metrics:
                value = metrics[node_id]["numRecordsIn"]
                # Ensure the value is treated as int
                if isinstance(value["sum"], (int, float)):
                    in_values.append(int(value["sum"]))
                else:
                    self.logger.warning(
                        "Non-numeric sum value for node %s: %s. Defaulting to 0.",
                        node_id,
                        value["sum"],
                    )
            else:
                self.logger.warning(
                    "Metrics not found for source node %s. Skipping this source node.",
                    node_id,
                )

        return sum(in_values)

    def _get_yarn_app_attempt_id(self) -> Optional[str]:
        """Fetches the first (usually only) application attempt ID from YARN."""
        if not self._current_yarn_app_id:
            return None

        url = f"{self.yarn_rm_http_address}/ws/v1/cluster/apps/{self._current_yarn_app_id}/appattempts"
        try:
            response = self._make_request(url)
            if (
                response
                and "appAttempts" in response
                and "appAttempt" in response["appAttempts"]
            ):
                attempts = response["appAttempts"]["appAttempt"]
                if attempts and isinstance(attempts, list) and len(attempts) > 0:
                    return attempts[-1].get("appAttemptId")
            raise ValueError(
                f"Could not find app attempts for {self._current_yarn_app_id} via YARN API."
            )
        except ConnectionError as e:
            raise ValueError(
                f"Error fetching YARN app attempts: {self._current_yarn_app_id}"
            )

    def _get_yarn_containers(self) -> Optional[list[dict[str, Any]]]:
        """Fetches container information from YARN for the current app attempt."""
        if not self._current_yarn_app_id:
            raise ValueError("No YARN application ID is being tracked.")

        app_attempt_id = self._get_yarn_app_attempt_id()
        if not app_attempt_id:
            self.logger.error(
                "Cannot get containers without a YARN application attempt ID."
            )
            return None

        url = f"{self.yarn_rm_http_address}/ws/v1/cluster/apps/{self._current_yarn_app_id}/appattempts/{app_attempt_id}/containers"
        try:
            response = self._make_request(url)
            # The actual containers might be nested under 'containers': {'container': [...]}
            if response and "container" in response:
                containers_list = response["container"]
                if isinstance(containers_list, list):
                    return containers_list
                else:
                    self.logger.error(
                        "Warning: Expected a list of containers, but got type %s.}",
                        type(containers_list),
                    )
                    return None  # Or handle single container case if API allows it
            else:
                # It's possible there are no containers *yet* or the app finished quickly
                self.logger.error(
                    "Warning: 'containers' or 'container' key not found or invalid in YARN response for %s",
                    app_attempt_id,
                )
                return None  # Return None to indicate data not available as expected
        except ConnectionError as e:
            self.logger.error("Error fetching YARN containers: %s", e)
            return None  # Indicate failure

    @check_valid_state
    def get_core_usage(self) -> int:
        """
        Gets the total number of vCores allocated to the Flink YARN application.

        Returns:
            Total allocated vCores, or 0 if unable to fetch.

        Raises:
            ValueError: If no job is tracked.
            ConnectionError: If YARN API requests fail.
        """
        containers = self._get_yarn_containers()
        if containers is None:  # Handles case where app ID is missing or API fails
            raise ValueError(
                "Could not retrieve container information to calculate core usage."
            )

        total_vcores = 0
        for container in containers:
            try:
                # Ensure the value is treated as int
                vcores = int(container.get("allocatedVCores", 0))
                total_vcores += vcores
            except (ValueError, TypeError):
                raise ValueError(
                    f"Warning: Could not parse 'allocatedVCores' from container info: {container}"
                )

        return total_vcores

    @check_valid_state
    def get_memory_usage(self) -> int:
        """
        Gets the total memory (in MB) allocated to the Flink YARN application.

        Returns:
            Total allocated memory in MB, or 0 if unable to fetch.

        Raises:
            ValueError: If no job is tracked.
            ConnectionError: If YARN API requests fail.
        """
        containers = self._get_yarn_containers()
        if containers is None:  # Handles case where app ID is missing or API fails
            raise ValueError(
                "Could not retrieve container information to calculate memory usage."
            )

        total_memory_mb = 0
        for container in containers:
            try:
                # Ensure the value is treated as int
                memory_mb = int(container.get("allocatedMB", 0))
                total_memory_mb += memory_mb
            except (ValueError, TypeError):
                raise ValueError(
                    f"Warning: Could not parse 'allocatedMB' from container info: {container}"
                )

        return total_memory_mb

    def _clear_current_job_state(self):
        self._current_job_id = None
        self._current_flink_rest_url = None
        self._current_yarn_app_id = None
        self._current_monitor_stop_event = None

    @check_valid_state
    def stop(self, with_savepoint: bool = True) -> Optional[str]:
        """
        Stops the currently running Flink job via YARN, triggering a savepoint.

        Args:
            with_savepoint: If True, triggers a savepoint before stopping the job.

        Returns:
            The path to the completed savepoint, or None if savepoint is not required.

        Raises:
            ValueError: If no job is currently tracked.
            RuntimeError: If the stop command fails.
            FileNotFoundError: If flink executable is not found.
        """
        if not self._current_job_id or not self._current_yarn_app_id:
            raise ValueError("No active Flink job/YARN application is being tracked.")

        # Interrupt monitor thread
        if self._current_monitor_stop_event:
            self.logger.info("Interrupting monitor thread.")
            self._current_monitor_stop_event.set()

        self._base_time = int(time.time() * 1000)
        savepoint_path = None
        if with_savepoint:
            command = [
                os.path.join(self.flink_home, "bin", "flink"),
                "savepoint",
                "--type",
                "native",
                self._current_job_id,
                self.savepoint_dir,
                "-yid",
                self._current_yarn_app_id,
            ]

            try:
                stdout = self._run_command(command)
                # Example output: "Savepoint completed. Path: hdfs://namenode:8020/flink-savepoints/savepoint-6eb4ec-b17e60c97d74"
                savepoint_match = re.search(
                    r"Savepoint completed\. Path: (\S+)", stdout
                )
                if savepoint_match:
                    savepoint_path = savepoint_match.group(1)
                    print(f"Savepoint successfully created at: {savepoint_path}")
                else:
                    self.logger.warning(
                        "Savepoint creation failed. None will be returned as savepoint directory. Command output:\n%s",
                        stdout,
                    )
            except Exception as e:
                self.logger.error(
                    "Error creating savepoint for job %s, None will be retuend as savepoint directory: %s",
                    self._current_job_id,
                    e,
                )
        command = [
            os.path.join(self.flink_home, "bin", "flink"),
            "cancel",
            "-yid",
            self._current_yarn_app_id,
            self._current_job_id,
        ]

        try:
            stdout = self._run_command(command)
            # No savepoint required, just reset state
            self._clear_current_job_state()
            return savepoint_path
        except RuntimeError as e:
            self.logger.error("Error canceling job %s: %s", self._current_job_id, e)
            # Do not reset state if stop failed
            raise  # Re-raise the exception
        except Exception as e:
            self.logger.error("Unexpected error during cancel: %s", e)
            raise RuntimeError(f"Unexpected error stopping job: {e}") from e

    def reset_base_time(self) -> None:
        self._base_time = None

    def remove_all_savepoints(self) -> None:
        """
        Removes all savepoints from the configured savepoint directory to save storage.
        This is a destructive operation and should be used with caution.
        """
        command = [
            "/home/User/java/hadoop-3.4.1/bin/hdfs",
            "dfs",
            "-rm",
            "-r",
            "-f",
            self.savepoint_dir,
        ]
        try:
            self._run_command(command)
            self.logger.info("All savepoints removed from %s", self.savepoint_dir)
        except subprocess.CalledProcessError as e:
            self.logger.error(
                "Error removing savepoints from %s: %s", self.savepoint_dir, e
            )

    def reset(self) -> None:
        """
        Resets the connector state, killing all running and pending jobs.
        """
        # Find any running Yarn app and kill it
        url = f"{self.yarn_rm_http_address}/ws/v1/cluster/apps"
        try:
            response = self._make_request(url)
            if response and "apps" in response and "app" in response["apps"]:
                for app in response["apps"]["app"]:
                    if app.get("state") in [
                        "RUNNING",
                        "ACCEPTED",
                        "NEW",
                        "NEW_SAVING",
                        "SUBMITTED",
                    ]:
                        app_id = app.get("id")
                        if app_id:
                            self.logger.debug(
                                "Killing %s YARN application: %s",
                                app.get("state"),
                                app_id,
                            )
                            # Kill the application
                            kill_url = f"{self.yarn_rm_http_address}/ws/v1/cluster/apps/{app_id}/state"
                            self._make_request(
                                kill_url, method="PUT", data={"state": "KILLED"}
                            )
        except ConnectionError as e:
            self.logger.error("Error fetching YARN applications: %s", e)

        # Remove Flink checkpoint
        self._hdfs_rm([FLINK_SAVEPOINT_PATH, FLINK_JAR_UPLOAD_PATH])

        # Reset internal state
        self._clear_current_job_state()
        self._in_valid_state = True

    def _hdfs_rm(self, paths: list[str]) -> None:
        """
        Removes the specified paths from HDFS.
        Args:
            paths: List of HDFS paths to remove.
        """
        command = [
            "/home/User/java/hadoop-3.4.1/bin/hdfs",
            "dfs",
            "-rm",
            "-r",
            "-f",
        ]
        command.extend(paths)
        try:
            self._run_command(command)
            self.logger.info("Removed HDFS paths: %s", paths)
        except subprocess.CalledProcessError as e:
            self.logger.error("Error removing HDFS paths %s: %s", paths, e)
