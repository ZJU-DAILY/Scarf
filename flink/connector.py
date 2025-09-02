from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Optional, Union

from utils.config import RootConfig
from utils.types import JobPlanDict, MetricStatDict, VertexDict


@dataclass
class FlinkJob:
    """Represents a Flink job application."""

    jar_path: str
    main_class: Optional[str]
    name: str
    default_job_params: Optional[str]
    base_time_param_name: str
    rate_param_name: str


class FlinkConnector(ABC):

    logger: Logger

    @abstractmethod
    def get_execution_plan_id(
        self, job: FlinkJob, params: str, knob_values: dict[str, str]
    ) -> str:
        """
        Retrieves the job plan for a Flink job.

        Args:
            job: The FlinkJob instance.
            params: Command-line parameters for the Flink job.
            knob_values: Flink configuration (-D key=value).

        Returns:
            The job plan dictionary.
        """
        pass

    @abstractmethod
    def submit(
        self,
        job: FlinkJob,
        params: Optional[str],
        knob_values: dict[str, str],
        source_rate: int,
        savepoint_dir: Optional[str] = None,
    ) -> JobPlanDict:
        """
        Submits the FlinkJob to YARN in yarn-application mode.

        Args:
            job: The FlinkJob instance.
            params: Command-line parameters for the Flink job.
            knob_values: Flink configuration (-D key=value).
            savepoint_dir: Optional path to restore from a savepoint.

        Returns:
            The job plan dictionary obtained from get_job_plan().
        """

    @abstractmethod
    def submit_and_wait_until_stable(
        self,
        job: FlinkJob,
        params: str,
        knob_values: dict[str, str],
        source_rate: int | str,
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

    @abstractmethod
    def is_running(self) -> bool:
        """
        Checks if the current Flink job is running.
        :return: True if the job is running, False otherwise.
        """

    @abstractmethod
    def is_ended(self) -> bool:
        """
        Checks if the current Flink job has failed.
        :return: True if the job has failed, False otherwise.
        """

    @abstractmethod
    def get_vertices(self) -> list[VertexDict]:
        """
        Retrieves the vertices of the currently tracked running job.

        Returns:
            A list of dictionaries representing the vertices.
        """
        pass

    @abstractmethod
    def is_all_subtasks_running(self) -> bool:
        """
        Checks if all subtasks of the current Flink job are running.
        :return: True if all subtasks are running, False otherwise.
        """

    @abstractmethod
    def get_running_job_plan(self) -> JobPlanDict:
        """
        Retrieves the job plan for the currently tracked running job.

        Returns:
            The dictionary representing the job plan's "plan" content.
        """

    @abstractmethod
    def is_backpressured(self, plan: JobPlanDict) -> bool:
        """
        Checks if the job is backpressured.

        Returns:
            True if the job is backpressured, False otherwise.
        """
        pass

    @abstractmethod
    def observe_task_metrics(
        self, plan: JobPlanDict, metric_names: list[str]
    ) -> dict[str, dict[str, MetricStatDict]]:
        """
        Collects specified task-scoped metrics for each vertex in the job plan.

        Args:
            plan: The job plan dictionary (output of get_job_plan).
            metric_names: A list of metric names to query (e.g., 'numRecordsInPerSecond').

        Returns:
            A dictionary mapping vertex IDs to a list of metric values (one per subtask), each metric value a dict
            containing min, max, avg, sum, skew.
        """

    @abstractmethod
    def get_throughput(self, plan: JobPlanDict) -> list[float]:
        """
        Calculates the overall job throughput based on source operators'
        numRecordsInPerSecond.

        Returns:
            A list containing numRecordsInPerSecond values from each subtask
            of each source operator.
        """

    @abstractmethod
    def get_total_processed(self, plan: JobPlanDict) -> int:
        """
        Calculates the total number of records processed by the job.
        Returns:
            The total number of records processed by the job.
        """

    @abstractmethod
    def get_core_usage(self) -> int:
        """
        Gets the total number of vCores allocated to the Flink YARN application.

        Returns:
            Total allocated vCores.
        """

    @abstractmethod
    def get_memory_usage(self) -> int:
        """
        Gets the total memory (in MB) allocated to the Flink YARN application.

        Returns:
            Total allocated memory in MB.
        """

    @abstractmethod
    def stop(self, with_savepoint: bool) -> Optional[str]:
        """
        Stops the currently running Flink job via YARN, triggering a savepoint.

        Args:
            with_savepoint: Whether to trigger a savepoint.

        Returns:
            The path to the completed savepoint, or None if failed.
        """

    @abstractmethod
    def reset_base_time(self) -> None:
        """
        Resets the base time for the connector, used for calculating the source rate.
        This is useful when initiating a new reconfiguration test.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the connector state, killing all running and pending jobs.
        """
        pass

    def remove_all_savepoints(self):
        """
        Remove all savepoints to save space.
        """
        pass
