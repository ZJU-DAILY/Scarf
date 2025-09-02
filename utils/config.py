import socket
from typing import List, Optional, Union

import yaml
from humps import camelize
from pydantic import BaseModel, ConfigDict, field_validator


class BaseConfig(BaseModel):
    model_config = ConfigDict(
        alias_generator=camelize,
        validate_by_alias=True,
    )


class ModeConfig(BaseConfig):
    env: str
    """Environment name"""

    algo: str
    """Algorithm, sorl (single-objective) or morl (multi-objective)"""

    connector: str
    """Flink connector, yarn or mock"""


class DefaultKnobConfig(BaseConfig):
    name: str
    """Knob name"""

    value: str
    """Knob value"""


class FlinkConfig(BaseConfig):
    savepoint_dir: str
    yarn_rm_http_address: str
    flink_home: str
    hadoop_home: str
    default_cluster_knobs: List[DefaultKnobConfig]


class JobConfig(BaseConfig):
    jar_path: str
    main_class: str
    job_name: str
    default_job_params: str  # excluding recovery-mode, event-rates, and base-time
    base_time_param_name: str
    rate_param_name: str
    offline_source_rate: int
    online_source_rates: list[int]


class EnvConfig(BaseConfig):
    job_warmup_sec: float
    """Warmup time for the job before collecting metrics"""

    monitor_interval_sec: float
    """Interval for monitoring job metrics"""

    max_wait_attempts: int
    """Maximum attempts to wait for job to run"""

    max_monitor_attempts: int
    """Maximum attempts to monitor job metrics"""

    stable_window_size: int
    """Window size for stability check"""

    throughput_weight: float
    """Weight for throughput in the objective function"""

    resource_weight: float
    """Weight for resource utilization in the objective function"""

    max_core_usage: int
    """Maximum possible core usage for a job. Only for normalization, does not limit anything"""

    max_memory_m_bytes: int
    """Maximum possible memory usage for a job. Only for normalization, does not limit anything"""

    savepoint_size: int
    """Input size for savepoint. 0 means no savepoint."""

    savepoint_store: str = ""
    """Savepoint store. If not set, in-memory database will be used."""

    single_objective_weight: list[float]
    """For single-objective learning: weights for each objective."""


class TunerConfig(BaseConfig):
    max_steps: int
    """Total number of environment steps to train"""

    episode_length: int
    """Episode length (soft)"""

    train_batch_size: int
    """Train batch size"""

    actor_lr: float
    """Actor LR"""

    critic_lr: float
    """Critic LR"""

    alpha_lr: float
    """Alpha LR"""

    initial_alpha: float
    """Initial alpha"""

    tau: float
    """Tau (soft target update rate)"""

    gamma: float
    """Gamma (discount factor)"""

    policy_class: str
    """Policy class (MOGcnPolicy or (MOMlpPolicy)"""

    warmup_iter: int
    """Number of warmup episodes"""

    update_iter: int
    """Number of episodes between updates"""

    log_iter: int
    """Number of episodes between logs"""

    gradient_steps: int
    """Number of gradient steps per update"""

    eval_steps: int
    """Number of steps for each evaluation"""

    min_weight: float
    """Population warmup: minimum weight"""

    max_weight: float
    """Population warmup: maximum weight"""

    delta_weight: float
    """Population warmup: weight delta"""

    num_tasks: int
    """Number of tasks to be selected"""

    num_weight_candidates: int
    """Number of weight candidates"""

    num_population_buffers: int
    """Number of population buffers"""

    population_buffer_size: int
    """Population buffer size"""

    sparsity: float
    """Alpha of sparsity metric"""

    save_dir: str
    """Directory to save results"""

    load_dir: Optional[str]
    """Directory to load model from. If not set, model will not be recovered."""


class KnobConfig(BaseConfig):

    model_config = ConfigDict(coerce_numbers_to_str=True)

    name: str
    type: str
    default: str
    min: Optional[str] = None
    max: Optional[str] = None
    power_of_two: Optional[bool] = None
    values: Optional[List[str]] = None


class KnobsConfig(BaseConfig):
    operator_names: List[str]
    """Names of operator groups"""

    operator_knobs: List[KnobConfig]
    """Selected operator knobs."""

    cluster_knobs: List[KnobConfig]
    """Selected cluster knobs."""

    excluded_cluster_knob_prefixes: List[str]
    """In knob selection, exclude cluster knobs with these prefixes."""


class MetricConfig(BaseConfig):
    name: str
    scope: str
    min: int
    max: int


class MetricsConfig(BaseConfig):
    observed: List[MetricConfig]
    included_stats: List[str]


class LogLevelsConfig(BaseConfig):
    root: str = "INFO"
    """Root logger"""

    connector: str = "DEBUG"
    """Logger for the Flink connector"""

    env: str = "DEBUG"
    """Logger for the Flink RL environment"""

    algo: str = "DEBUG"
    """Logger for the MOSAC algorithm"""

    selection: str = "DEBUG"
    """Logger for knob selection"""

    trainer: str = "DEBUG"
    """Logger for the RL trainer"""

    eval: str = "DEBUG"
    """Logger for evaluation"""

    conttune: str = "DEBUG"
    """Logger for ContTune baseline"""

    ds2: str = "DEBUG"
    """Logger for DS2 baseline"""

    streamtune: str = "DEBUG"
    """Logger for StreamTune baseline"""

    bo: str = "DEBUG"
    """Logger for Bayesian Optimization baseline"""

    zerotune: str = "DEBUG"
    """Logger for ZeroTune baseline"""


class LogConfig(BaseConfig):
    levels: LogLevelsConfig


class RootConfig(BaseConfig):
    """
    General configurations for the knob selection and offline/online tuning process.
    """

    seed: int
    mode: ModeConfig
    flink: FlinkConfig
    job: JobConfig
    env: EnvConfig
    tuner: TunerConfig
    knobs: KnobsConfig
    metrics: MetricsConfig
    log: LogConfig


class FullKnobRootConfig(BaseConfig):
    """
    Additional configurations for the knob selection process.
    """

    force_check: bool
    """Check knob format rigorously"""

    num_samples: int
    """Number of samples to collect"""

    knobs: List[KnobConfig]
    """Full cluster knobs."""


# --- Loading Function ---


def load_tuning_root_config(config_path: str) -> RootConfig:
    """Loads configuration from a YAML file into a Pydantic model."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    config_data["flink"]["yarnRmHttpAddress"] = config_data["flink"][
        "yarnRmHttpAddress"
    ].replace("$HOSTNAME", socket.gethostname())
    return RootConfig.model_validate(config_data)


def save_to_yaml(config: RootConfig, output_path: str):
    """Saves the Pydantic model to a YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(config.model_dump(by_alias=True), f, default_flow_style=False)


def load_full_knob_root_config(config_path) -> FullKnobRootConfig:
    """Loads full knob configuration from a YAML file into a Pydantic model."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return FullKnobRootConfig.model_validate(config_data)
