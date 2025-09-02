from utils.config import RootConfig, load_tuning_root_config


def read_command_and_metrics(log_path: str) -> list[tuple[str, float, float, float]]:
    """
    Parse the given log file and return a list of tuples:
    (command, throughput, core_usage, memory_usage)

    - Command is taken from lines containing "Submit command:" (everything after it).
    - Metrics are taken from the next line (before another command) matching
      "Throughput: <num>, Core usage: <num>, Memory usage: <num> MB".
    - Memory is returned as the numeric value only (unit stripped).
    - Pairs each command with the first subsequent metrics line before the next command.
    """
    import re

    results: list[tuple[str, float, float, float]] = []

    # Regex to capture everything after 'Submit command:'
    submit_re = re.compile(r"Submit command:\s*(.+)")

    # Flexible float matcher (supports integers, decimals, and scientific notation)
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    # Regex to capture the three metrics; ignore case for 'MB'
    metrics_re = re.compile(
        rf"Throughput:\s*({num})\s*,\s*Core\s+usage:\s*({num})\s*,\s*Memory\s+usage:\s*({num})\s*(?:MB)\b",
        re.IGNORECASE,
    )

    current_command = None

    # Read file line by line for scalability; ignore undecodable bytes
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Check for a new command submission line
            m_cmd = submit_re.search(line)
            if m_cmd:
                current_command = m_cmd.group(1).strip()
                continue

            # If we are waiting for metrics for the current command, try to match them
            if current_command is not None:
                m_met = metrics_re.search(line)
                if m_met:
                    throughput = float(m_met.group(1))
                    core_usage = float(m_met.group(2))
                    memory_usage = float(m_met.group(3))
                    results.append(
                        (current_command, throughput, core_usage, memory_usage)
                    )
                    current_command = None  # Pair consumed

    return results


def get_best_command(
    commands_and_metrics: list[tuple[str, float, float, float]],
    config: RootConfig,
    throughput: float,
) -> str:
    """
    Find the command with throughput greater than the given parameter,
    and minimal memory/core usage based on the formula:
    memory / config.env.maxMemoryMBytes + cores / config.env.maxCoreUsage.

    Args:
        commands_and_metrics: List of tuples (command, throughput, core_usage, memory_usage).
        config: Configuration object containing maxMemoryMBytes and maxCoreUsage.
        throughput: Minimum throughput threshold.

    Returns:
        The command string that meets the criteria, or an empty string if none found.
    """
    best_command = ""
    best_score = float("inf")

    for command, cmd_throughput, core_usage, memory_usage in commands_and_metrics:
        if cmd_throughput > throughput:
            # Calculate the score based on memory and core usage
            score = (memory_usage / config.env.max_memory_m_bytes) + (
                core_usage / config.env.max_core_usage
            )
            if score < best_score:
                best_score = score
                best_command = command

    return best_command
