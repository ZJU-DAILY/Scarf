from typing import Any

from utils.types import JobPlanDict, JobPlanNodeDict, ParallelismKnobSet, VertexDict


def is_throughput_stable(
    throughputs: list[list[float]], stable_window_size: int
) -> bool:
    if len(throughputs) < stable_window_size:
        return False

    return True


def get_source_node_ids(plan: JobPlanDict) -> list[str]:
    source_node_ids: list[str] = []
    if "nodes" in plan:
        for node in plan["nodes"]:
            # Source nodes have no inputs
            if "inputs" not in node and "id" in node:
                source_node_ids.append(node["id"])
    return source_node_ids


def get_sink_node_ids(plan: JobPlanDict) -> list[str]:
    """
    Returns a list of node IDs that are sink nodes in the job plan.
    Sink nodes are identified by having no outputs.
    """
    all_node_ids: set[str] = set()
    input_node_ids: set[str] = set()
    if "nodes" in plan:
        for node in plan["nodes"]:
            all_node_ids.add(node["id"])
            # Sink nodes have no outputs
            if "inputs" in node:
                for input in node["inputs"]:
                    input_node_ids.add(input["id"])
    return list(all_node_ids - input_node_ids)


def fuzzy_get_id(vertices: list[VertexDict], node_name: str) -> str:
    """Fuzzy match the node name to get the node ID.
    :param vertices: List of vertices in the job.
    :param node_name: Name of the node to match.
    """
    for v in vertices:
        if node_name in v["name"]:
            return v["id"]
    raise ValueError(f"Node {node_name} not found in job plan")


def toposort(nodes: list[JobPlanNodeDict]) -> list[JobPlanNodeDict]:
    in_degrees: dict[str, int] = {}
    outputs: dict[str, list[str]] = {}

    for node in nodes:
        if "inputs" in node:
            in_degrees[node["id"]] = len(node["inputs"])
            for input_node in node["inputs"]:
                if input_node["id"] not in outputs:
                    outputs[input_node["id"]] = []
                outputs[input_node["id"]].append(node["id"])
        else:
            in_degrees[node["id"]] = 0

    queue: list[JobPlanNodeDict] = [
        node for node in nodes if in_degrees[node["id"]] == 0
    ]
    sorted_nodes: list[JobPlanNodeDict] = []
    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        if node["id"] in outputs:
            for output_id in outputs[node["id"]]:
                in_degrees[output_id] -= 1
                if in_degrees[output_id] == 0:
                    queue.append(next(n for n in nodes if n["id"] == output_id))

    return sorted_nodes


def parallelisms_in_array_form(
    optimal_policy: dict[str, int],
    knob_set: ParallelismKnobSet,
    vertices: list[VertexDict],
) -> list[int]:
    """
    Convert the optimal policy to an array form based on the knob set.
    :param optimal_policy: A dictionary mapping operator IDs to suggested parallelism.
    :param knob_set: The set of knobs for the job.
    :param vertices: List of vertices in the job.
    :return: List of parallelisms in the order of the knobs.
    """
    if knob_set.is_sql_job():
        return [max(optimal_policy.values())]

    p_next = []
    for knob in knob_set:
        node_id = fuzzy_get_id(vertices, knob.operator)
        if node_id not in optimal_policy:
            raise ValueError(f"Node {node_id} not found in optimal policy")
        p_next.append(optimal_policy[node_id])
    return p_next
