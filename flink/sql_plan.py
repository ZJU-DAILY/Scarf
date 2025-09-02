import json
import hashlib
import logging
from pprint import pprint
from typing import Any, Optional

import sqlite3

from utils.types import JobPlanNodeDict

logger = logging.getLogger("env")


def _compare_nodes(node1, node2):
    if not isinstance(node1, dict) or not isinstance(node2, dict):
        return node1 == node2

    for key in node1:
        if key == "parallelism":
            continue
        if key not in node2:
            return False
        if not _compare_nodes(node1[key], node2[key]):
            return False
    for key in node2:
        if key == "parallelism":
            continue
        if key not in node1:
            return False
    return True


def compare_plans(plan1: dict[str, Any], plan2: dict[str, Any]):
    if not isinstance(plan1, dict) or not isinstance(plan2, dict):
        return plan1 == plan2

    nodes1 = plan1.get("nodes", [])
    nodes2 = plan2.get("nodes", [])

    if len(nodes1) != len(nodes2):
        return False

    for node1, node2 in zip(nodes1, nodes2):
        if not _compare_nodes(node1, node2):
            return False

    return True


def _canonicalize_node(node, nodes_map, visited, id_map):
    out = {
        "type": node["type"],
        "pact": node["pact"],
        "contents": node["contents"],
    }
    if "predecessors" in node:
        preds = []
        for pred in node["predecessors"]:
            pred_id = pred["id"]
            if pred_id not in id_map:
                id_map[pred_id] = len(id_map)
            preds.append(
                _canonicalize_node(nodes_map[pred_id], nodes_map, visited, id_map)
            )
        out["predecessors"] = preds
    return out


def generate_plan_hash(original_plan: dict[str, Any]) -> str:
    nodes = original_plan["nodes"]
    nodes_map = {node["id"]: node for node in nodes}
    all_ids = set(node["id"] for node in nodes)
    referenced_ids = set()
    for node in nodes:
        if "predecessors" in node:
            referenced_ids.update(pred["id"] for pred in node["predecessors"])
    root_ids = all_ids - referenced_ids
    roots = [nodes_map[rid] for rid in sorted(root_ids)]
    visited = set()
    id_map = {}
    canonicalized = [
        _canonicalize_node(root, nodes_map, visited, id_map) for root in roots
    ]
    plan_str = json.dumps(canonicalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(plan_str.encode("utf-8")).hexdigest()


import hashlib
from typing import Dict, Set


def get_job_plan_hash(nodes: list[JobPlanNodeDict]) -> str:
    """
    Get a hash of the job plan topology that is independent of parallelism values.

    Args:
        nodes: List of job plan nodes

    Returns:
        A hash string representing the topology structure
    """
    # Create a mapping from node id to a stable identifier based on operator type
    # This ensures the hash is consistent regardless of actual node IDs
    node_id_to_operator = {node["id"]: node["operator"] for node in nodes}

    # Sort nodes by ID to ensure consistent ordering
    sorted_nodes = sorted(nodes, key=lambda x: x["id"])

    # Build topology representation as a list of edges and node operators
    topology_data = []

    # Add node information (operator type only, not parallelism or ID)
    node_operators = [node["operator"] for node in sorted_nodes]
    topology_data.append(("nodes", tuple(sorted(node_operators))))

    # Add edge information based on inputs
    edges = []
    for node in sorted_nodes:
        if "inputs" in node and node["inputs"]:
            for input_info in node["inputs"]:
                source_operator = node_id_to_operator[input_info["id"]]
                target_operator = node["operator"]
                # Include exchange and ship_strategy as they affect topology
                edge_info = (
                    source_operator,
                    target_operator,
                    input_info["exchange"],
                    input_info["ship_strategy"],
                )
                edges.append(edge_info)

    # Sort edges to ensure consistent ordering
    edges.sort()
    topology_data.append(("edges", tuple(edges)))

    # Create hash from the topology data
    topology_str = str(topology_data)
    return hashlib.sha256(topology_str.encode("utf-8")).hexdigest()


class SavepointStore:
    """
    A class to manage savepoints for Flink jobs.
    """

    def __init__(self, db_path: Optional[str]):
        if db_path is None:
            # Use in-memory store
            db_path = ":memory:"
        db = sqlite3.connect(db_path)
        self._cursor = db.cursor()
        self._db = db
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS savepoints (
                plan_hash TEXT PRIMARY KEY,
                savepoint_dir TEXT
            )
            """
        )

    def save(self, plan_hash: str, savepoint_dir: str):
        """
        Save a savepoint directory for a given plan hash.
        """
        self._cursor.execute(
            "INSERT INTO savepoints (plan_hash, savepoint_dir) VALUES (?, ?)",
            (plan_hash, savepoint_dir),
        )
        self._db.commit()

    def get(self, plan_hash: str) -> dir:
        """
        Retrieve the savepoint directory for a given plan hash.
        """
        self._cursor.execute(
            "SELECT savepoint_dir FROM savepoints WHERE plan_hash = ?",
            (plan_hash,),
        )
        result = self._cursor.fetchone()
        if result:
            return result[0]
        return None
