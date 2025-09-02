from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
import torch_geometric.utils as tg_utils
import torch_geometric.data as tg_data
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from utils.types import GraphDictReplayBufferSamples


def batch_graph_obs(
    features: list[np.ndarray], adj_matrices: list[np.ndarray], device: th.device
):
    """
    Batch graph observations into a dictionary with node features and adjacency matrices.

    :param features: List of node feature arrays for each graph.
    :param adj_matrices: List of adjacency matrices for each graph.
    :param device: PyTorch device to use.
    :return: Dictionary with batched node features and edge indices.
    """
    data_list = []
    for x, adj in zip(features, adj_matrices):
        # Convert numpy arrays to PyTorch tensors
        edge_index = tg_utils.dense_to_sparse(th.as_tensor(adj, device=device))[0]
        data_list.append(
            tg_data.Data(x=th.as_tensor(x, device=device), edge_index=edge_index)
        )

    batch = tg_data.Batch.from_data_list(data_list)
    return batch.x, batch.edge_index, batch.batch


class MOGraphReplayBuffer(DictReplayBuffer):
    """
    Replay buffer for variable size graph observations.

    - The observation space must be a Dict with "node_features" and "adjacency_matrix".
      Their shape is (0, feature_dim) and (0, 0) respectively in the obs space,
      where 0 indicates the num_nodes varies per-sample.
    - All other buffer fields are stored and sampled as usual.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        n_objs: int,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        """
        Replay buffer used in multi-objective off-policy algorithms like MOSAC.

        :param buffer_size: Max number of element in the buffer
        :param observation_space: Observation space
        :param action_space: Action space
        :param n_objs: Number of objectives
        :param device: PyTorch device
        :param n_envs: Number of parallel environments
        :param optimize_memory_usage: Enable a memory efficient variant
            of the replay buffer which reduces by almost a factor two the memory used,
            at a cost of more complexity.
            See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
            and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
            Cannot be used in combination with handle_timeout_termination.
        :param handle_timeout_termination: Handle timeout termination (due to timelimit)
            separately and treat the task as infinite horizon task.
            https://github.com/DLR-RM/stable-baselines3/issues/284
        """
        # For node_features/adjacency_matrix, can't preallocate [buffer_size, n_envs, 0, ...], use lists instead.
        # For other keys, use default implementation.
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.n_objs = n_objs
        self.rewards = np.zeros(
            (self.buffer_size, self.n_envs, n_objs), dtype=np.float64
        )

        # Replace storage of 'node_features' and 'adjacency_matrix' as lists
        for key in ["node_features", "adjacency_matrix"]:
            # List dimension (buffer_size, n_envs), element type ndarray
            self.observations[key] = [
                [None for _ in range(n_envs)] for _ in range(self.buffer_size)
            ]
            # List dimension (buffer_size, n_envs), element type ndarray
            self.next_observations[key] = [
                [None for _ in range(n_envs)] for _ in range(self.buffer_size)
            ]

        # Feature dim (must be fixed, determined by obs_space)
        self._graph_feature_dim = self.observation_space["node_features"].shape[1]

        # Remove storage for these keys from default buffer (avoid nbytes checks)
        # (Do not change .obs_shape[...], only the data allocations.)

    def add(
        self,
        obs: Dict[str, list[np.ndarray]],
        next_obs: Dict[str, list[np.ndarray]],
        action: np.ndarray,
        reward: np.ndarray,  # shape: (n_envs, n_objs)
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Expects obs and next_obs as dict containing 'node_features' and 'adjacency_matrix' with variable num_nodes.
        All other data handled as normal.
        """
        # Save the graph fields (as object arrays of numpy)
        for key in ["node_features", "adjacency_matrix"]:
            # obs[key] shape: (n_envs, num_nodes, ...) for node_features, (n_envs, num_nodes, num_nodes) for adjacency_matrix
            for env_idx in range(self.n_envs):
                self.observations[key][self.pos][env_idx] = np.array(obs[key][env_idx])
                self.next_observations[key][self.pos][env_idx] = np.array(
                    next_obs[key][env_idx]
                )

        # Actions, rewards, done, timeouts
        action = action.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> GraphDictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        n_samples = len(batch_inds)

        # Collect lists for graph fields
        node_features_list, next_node_features_list = [], []
        adj_list, next_adj_list = [], []

        for buf_idx, env_idx in zip(batch_inds, env_indices):
            nf = self.observations["node_features"][buf_idx][env_idx]
            adj = self.observations["adjacency_matrix"][buf_idx][env_idx]
            next_nf = self.next_observations["node_features"][buf_idx][env_idx]
            next_adj = self.next_observations["adjacency_matrix"][buf_idx][env_idx]
            assert (
                nf.shape[1] == self._graph_feature_dim
            ), f"node_features dim {nf.shape} mismatch graph feature dim {self._graph_feature_dim}"
            assert adj.shape[0] == adj.shape[1], "adjacency_matrix is square"
            assert adj.shape[0] == nf.shape[0], "adjacency_matrix matches node_features"
            node_features_list.append(nf)
            adj_list.append(adj)
            next_node_features_list.append(next_nf)
            next_adj_list.append(next_adj)

        batched_node_features, edge_indices, batch_vector = batch_graph_obs(
            node_features_list, adj_list, self.device
        )
        batched_next_node_features, next_edge_indices, next_batch_vector = (
            batch_graph_obs(next_node_features_list, next_adj_list, self.device)
        )

        observations = {
            "node_features": batched_node_features,
            "edge_indices": edge_indices,
            "batch_vector": batch_vector,
        }
        next_observations = {
            "node_features": batched_next_node_features,
            "edge_indices": next_edge_indices,
            "batch_vector": next_batch_vector,
        }

        # Skip normalization
        # observations = self._normalize_obs(obs_dict, env)
        # next_observations = self._normalize_obs(next_obs_dict, env)

        # Actions, rewards, dones as in DictReplayBuffer
        actions = self.to_torch(self.actions[batch_inds, env_indices])
        dones = self.to_torch(
            self.dones[batch_inds, env_indices]
            * (1 - self.timeouts[batch_inds, env_indices])
        ).reshape(-1, 1)
        rewards = self.to_torch(
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, self.n_objs), env
            )
        )

        return GraphDictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )
