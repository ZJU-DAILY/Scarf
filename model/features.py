from typing import Callable, Optional, Self, Type, Union

import torch as th
import torch_geometric.nn as tg_nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from utils.types import SparseObsDict


class GraphFlattenExtractor(BaseFeaturesExtractor):
    """Flatten features extractor for comparison."""

    def __init__(
        self,
        observation_space: spaces.Space,
        n_nodes: int,
        history_modules: Optional[list["GraphFlattenExtractor"]] = None,
    ) -> None:
        assert isinstance(
            observation_space, spaces.Dict
        ), "GcnFeatureExtractor must be used with a gymnasium.spaces.Dict observation space."
        must_contain = ["node_features", "edge_indices", "batch_vector"]
        for field in must_contain:
            assert (
                field in observation_space.spaces
            ), f"Observation space Dict must contain '{field}'."

        # --- Determine node_feature_dim from observation_space ---
        # We still need the feature dimension per node, which should be fixed.
        node_feature_space = observation_space["node_features"]
        assert (
            isinstance(node_feature_space, spaces.Box)
            and len(node_feature_space.shape) == 2
        ), (
            "'node_features' space should be a Box with shape (num_nodes, node_feature_dim). "
            "num_nodes can be symbolic (e.g., -1) or a max size if using padding."
        )

        self.node_feature_dim = node_feature_space.shape[1]
        input_dim = self.node_feature_dim * n_nodes
        super().__init__(observation_space, input_dim)
        self.n_nodes = n_nodes

    def forward(self, observations: SparseObsDict) -> th.Tensor:
        # Shape: (num_nodes, node_feature_dim)
        node_features = observations["node_features"]
        assert node_features.shape[0] % self.n_nodes == 0
        reshaped = node_features.reshape(node_features.shape[0] // self.n_nodes, -1)
        return reshaped


class GnnFeaturesExtractor(BaseFeaturesExtractor):
    """
    A feature extractor for graph-structured observations using Graph Convolutional Networks (GCN).
    Expects observations to be a dictionary containing:
    - 'node_features': Tensor of shape (batch_size, num_nodes, node_feature_dim)
    - 'adjacency_matrix': Tensor of shape (batch_size, num_nodes, num_nodes)
    The number of nodes (`num_nodes`) is inferred dynamically from the input tensor dimensions.

    Uses torch_geometric for GCN layers.

    :param observation_space: The observation space (must be a Dict space).
    :param n_nodes: Number of nodes in the graph. SQL jobs should set this to 0.
    :param local_action_dim: Dimension of local actions to be concatenated with node features. Actor's features extractor should set this to 0.
    :param global_action_dim: Dimension of global actions to be concatenated with node features. Actor's features extractor should set this to 0.
    :param features_dim: Number of features extracted. Output dimension of the extractor.
    :param gcn_layers: Number of GCN layers.
    :param activation_fn: Activation function to use between GCN layers.
    :param pooling_fn: Pooling function to aggregate node features (e.g., global_mean_pool).
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        n_nodes: int,
        local_action_dim: int,
        global_action_dim: int,
        features_dim: int = 256,
        gcn_layers: int = 2,
        activation_fn: Type[nn.Module] = nn.ReLU,
        pooling_fn: Union[None, Callable] = None,
        graph_conv_class=tg_nn.SAGEConv,
        history_modules: Optional[list["GnnFeaturesExtractor"]] = None,
    ) -> None:
        super().__init__(observation_space, features_dim)

        assert isinstance(
            observation_space, spaces.Dict
        ), "GcnFeatureExtractor must be used with a gymnasium.spaces.Dict observation space."
        must_contain = ["node_features", "edge_indices", "batch_vector"]
        for field in must_contain:
            assert (
                field in observation_space.spaces
            ), f"Observation space Dict must contain '{field}'."

        # --- Determine node_feature_dim from observation_space ---
        # We still need the feature dimension per node, which should be fixed.
        node_feature_space = observation_space["node_features"]
        assert (
            isinstance(node_feature_space, spaces.Box)
            and len(node_feature_space.shape) == 2
        ), (
            "'node_features' space should be a Box with shape (num_nodes, node_feature_dim). "
            "num_nodes can be symbolic (e.g., -1) or a max size if using padding."
        )

        # For SQL jobs
        local_action_dim = 0 if n_nodes == 0 else local_action_dim

        self.node_feature_dim = (
            node_feature_space.shape[1] + local_action_dim + global_action_dim
        )

        self.activation_fn = activation_fn()
        self.pooling = pooling_fn

        # Define GCN layers
        gcn_modules = []
        lateral_conns = []
        lateral_size = len(history_modules) if history_modules else 0
        in_channels = self.node_feature_dim
        for i in range(gcn_layers):
            out_channels = features_dim
            gcn_modules.append(
                graph_conv_class(in_channels=in_channels, out_channels=out_channels)
            )
            lateral_conns.append(
                nn.ModuleList(
                    [
                        graph_conv_class(
                            in_channels=in_channels, out_channels=out_channels
                        )
                        for _ in range(lateral_size)
                    ]
                )
            )
            in_channels = out_channels

        self.gcn_layers = nn.ModuleList(gcn_modules)
        self.history_modules = nn.ModuleList(history_modules)
        self.lateral_conns = nn.ModuleList(lateral_conns)
        # self.linear = nn.Linear(in_channels, features_dim)

    def forward(self, observations: SparseObsDict) -> th.Tensor:
        # Shape: (num_nodes, node_feature_dim)
        node_features = observations["node_features"]
        edge_index_batch = observations["edge_indices"]  # Shape: (2, num_edges)
        batch_vector = observations["batch_vector"]  # Shape: (num_nodes,)

        # --- Infer batch_size and num_nodes dynamically ---
        if node_features.dim() != 2:
            raise ValueError(
                f"Expected node_features to have 3 dimensions (batch, nodes, features), got {node_features.dim()}"
            )
        if edge_index_batch.dim() != 2:
            raise ValueError(
                f"Expected adjacency_matrix to have 3 dimensions (batch, nodes, nodes), got {edge_index_batch.dim()}"
            )
        if batch_vector.dim() != 1:
            raise ValueError(
                f"Expected batch_vector to have 1 dimension, got {batch_vector.dim()}"
            )

        num_nodes = node_features.shape[0]  # Infer num_nodes here
        node_feature_dim = node_features.shape[1]
        device = node_features.device
        if self.node_feature_dim != node_feature_dim:
            raise ValueError(
                f"Inconsistent node_feature_dim: {self.node_feature_dim} != {node_feature_dim}"
            )

        # --- Validate dimensions ---
        if batch_vector.shape[0] != num_nodes:
            raise ValueError(
                f"Inconsistent shapes: batch_vector shape {batch_vector.shape} "
                f"and node_features shape {node_features.shape} do not match on num_nodes."
            )

        x = node_features
        lateral_size = len(self.history_modules) if self.history_modules else 0
        lateral_outputs = [x for _ in range(lateral_size)]

        # --- Apply GCN Layers ---
        for i in range(len(self.gcn_layers)):
            layer = self.gcn_layers[i]
            x = layer(x, edge_index_batch)

            for j in range(lateral_size):
                x += self.lateral_conns[i][j](lateral_outputs[j], edge_index_batch)

                with th.no_grad():
                    history_module: "GnnFeaturesExtractor" = self.history_modules[j]  # type: ignore
                    history_layer = history_module.gcn_layers[i]
                    lateral_outputs[j] = history_layer(
                        lateral_outputs[j], edge_index_batch
                    )

            # Apply activation between layers, but maybe not after the very last GCN layer if pooling follows directly
            if i < len(self.gcn_layers) - 1:
                x = self.activation_fn(x)
                for j in range(lateral_size):
                    lateral_outputs[j] = self.activation_fn(lateral_outputs[j])

        if self.pooling is not None:
            # --- Apply Pooling ---
            # Shape: (batch_size, gcn_output_dim)
            pooled_features = self.pooling(x, batch_vector)  # type: ignore

            # # --- Apply Final Linear Layer ---
            # # Shape: (batch_size, features_dim)
            # final_features = self.linear(pooled_features)
            # final_features = self.activation_fn(
            #     final_features
            # )  # Apply activation after final linear layer
            return pooled_features
        else:
            return x
