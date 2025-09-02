import warnings
from typing import Any, Optional, Type, Union

import numpy as np
import torch as th
import torch_geometric.nn as tg_nn
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import Actor, LOG_STD_MAX, LOG_STD_MIN
from torch import nn
from typing_extensions import override

from model.buffer import batch_graph_obs
from model.features import GnnFeaturesExtractor, GraphFlattenExtractor
from model.layers import PNNLayers
from utils.types import SparseObsDict


class MOActor(BasePolicy):
    """
    Actor network (policy) for MOSAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param log_std_init: Initial value for the log standard deviation
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        n_nodes: int,
        local_action_dim: int,
        global_action_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        use_separate_heads: bool = True,
        history_modules: Optional[list["MOActor"]] = None,
    ):
        action_dim = get_action_dim(action_space)
        assert n_nodes * local_action_dim + global_action_dim == action_dim

        if use_sde:
            raise ValueError("use_sde")
        if isinstance(features_extractor, GnnFeaturesExtractor):
            net_arch = []

        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        self.n_nodes = n_nodes
        self.history_modules = history_modules if history_modules else []

        self.latent_pi = PNNLayers(
            features_dim,
            -1,
            net_arch,
            activation_fn,
            history_modules=[m.latent_pi for m in self.history_modules],
        )
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        self.use_separate_heads = (
            isinstance(features_extractor, GnnFeaturesExtractor)
            and self.n_nodes > 0
            and use_separate_heads
        )

        if self.use_separate_heads:
            self.local_mu = nn.Linear(last_layer_dim, local_action_dim)
            self.local_log_std = nn.Linear(last_layer_dim, local_action_dim)
            self.global_mu = nn.Linear(last_layer_dim, global_action_dim)
            self.global_log_std = nn.Linear(last_layer_dim, global_action_dim)
        else:
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
                normalize_images=self.normalize_images,
                history_modules=self.history_modules,
            )
        )
        return data

    def get_action_dist_params(
        self, obs: SparseObsDict
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        batch_vector = obs["batch_vector"]
        features = self.extract_features(obs, self.features_extractor)
        if self.use_separate_heads:
            local_mean_actions = self.local_mu(features)
            local_log_std = self.local_log_std(features)

            pooled_features = tg_nn.global_mean_pool(features, batch_vector)
            global_mean_actions = self.global_mu(pooled_features)
            global_log_std = self.global_log_std(pooled_features)

            # Concatenate local and global actions
            # Local actions of the same graph are first concatenated, then with the global action
            assert features.shape[0] % self.n_nodes == 0
            n_graphs = features.shape[0] // self.n_nodes
            mean_actions = th.cat(
                [local_mean_actions.reshape(n_graphs, -1), global_mean_actions],
                dim=1,
            )
            log_std = th.cat(
                [local_log_std.reshape(n_graphs, -1), global_log_std], dim=1
            )
            log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        else:
            if isinstance(self.features_extractor, GnnFeaturesExtractor):
                features = tg_nn.global_mean_pool(features, batch_vector)
            latent_pi = self.latent_pi(features)
            mean_actions = self.mu(latent_pi)
            # Unstructured exploration (Original implementation)
            log_std = self.log_std(latent_pi)  # type: ignore[operator]
            # Original Implementation to cap the standard deviation
            log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs
        )

    def action_log_prob(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        return self(observation, deterministic)

    def extract_features(
        self, obs: SparseObsDict, features_extractor: BaseFeaturesExtractor
    ) -> th.Tensor:
        return features_extractor(obs)


class MOContinuousCritic(BaseModel):
    """
    Critic network(s) for MOSAC.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a vector Q(s, a).
    Each dimension of the output tensor represents the Q-value for each objective.
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param n_objs: Number of objectives
    :param n_nodes: Number of nodes in the job plan. For SQL jobs this is zero.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        n_objs: int,
        n_nodes: int,
        local_action_dim: int,
        global_action_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        history_modules: Optional[list["MOContinuousCritic"]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if isinstance(features_extractor, GnnFeaturesExtractor):
            # Action is embedded in node features
            action_dim = 0
            net_arch = []
        else:
            action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.n_objs = n_objs
        self.n_nodes = n_nodes
        self.local_action_dim = local_action_dim
        self.global_action_dim = global_action_dim
        self.q_networks: list[nn.Module] = []
        self.history_modules = history_modules if history_modules else []

        for idx in range(n_critics):
            q_net = PNNLayers(
                features_dim + action_dim,
                n_objs,
                net_arch,
                activation_fn,
                history_modules=[m.q_networks[idx] for m in self.history_modules],  # type: ignore
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def _expand(self, obs, actions):
        assert (
            self.local_action_dim * self.n_nodes + self.global_action_dim
            == actions.shape[-1]
        )

        batch_size = obs["node_features"].shape[0]
        obs_with_actions = []
        if self.n_nodes == 0:
            for i in range(batch_size):
                batch_id = obs["batch_vector"][i]
                obs_with_actions.append(
                    th.cat([obs["node_features"][i], actions[batch_id]])
                )
        else:
            for i in range(batch_size):
                node_id = i % self.n_nodes
                batch_id = obs["batch_vector"][i]
                obs_with_actions.append(
                    th.cat(
                        [
                            obs["node_features"][i],
                            actions[
                                batch_id,
                                node_id
                                * self.local_action_dim : (node_id + 1)
                                * self.local_action_dim,
                            ],
                            actions[batch_id, self.n_nodes * self.local_action_dim :],
                        ]
                    )
                )
        return {
            "node_features": th.stack(obs_with_actions),
            "edge_indices": obs["edge_indices"],
            "batch_vector": obs["batch_vector"],
        }

    def forward(self, obs: dict, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        # Concat node features with actions

        node_feature_dim = self.features_extractor.node_feature_dim
        obs_dim = obs["node_features"].shape[1]

        if obs_dim < node_feature_dim:
            expanded_obs = self._expand(obs, actions)
        else:
            expanded_obs = obs

        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(expanded_obs, self.features_extractor)
            if isinstance(self.features_extractor, GnnFeaturesExtractor):
                features = tg_nn.global_mean_pool(
                    features, expanded_obs["batch_vector"]
                )
        if obs_dim < node_feature_dim:
            qvalue_input = features
        else:
            # qvalue_input: (batch_size, features_dim + action_dim)
            qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: dict, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """

        node_feature_dim = self.features_extractor.node_feature_dim
        obs_dim = obs["node_features"].shape[1]

        if obs_dim < node_feature_dim:
            obs = self._expand(obs, actions)

        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        if obs_dim < node_feature_dim:
            qvalue_input = features
        else:
            # qvalue_input: (batch_size, features_dim + action_dim)
            qvalue_input = th.cat([features, actions], dim=1)
        return self.q_networks[0](qvalue_input)

    def extract_features(
        self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor
    ) -> th.Tensor:
        # Skip preprocessing of the observation
        return features_extractor(obs)


class MOSACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for MOSAC.

    :param observation_space: Observation space
    :param action_space: Action space = (local_action_dim * n_nodes + global_action_dim, )
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_objs: Number of objectives
    :param n_nodes: Number of nodes in the graph. SQL jobs should set this to 0.
    :param local_action_dim: Dimension of local actions.
    :param global_action_dim: Dimension of global actions.
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
       a positive standard deviation (cf paper). It allows to keep variance
       above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
       to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
        dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
       ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
       excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
       between the actor and the critic (this saves computation time)
    """

    actor: Actor
    critic: MOContinuousCritic
    critic_target: MOContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        *,
        actor_lr_schedule: Schedule,
        critic_lr_schedule: Schedule,
        n_objs: int,
        n_nodes: int,
        local_action_dim: int,
        global_action_dim: int,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = GraphFlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        history_modules: Optional[list["MOSACPolicy"]] = None,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.history_modules = history_modules if history_modules else []

        # Update feature extractor kwargs with GCN specific params
        features_extractor_kwargs.update(
            {
                "n_nodes": n_nodes,
            }
        )

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.n_objs = n_objs
        self.n_nodes = n_nodes
        self.local_action_dim = local_action_dim
        self.global_action_dim = global_action_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "n_nodes": n_nodes,
            "local_action_dim": local_action_dim,
            "global_action_dim": global_action_dim,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_objs": n_objs,
                "local_action_dim": local_action_dim,
                "global_action_dim": global_action_dim,
                "n_critics": n_critics,
                "n_nodes": n_nodes,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(actor_lr_schedule, critic_lr_schedule)

    def _build(self, actor_lr_schedule: Schedule, critic_lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=actor_lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(
                features_extractor=self.actor.features_extractor,
                history_critics=[m.critic for m in self.history_modules],
            )
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [
                param
                for name, param in self.critic.named_parameters()
                if "features_extractor" not in name
            ]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(
                features_extractor=None,
                history_critics=[m.critic for m in self.history_modules],
            )
            critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(
            features_extractor=None,
            history_critics=[m.critic_target for m in self.history_modules],
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=critic_lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        # observation_space, action_space, normalize_images
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                actor_lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone  # TODO: check reason
                critic_lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                n_objs=self.n_objs,
                n_nodes=self.n_nodes,
                local_action_dim=self.local_action_dim,
                global_action_dim=self.global_action_dim,
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                n_critics=self.critic_kwargs["n_critics"],
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        actor_kwargs["history_modules"] = [m.actor for m in self.history_modules]
        return MOActor(**actor_kwargs).to(self.device)

    def make_critic(
        self,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        history_critics: Optional[list["MOContinuousCritic"]] = None,
    ) -> MOContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        critic_kwargs["history_modules"] = history_critics
        return MOContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def obs_to_tensor(
        self, observation: dict[str, list[np.ndarray]]
    ) -> tuple[SparseObsDict, bool]:

        node_features, edge_indices, batch_vector = batch_graph_obs(
            observation["node_features"], observation["adjacency_matrix"], self.device
        )

        return {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "batch_vector": batch_vector,
        }, True


MOMlpPolicy = MOSACPolicy


class MOGcnPolicy(MOSACPolicy):
    """
    Multi-Objective Soft Actor-Critic (SAC) policy using a GCN features extractor.

    :param observation_space: Observation space (must be Dict with 'node_features' and 'adjacency_matrix').
    :param action_space: Action space.
    :param lr_schedule: Learning rate schedule.
    :param n_objs: Number of objectives for MORL.
    :param n_nodes: Number of nodes in the graph. SQL jobs should set this to 0.
    :param local_action_dim: Dimension of local actions.
    :param global_action_dim: Dimension of global actions.
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function.
    :param features_dim: Output dimension of the GCN feature extractor.
    :param gcn_layers: Number of layers in the GCN feature extractor.
    :param gcn_activation_fn: Activation function for the GCN layers.
    :param gcn_pooling_fn: Pooling function for the GCN layers.
    :param use_sde: Whether to use State Dependent Exploration or not.
    :param log_std_init: Initial value for the log standard deviation.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE.
    :param clip_mean: Clip the mean output when using gSDE.
    :param features_extractor_class: Should be GnnFeaturesExtractor.
    :param features_extractor_kwargs: Keyword arguments for GnnFeaturesExtractor.
    :param normalize_images: Should be False for Dict space.
    :param optimizer_class: The optimizer to use.
    :param optimizer_kwargs: Additional keyword arguments for the optimizer.
    :param n_critics: Number of critic networks.
    :param share_features_extractor: Whether to share the GCN extractor between actor and critic.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        actor_lr_schedule: Schedule,
        critic_lr_schedule: Schedule,
        n_objs: int,
        n_nodes: int,
        local_action_dim: int,
        global_action_dim: int,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        # GCN specific parameters
        features_dim: int = 256,
        gcn_layers: int = 2,
        gcn_activation_fn: Type[nn.Module] = nn.ReLU,
        gcn_pooling_fn=None,
        # SAC specific parameters
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = GnnFeaturesExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = False,  # dict space is not image
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,  # Can be shared
        history_modules: Optional[list["MOGcnPolicy"]] = None,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        # Update feature extractor kwargs with GCN specific params
        features_extractor_kwargs.update(
            {
                "n_nodes": n_nodes,
                "features_dim": features_dim,
                "gcn_layers": gcn_layers,
                "activation_fn": gcn_activation_fn,
                "pooling_fn": gcn_pooling_fn,
            }
        )

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            actor_lr_schedule=actor_lr_schedule,
            critic_lr_schedule=critic_lr_schedule,
            n_objs=n_objs,
            n_nodes=n_nodes,
            local_action_dim=local_action_dim,
            global_action_dim=global_action_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            history_modules=history_modules,
        )
        self.local_action_dim = local_action_dim
        self.global_action_dim = global_action_dim

    @override
    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> MOActor:
        if self.features_extractor_class != GnnFeaturesExtractor:
            raise ValueError("MOSAC requires GnnFeaturesExtractor")
        net_kwargs = self.actor_kwargs.copy()
        features_extractor = GnnFeaturesExtractor(
            self.observation_space,
            local_action_dim=0,
            global_action_dim=0,
            history_modules=[m.actor.features_extractor for m in self.history_modules],  # type: ignore
            **self.features_extractor_kwargs,
        )
        net_kwargs.update(
            dict(
                features_extractor=features_extractor,
                features_dim=features_extractor.features_dim,
                history_modules=[m.actor for m in self.history_modules],  # type: ignore
            )
        )
        return MOActor(**net_kwargs).to(self.device)

    @override
    def make_critic(
        self,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        history_critics: Optional[list["MOContinuousCritic"]] = None,
    ) -> MOContinuousCritic:
        if self.features_extractor_class != GnnFeaturesExtractor:
            raise ValueError("MOSAC requires GnnFeaturesExtractor")
        net_kwargs = self.critic_kwargs.copy()
        features_extractor = GnnFeaturesExtractor(
            self.observation_space,
            local_action_dim=self.local_action_dim,
            global_action_dim=self.global_action_dim,
            history_modules=[c.features_extractor for c in history_critics],  # type: ignore
            **self.features_extractor_kwargs,
        )
        net_kwargs.update(
            dict(
                features_extractor=features_extractor,
                features_dim=features_extractor.features_dim,
                history_modules=history_critics,
            )
        )
        return MOContinuousCritic(**net_kwargs).to(self.device)

    @override
    def _update_features_extractor(
        self,
        net_kwargs: dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "MOGcnPolicy handles features extractors for actor and critic differently. Use `make_actor` and "
            "`make_critic` methods instead."
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                features_dim=self.features_extractor_kwargs["features_dim"],
                gcn_layers=self.features_extractor_kwargs["gcn_layers"],
                gcn_activation_fn=self.features_extractor_kwargs["activation_fn"],
                gcn_pooling_fn=self.features_extractor_kwargs["pooling_fn"],
            )
        )
        return data

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            if features_extractor is None:
                features_extractor = self.features_extractor
            return features_extractor(obs)  # Skip preprocessing
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = self.pi_features_extractor(obs)
            vf_features = self.vf_features_extractor(obs)
            return pi_features, vf_features
