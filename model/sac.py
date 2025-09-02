import io
import logging
import pathlib
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.base_class import SelfBaseAlgorithm, maybe_make_env
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import (
    OffPolicyAlgorithm,
    SelfOffPolicyAlgorithm,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    TrainFreq,
)
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_parameters_by_name,
    get_schedule_fn,
    get_system_info,
    polyak_update,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.sac.policies import Actor
from torch.nn import functional as F

from environment.graph_vec_env import GraphVecEnv
from offline_learning.scalarization_methods import ScalarizationFunction
from model.buffer import MOGraphReplayBuffer
from model.policies import MOContinuousCritic, MOGcnPolicy, MOMlpPolicy, MOSACPolicy

SelfMOSAC = TypeVar("SelfMOSAC", bound="MOSAC")


# SAC contains lock objects which will cause deepcopy to fail
# We patch the pickle module to ignore the lock objects
def _pickle_ignore_lock():
    import _thread
    import copyreg
    import threading

    def _pickle_lock(lock):
        # Just return necessary args to recreate a new lock
        return threading.Lock, ()

    copyreg.pickle(threading.Lock, _pickle_lock)


_pickle_ignore_lock()


class MOAlgorithm(ABC):
    @abstractmethod
    def learn_mo(
        self,
        total_timesteps: int,
        scalarization: ScalarizationFunction,
        callback: MaybeCallback = None,
        log_interval: int = 0,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        pass


class MOSAC(OffPolicyAlgorithm, MOAlgorithm):
    """
    Multi-Objective Soft Actor-Critic (MOSAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param n_objs: Number of objectives
    :param n_nodes: Number of nodes in the graph. SQL jobs should set this to 0.
    :param local_action_dim: Dimension of local actions.
    :param global_action_dim: Dimension of global actions.
    :param actor_lr: learning rate for actor, it can be a function of the current progress remaining (from 1 to 0)
    :param critic_lr: learning rate for critic, it can be a function of the current progress remaining (from 1 to 0)
    :param alpha_lr: learning rate for alpha, it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    shallow_copy_attributes: ClassVar[set[str]] = {"replay_buffer", "_logger"}

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MOMlpPolicy": MOMlpPolicy,
        "MOGcnPolicy": MOGcnPolicy,
    }

    policy: MOSACPolicy
    actor: Actor
    critic: MOContinuousCritic
    critic_target: MOContinuousCritic

    actor_lr_schedule: Schedule
    critic_lr_schedule: Schedule
    alpha_lr_schedule: Schedule

    def __init__(
        self,
        policy: Union[str, type[MOSACPolicy]],
        env: Union[GymEnv, str],
        n_objs: int = -1,
        n_nodes: int = -1,
        local_action_dim: int = -1,
        global_action_dim: int = -1,
        actor_lr: Union[float, Schedule] = 3e-4,
        critic_lr: Union[float, Schedule] = 3e-4,
        alpha_lr: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 10,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Union[str, type[MOGraphReplayBuffer]]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        history_algorithms: Optional[list[SelfMOSAC]] = None,
        _init_setup_model: bool = True,
    ):
        self.text_logger = logging.getLogger("algo")

        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        replay_buffer_kwargs["n_objs"] = n_objs

        if policy_kwargs is None:
            policy_kwargs = {}

        self.actor_lr_schedule = get_schedule_fn(actor_lr)
        self.critic_lr_schedule = get_schedule_fn(critic_lr)
        self.alpha_lr_schedule = get_schedule_fn(alpha_lr)
        policy_kwargs["actor_lr_schedule"] = self.actor_lr_schedule
        policy_kwargs["critic_lr_schedule"] = self.critic_lr_schedule
        policy_kwargs["n_objs"] = n_objs
        policy_kwargs["n_nodes"] = n_nodes
        policy_kwargs["local_action_dim"] = local_action_dim
        policy_kwargs["global_action_dim"] = global_action_dim

        history_algorithms = history_algorithms if history_algorithms else []
        for history_algo in history_algorithms:
            history_algo.policy.set_training_mode(False)
        policy_kwargs["history_modules"] = [algo.policy for algo in history_algorithms]

        if replay_buffer_class is None:
            replay_buffer_class = MOGraphReplayBuffer

        super().__init__(
            policy,
            env,
            actor_lr,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.env_class = env
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.final_reward = np.zeros(2, dtype=np.float32)

        if _init_setup_model:
            self._setup_model()
        self._reset_env()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float64))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert (
                    init_value > 0.0
                ), "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _reset_replay_buffer(self) -> None:
        # Make a local copy as we should not pickle
        # the environment when using HerReplayBuffer
        replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
        if issubclass(self.replay_buffer_class, HerReplayBuffer):
            assert (
                self.env is not None
            ), "You must pass an environment when using `HerReplayBuffer`"
            replay_buffer_kwargs["env"] = self.env
        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **replay_buffer_kwargs,
        )

    def _reset_env(self) -> None:
        if self.env_class is not None:
            env = GraphVecEnv([lambda: maybe_make_env(self.env_class, self.verbose)])
            env = self._wrap_env(env, self.verbose, True)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            # get VecNormalize object if needed
            self._vec_normalize_env = unwrap_vec_normalize(env)

    def learn_mo(
        self: SelfMOSAC,
        total_timesteps: int,
        scalarization: ScalarizationFunction,
        callback: MaybeCallback = None,
        log_interval: int = None,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfMOSAC:
        assert self.env
        scalarization.to_device(self.device)

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        self.text_logger.info(
            "learn_mo: started, num_timesteps=%d, total_timesteps=%d",
            self.num_timesteps,
            total_timesteps,
        )

        # self._reset_env()
        # self._reset_replay_buffer()

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = (
                    self.gradient_steps
                    if self.gradient_steps >= 0
                    else rollout.episode_timesteps
                )
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train_mo(
                        batch_size=self.batch_size,
                        scalarization=scalarization,
                        gradient_steps=gradient_steps,
                    )

        # Fetch the final info in the replay buffer
        if self.replay_buffer:
            self.final_reward = self.replay_buffer.rewards[self.replay_buffer.pos - 1]
        self.env.reset()

        callback.on_training_end()

        return self

    def _update_info_buffer(
        self, infos: list[dict[str, Any]], dones: Optional[np.ndarray] = None
    ) -> None:
        super()._update_info_buffer(infos, dones)

        throughputs = []
        cores = []
        memories = []
        for info in infos:
            if "throughput" in info:
                throughputs.append(info["throughput"])
            if "core" in info:
                cores.append(info["core"])
            if "memory" in info:
                memories.append(info["memory"])

        if throughputs:
            self.logger.record("train/throughput_mean", np.mean(throughputs))
            self.logger.record("train/throughput_max", np.max(throughputs))
            self.logger.record("train/throughput_min", np.min(throughputs))
        if cores:
            self.logger.record("train/core_mean", np.mean(cores))
            self.logger.record("train/core_max", np.max(cores))
            self.logger.record("train/core_min", np.min(cores))
        if memories:
            self.logger.record("train/memory_mean", np.mean(memories))
            self.logger.record("train/memory_max", np.max(memories))
            self.logger.record("train/memory_min", np.min(memories))

    def train_mo(
        self,
        gradient_steps: int,
        scalarization: ScalarizationFunction,
        batch_size: int = 64,
    ) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate_with_custom_schedule(
            self.actor.optimizer, self.actor_lr_schedule, "actor_lr"
        )
        self._update_learning_rate_with_custom_schedule(
            self.critic.optimizer, self.critic_lr_schedule, "critic_lr"
        )
        if self.ent_coef_optimizer is not None:
            self._update_learning_rate_with_custom_schedule(
                self.ent_coef_optimizer, self.alpha_lr_schedule, "alpha_lr"
            )

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            # actions_pi: (batch_size)
            # log_prob: (batch_size)
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_log_prob = next_log_prob.reshape(-1, 1)  # shape: (batch_size, 1)
                # Compute the next Q values: min over all critics targets
                # Note: critic_target returns a list of q_values, each of dim (batch_size, n_objs)
                # next_q_values: (batch_size, n_objs, n_critic)
                next_q_values = th.stack(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=2,
                )
                # Select the minimum Q value over all critics
                # next_q_values: (batch_size, n_objs)
                next_q_values, _ = th.min(next_q_values, dim=2, keepdim=False)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob
                # td error + entropy term
                rewards = replay_data.rewards  # shape: (batch_size, n_objs)
                dones = replay_data.dones.reshape(-1, 1)  # shape: (batch_size, 1)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            # current_q_values: tuple[(batch_size, n_objs)]
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            # q_values_pi: (batch_size, n_objs, n_critic)
            q_values_pi = th.stack(
                self.critic(replay_data.observations, actions_pi), dim=2
            )
            # min_qf_pi: (batch_size, n_objs)
            min_qf_pi, _ = th.min(q_values_pi, dim=2, keepdim=False)
            weighted_min_qf_pi = scalarization.evaluate(min_qf_pi)

            actor_loss = (ent_coef * log_prob - weighted_min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _update_learning_rate_with_custom_schedule(
        self,
        optimizer: th.optim.Optimizer,
        lr_schedule: Schedule,
        lr_schedule_name: str,
    ) -> None:
        self.logger.record(
            f"train/{lr_schedule_name}", lr_schedule(self._current_progress_remaining)
        )
        update_learning_rate(optimizer, lr_schedule(self._current_progress_remaining))

    def _update_learning_rate(
        self, optimizers: Union[list[th.optim.Optimizer], th.optim.Optimizer]
    ) -> None:
        raise NotImplementedError(
            "Use _update_learning_rate_with_custom_schedule to update the learning rate using actor_lr and critic_lr "
            "respectively."
        )

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer. Specially adapted for MOSAC that the actually
        stored reward is info['objs'], a tuple of (obj_1, obj_2).

        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            # new_obs_ = self._vec_normalize_env.get_original_obs()
            # reward_ = self._vec_normalize_env.get_original_reward()
            raise ValueError(
                "MOSAC does not support VecNormalize, please set `vec_normalize_env` to None."
            )
        else:
            # Avoid changing the original ones
            # Note that the objectives are stored in info['objs'], and reward is not used
            # This is to comply with the VecEnv which assumes reward is a scalar
            # reward_: (batch_size, n_objs)
            reward_ = np.stack([info["objs"] for info in infos], axis=0)
            for info in infos:
                info.pop("objs", None)
            self._last_original_obs, new_obs_ = self._last_obs, new_obs

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(
                            next_obs[i, :]
                        )  # type: ignore[assignment]

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
        ]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def learn(
        self: SelfOffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:
        raise NotImplementedError("Use learn_mo instead.")

    def train(self, gradient_steps: int, batch_size: int) -> None:
        raise NotImplementedError("Use train_mo instead.")

    def __deepcopy__(self, memodict={}):
        # shallow copy replay buffer, deep copy the rest
        # Create a new instance of the class
        cls = self.__class__
        result = cls.__new__(cls)

        # Deep copy all attributes except the replay buffer
        memodict[id(self)] = result
        for key, value in self.__dict__.items():
            if key in self.shallow_copy_attributes:
                # Shallow copy the replay buffer
                setattr(result, key, value)
            else:
                # Deep copy other attributes
                try:
                    setattr(result, key, deepcopy(value, memodict))
                except TypeError as e:
                    print(f"Error copying attribute {key}: {e}")
                    raise e

        return result

    @classmethod
    def load(  # noqa: C901
        cls: type[SelfMOSAC],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[Union[GymEnv, str]] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfMOSAC:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs"].get("net_arch")
            if (
                saved_net_arch
                and isinstance(saved_net_arch, list)
                and isinstance(saved_net_arch[0], dict)
            ):
                data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if (
            "policy_kwargs" in kwargs
            and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify new environments"
            )

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if isinstance(env, gym.Env):
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(
                env, data["observation_space"], data["action_space"]
            )
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        elif "env" in data:
            env = data["env"]

        if force_reset and data is not None:
            data["_last_obs"] = None
            data["num_timesteps"] = 0
            data["_total_timesteps"] = 0
            data["_num_timesteps_at_start"] = 0

        model: SelfMOSAC = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]

        return model
