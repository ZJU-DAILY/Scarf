import logging
import os

import gymnasium as gym
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecMonitor, is_vecenv_wrapped

from environment.graph_vec_env import GraphVecEnv
from offline_learning.mopg import evaluate
from offline_learning.sample import Sample
from offline_learning.scalarization_methods import WeightedSumScalarization
from offline_learning.util import generate_weights_batch_dfs
from model.sac import MOSAC
from utils.config import RootConfig

"""
initialize_warm_up_batch: method to generate tasks in the warm-up stage.
Each task is a pair of an initial random policy and an evenly distributed optimization weight.
The optimization weight is represented by a weighted-sum scalarization function.
"""

logger = logging.getLogger("trainer")


def initialize_warm_up_batch(conf: RootConfig, device):
    # using evenly distributed weights for warm-up stage
    weights_batch = [[0.01, 0.99], [0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.99, 0.1]]
    # generate_weights_batch_dfs(
    #     0,
    #     2,
    #     conf.tuner.min_weight,
    #     conf.tuner.max_weight,
    #     conf.tuner.delta_weight,
    #     [],
    #     weights_batch,
    # )
    sample_batch = []
    scalarization_batch = []

    for weights in weights_batch:
        # TODO: SAC parameters
        algo = MOSAC(
            conf.tuner.policy_class,
            conf.mode.env,
            2,
            len(conf.knobs.operator_names),
            len(conf.knobs.operator_knobs),
            len(conf.knobs.cluster_knobs),
            actor_lr=conf.tuner.actor_lr,
            critic_lr=conf.tuner.critic_lr,
            alpha_lr=conf.tuner.alpha_lr,
            ent_coef=f"auto_{conf.tuner.initial_alpha}",
            tau=conf.tuner.tau,
            gamma=conf.tuner.gamma,
            batch_size=conf.tuner.train_batch_size,
            device=device,
            learning_starts=conf.tuner.warmup_iter * conf.tuner.episode_length,
        )

        scalarization = WeightedSumScalarization(num_objs=2, weights=weights)

        objs, _ = evaluate(algo, conf.mode.env, conf.log.levels.algo, n_steps=1)
        sample = Sample(algo, objs[0], optgraph_id=-1)

        sample_batch.append(sample)
        scalarization_batch.append(scalarization)

    return sample_batch, scalarization_batch


def _get_env(env: str | GymEnv, verbose=False):
    if isinstance(env, str):
        env = GraphVecEnv([lambda: maybe_make_env(env, verbose)])

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped:
        # Wrap the environment with Monitor
        env = VecMonitor(env, None)
    return env


def create_pnn_with_history(conf: RootConfig, device):
    assert conf.tuner.load_dir is not None
    base_dir = os.path.join(conf.tuner.load_dir, "final")

    # Read objs
    objs_list = []
    with open(os.path.join(base_dir, "objs.txt"), "r") as f:
        for line in f:
            objs_list.append([float(x) for x in line.strip().split(",")])

    sample_batch = []
    scalarization_batch = []

    for i, objs in enumerate(objs_list):
        env = conf.mode.env
        history_algo = MOSAC.load(
            os.path.join(base_dir, f"EP_policy_{i}"),
            env=env,
            device=device,
        )

        weights = [1 / objs[0], 1 / objs[1]]
        scalarization = WeightedSumScalarization(num_objs=2, weights=weights)

        algo = MOSAC(
            conf.tuner.policy_class,
            conf.mode.env,
            2,
            len(conf.knobs.operator_names),
            len(conf.knobs.operator_knobs),
            len(conf.knobs.cluster_knobs),
            actor_lr=conf.tuner.actor_lr,
            critic_lr=conf.tuner.critic_lr,
            alpha_lr=conf.tuner.alpha_lr,
            ent_coef=f"auto_{conf.tuner.initial_alpha}",
            tau=conf.tuner.tau,
            gamma=conf.tuner.gamma,
            batch_size=conf.tuner.train_batch_size,
            device=device,
            learning_starts=conf.tuner.warmup_iter * conf.tuner.episode_length,
            history_algorithms=[history_algo],
        )
        # algo = history_algo

        objs, _ = evaluate(algo, conf.mode.env, conf.log.levels.algo, n_steps=1)
        sample = Sample(algo, objs[0], optgraph_id=-1)

        sample_batch.append(sample)
        scalarization_batch.append(scalarization)
        logger.info(
            "Loaded sample %i from %s", i, os.path.join(base_dir, f"EP_policy_{i}")
        )
    return sample_batch, scalarization_batch
