"""
Single-objective RL for testing model performance.
"""

import logging
import os
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from offline_learning.mopg import MOPG_worker, evaluate
from offline_learning.opt_graph import OptGraph
from offline_learning.sample import Sample
from offline_learning.scalarization_methods import WeightedSumScalarization
from offline_learning.task import Task
from offline_learning.util import print_info
from model.sac import MOSAC
from utils.config import RootConfig, save_to_yaml


def run(conf: RootConfig):
    # --------------------> Preparation <-------------------- #
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    logger = logging.getLogger("train")

    weights = conf.env.single_objective_weight
    if len(weights) != 2:
        raise ValueError("Weights must be a list of length 2.")
    weights = torch.tensor(weights, dtype=torch.float64)
    scalarization = WeightedSumScalarization(num_objs=2, weights=weights.to(device))

    total_num_episodes = int(conf.tuner.max_steps) // conf.tuner.episode_length

    start_time = time.time()

    episode = 0
    iteration = 0
    opt_graph = OptGraph()

    # Initialize task
    sb3_logger = configure(
        os.path.join(conf.tuner.save_dir, "log"), ["stdout", "log", "tensorboard"]
    )

    if conf.tuner.load_dir:
        logger.info("Loading model from %s", conf.tuner.load_dir)
        history_algos = [
            MOSAC.load(
                os.path.join(conf.tuner.load_dir, "final", "EP_policy_{}".format(0)),
                device=device,
            )
        ]
    else:
        history_algos = None

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
        learning_starts=conf.tuner.warmup_iter * conf.tuner.episode_length,
        ent_coef=f"auto_{conf.tuner.initial_alpha}",
        tau=conf.tuner.tau,
        gamma=conf.tuner.gamma,
        batch_size=conf.tuner.train_batch_size,
        device=device,
        tensorboard_log=os.path.join(conf.tuner.save_dir, "tensorboard"),
        history_algorithms=history_algos,
        gradient_steps=conf.tuner.gradient_steps,
    )

    algo.set_logger(sb3_logger)
    objs, _ = evaluate(
        algo, conf.mode.env, conf.log.levels.algo, 5
    )  # objs shape: (num_envs, num_objs)
    sample = Sample(algo, objs[0], optgraph_id=-1)
    sample.optgraph_id = opt_graph.insert(weights, deepcopy(objs[0]), -1)
    task_id = 0
    prev_node_id = sample.optgraph_id
    task = Task(sample, scalarization, copy=False)
    last_sample = sample

    rl_num_episodes = conf.tuner.warmup_iter

    while iteration < total_num_episodes:
        episode += 1
        logger.info("\n-------------------- Episode %d --------------------", episode)

        # --------------------> RL Optimization <-------------------- #
        results_queue = MOPG_worker(
            conf,
            task_id,
            task,
            iteration,
            rl_num_episodes,
            start_time,
            False,
            sb3_logger=sb3_logger,
        )
        last_sample = results_queue[-1]["offspring_batch"][-1]
        prev_node_id = opt_graph.insert(
            weights, deepcopy(last_sample.objs), prev_node_id
        )
        logger.info(
            "objs = %s, scalar = %f",
            last_sample.objs,
            scalarization.evaluate(last_sample.objs),
        )

        iteration = min(iteration + rl_num_episodes, total_num_episodes)
        rl_num_episodes = conf.tuner.update_iter

        # ----------------------> Save Results <---------------------- #
        # save final obj as ep
        ep_dir = os.path.join(conf.tuner.save_dir, str(iteration), "ep")
        os.makedirs(ep_dir, exist_ok=True)
        with open(os.path.join(ep_dir, "objs.txt"), "w") as fp:
            fp.write(("{:5f}" + 1 * ",{:5f}" + "\n").format(*last_sample.objs))

    # ----------------------> Save Final Model <----------------------

    os.makedirs(os.path.join(conf.tuner.save_dir, "final"), exist_ok=True)

    # save ep policies & env_params
    last_sample.algo.save(
        os.path.join(conf.tuner.save_dir, "final", "EP_policy_{}".format(0)),
    )

    # save all ep objectives
    with open(os.path.join(conf.tuner.save_dir, "final", "objs.txt"), "w") as fp:
        fp.write(("{:5f}" + 1 * ",{:5f}" + "\n").format(*last_sample.objs))


def tune(conf: RootConfig):
    """
    Tune the hyperparameters of the MOSAC algorithm.
    :param actor_lr: List of actor learning rates to try.
    :param critic_lr: List of critic learning rates to try.
    :param alpha_lr: List of alpha learning rates to try.
    :param initial_alpha: List of initial alpha values to try.
    """

    logger = logging.getLogger("trainer")

    actor_lr = [1e-5, 1e-4, 1e-3, 1e-2]
    critic_lr = [1e-5, 1e-4, 1e-3, 1e-2]
    alpha_lr = [1e-5, 1e-4, 1e-3, 1e-2]
    initial_alpha = [1.0, 0.1, 0.01]

    save_dir_base = conf.tuner.save_dir

    for a_lr in actor_lr:
        for c_lr in critic_lr:
            for al_lr in alpha_lr:
                for init_al in initial_alpha:
                    logger.info("")
                    logger.info("==========================================")
                    logger.info(
                        "Tuning with alpha_lr = %.6f, actor_lr = %.6f, critic_lr = %.6f, initial_alpha = %.6f",
                        a_lr,
                        c_lr,
                        al_lr,
                        init_al,
                    )
                    conf.tuner.alpha_lr = al_lr
                    conf.tuner.actor_lr = a_lr
                    conf.tuner.critic_lr = c_lr
                    conf.tuner.initial_alpha = init_al

                    conf.tuner.save_dir = os.path.join(
                        save_dir_base,
                        f"a_lr_{a_lr}_c_lr_{c_lr}_al_lr_{al_lr}_init_al_{init_al}",
                    )
                    os.makedirs(conf.tuner.save_dir, exist_ok=True)
                    os.makedirs(os.path.join(conf.tuner.save_dir, "log"), exist_ok=True)
                    os.makedirs(
                        os.path.join(conf.tuner.save_dir, "tensorboard"), exist_ok=True
                    )
                    # save arguments
                    save_to_yaml(conf, os.path.join(conf.tuner.save_dir, "args.txt"))
                    conf.env.savepoint_store = os.path.join(
                        conf.tuner.save_dir, "savepoint_store.db"
                    )

                    # Here you would call the run function with the appropriate configuration
                    run(conf)
