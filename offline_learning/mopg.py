import logging
import time
from copy import deepcopy
from typing import Any, Callable, Optional, Union

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm, maybe_make_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import (
    VecMonitor,
    is_vecenv_wrapped,
)

from environment.graph_vec_env import GraphVecEnv
from model.buffer import batch_graph_obs
from offline_learning.sample import Sample
from offline_learning.task import Task
from offline_learning.util import print_info
from utils.config import RootConfig


def evaluate(
    algo: BaseAlgorithm,
    env: Union[GymEnv, str],
    log_level: str,
    n_steps: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    return_raw_objs: bool = False,
    warn: bool = True,
    verbose: bool = False,
    sb3_logger: Optional[SB3Logger] = None,
) -> Union[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param algo: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_steps: Number of steps to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_raw_objs: If True, a list of objectives and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean objectives per step, std of objectives per step.
        Returns a list of objectives per step when ``return_episode_rewards`` is True.
    """
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    logger = logging.getLogger("eval")

    if isinstance(env, str):
        env = GraphVecEnv([lambda: maybe_make_env(env, verbose)])

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        # Wrap the environment with Monitor
        env = VecMonitor(env, None)

    n_envs = env.num_envs
    objs = []
    throughputs = []
    cores = []
    memories = []
    env.unwrapped.logger = logger

    episode_counts = np.zeros(n_envs, dtype="int")

    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    logger.info("Evaluating starts.")
    for step in range(n_steps):
        logger.debug("Evaluation %d/%d", step + 1, n_steps)

        node_features, edge_indices, batch_vector = batch_graph_obs(
            observations["node_features"], observations["adjacency_matrix"], algo.device
        )

        sparse_obs = {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "batch_vector": batch_vector,
        }

        actions, states = algo.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        # Shape: (n_envs, n_objs)
        objs.append(np.stack([info["objs"] for info in infos], axis=0))
        throughputs.append(infos[0]["throughput"])
        cores.append(infos[0]["core"])
        memories.append(infos[0]["memory"])

        for i in range(n_envs):
            if callback is not None:
                callback(locals(), globals())
        observations = new_observations

        if render:
            env.render()
    logger.info("Evaluation ends.")

    if return_raw_objs:
        return objs

    objs = np.array(objs)  # (n_steps, n_envs, n_objs)
    mean_objs = np.mean(objs, axis=0)
    std_objs = np.std(objs, axis=0)

    if sb3_logger:
        for i in range(mean_objs.shape[1]):
            sb3_logger.record(f"eval/obj_{i}", np.mean(mean_objs[:, i]))
        sb3_logger.record("eval/throughput", np.mean(throughputs))
        sb3_logger.record("eval/core", np.mean(cores))
        sb3_logger.record("eval/memory", np.mean(memories))

    return mean_objs, std_objs


def MOPG_worker(
    conf: RootConfig,
    task_id,
    task: Task,
    iteration: int,
    num_episodes: int,
    start_time: float,
    parallel: bool = False,
    sb3_logger: Optional[SB3Logger] = None,
) -> list[dict[str, Any]]:
    """
    Runs an MOPG iteration.

    :param args: the arguments include necessary SAC parameters.
    :param task_id: the task_id of the task to be optimized
    :param task: the task to be optimized
    :param iteration: the current iteration
    :param num_episodes: number of episodes to run
    :param start_time: starting time
    :param parallel: whether this worker is run in parallel
    :return: a list of intermediate/final results. Each result is a dictionary which contains the task id, the offspring
             batch, and whether the task is done or not.
    """
    logger = logging.getLogger("trainer")

    algo = task.sample.algo
    scalarization = task.scalarization

    total_num_episodes = int(conf.tuner.max_steps) // conf.tuner.episode_length
    start_iter, final_iter = iteration, min(
        iteration + num_episodes, total_num_episodes
    )
    logger.debug(
        "MOPG worker: task_id=%d, start_iter=%d, final_iter=%d",
        task_id,
        start_iter,
        final_iter,
    )

    offspring_batch = []
    results_queue = []

    for j in range(start_iter, final_iter):
        algo.learn_mo(
            conf.tuner.episode_length, scalarization, reset_num_timesteps=False
        )

        if (j + 1) * conf.tuner.episode_length >= algo.learning_starts:
            # Obtain the final reward from the training phase
            objs = algo.final_reward.flatten()  # shape: (n_objs,)
            sample = Sample(deepcopy(algo), objs)
            offspring_batch.append(sample)

            # put results back every update_iter iterations, to avoid the multi-processing crash
            if (j + 1) % conf.tuner.update_iter == 0 or j == final_iter - 1:
                offspring_batch = np.array(offspring_batch)
                results = {}
                results["task_id"] = task_id
                results["offspring_batch"] = offspring_batch
                if j == final_iter - 1:
                    results["done"] = True
                else:
                    results["done"] = False
                results_queue.append(results)
                offspring_batch = []

        if conf.tuner.log_iter > 0 and (j + 1) % conf.tuner.log_iter == 0:
            algo.dump_logs()
            total_num_steps = (j + 1) * conf.tuner.episode_length
            end = time.time()
            if not parallel or task_id == 0:
                logger.info(
                    "[RL] Updates %d, task %d, num of timesteps %d, FPS %.2f, time %.2f seconds",
                    j + 1,
                    task_id,
                    total_num_steps,
                    total_num_steps / (end - start_time),
                    end - start_time,
                )

    return results_queue
