import logging

import os
import time
from copy import deepcopy
from logging import Logger
from typing import Any

import numpy as np
import torch

from offline_learning.ep import EP
from offline_learning.mopg import MOPG_worker
from offline_learning.opt_graph import OptGraph
from offline_learning.population import Population
from offline_learning.sample import Sample
from offline_learning.scalarization_methods import WeightedSumScalarization
from offline_learning.task import Task
from offline_learning.warm_up import create_pnn_with_history, initialize_warm_up_batch
from utils.config import RootConfig


def run(conf: RootConfig, parallel=False):
    # --------------------> Preparation <-------------------- #
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    logger = logging.getLogger("trainer")

    # build a scalarization template
    scalarization_template = WeightedSumScalarization(
        num_objs=2, weights=np.ones(2) / 2
    )

    total_num_updates = int(conf.tuner.max_steps) // conf.tuner.episode_length

    start_time = time.time()

    # ------------------> Initialize or Recover from Checkpoint <------------------ #

    # initialize ep and population and opt_graph
    population = Population(conf)
    opt_graph = OptGraph()

    if conf.tuner.load_dir is None:
        # Construct tasks for warm up
        elite_batch, scalarization_batch = initialize_warm_up_batch(conf, device)
    else:
        elite_batch, scalarization_batch = create_pnn_with_history(conf, device)
    rl_num_updates = conf.tuner.warmup_iter
    for sample, scalarization in zip(elite_batch, scalarization_batch):
        sample.optgraph_id = opt_graph.insert(
            deepcopy(scalarization.weights), deepcopy(sample.objs), -1
        )

    episode = 0
    iteration = 0
    while iteration < total_num_updates:
        if episode == 0:
            logger.info(
                "------------------------------- Warm-up Stage -------------------------------"
            )
        else:
            logger.info(
                "-------------------- Evolutionary Stage: Generation %d --------------------",
                episode,
            )

        episode += 1

        # --------------------> RL Optimization <-------------------- #
        # compose task for each elite
        task_batch = []
        for elite, scalarization in zip(elite_batch, scalarization_batch):
            task_batch.append(
                Task(elite, scalarization)
            )  # each task is a (policy, weight) pair

        # run MOPG for each task sequentially
        results_queue: list[dict[str, Any]] = []

        if parallel:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        MOPG_worker,
                        conf,
                        task_id,
                        task,
                        iteration,
                        rl_num_updates,
                        start_time,
                        True,
                    )
                    for task_id, task in enumerate(task_batch)
                ]
                for future in futures:
                    results_queue.extend(future.result())
        else:
            for task_id, task in enumerate(task_batch):
                logger.info(f"Training task %d", id(task.sample.algo))
                results_queue.extend(
                    MOPG_worker(
                        conf,
                        task_id,
                        task,
                        iteration,
                        rl_num_updates,
                        start_time,
                        False,
                    )
                )

        # collect MOPG results for offsprings and insert objs into objs buffer
        all_offspring_batch = [[] for _ in range(len(task_batch))]
        cnt_done_workers = 0
        for rl_results in results_queue:
            task_id, offsprings = rl_results["task_id"], rl_results["offspring_batch"]
            for sample in offsprings:
                all_offspring_batch[task_id].append(Sample.copy_from(sample))
            if rl_results["done"]:
                cnt_done_workers += 1

        if cnt_done_workers != len(task_batch):
            logger.error(
                "Done workers %d and task number %d do not match",
                cnt_done_workers,
                len(task_batch),
            )

        # put all intermediate policies into all_sample_batch for EP update
        all_sample_batch: list[Sample] = []
        # store the last policy for each optimization weight for RA
        last_offspring_batch = [None] * len(task_batch)
        # only the policies with iteration % update_iter = 0 are inserted into offspring_batch for population update
        # after warm-up stage, it's equivalent to the last_offspring_batch
        offspring_batch: list[Sample] = []
        for task_id in range(len(task_batch)):
            offsprings = all_offspring_batch[task_id]
            logger.debug(f"Task %d offsprings size: %d", task_id, len(offsprings))
            prev_node_id = task_batch[task_id].sample.optgraph_id
            opt_weights = (
                deepcopy(task_batch[task_id].scalarization.weights).detach().numpy()
            )
            for i, sample in enumerate(offsprings):
                all_sample_batch.append(sample)
                if (i + 1) % conf.tuner.update_iter == 0:
                    prev_node_id = opt_graph.insert(
                        opt_weights, deepcopy(sample.objs), prev_node_id
                    )
                    sample.optgraph_id = prev_node_id
                    offspring_batch.append(sample)
            last_offspring_batch[task_id] = offsprings[-1]

        # -----------------------> Update EP <----------------------- #
        # update EP and population
        population.update(all_sample_batch)
        logger.info(
            "offspring_batch size: %d, update_iter: %d, population size: %d",
            len(offspring_batch),
            conf.tuner.update_iter,
            len(population.samples),
        )

        # ------------------- > Task Selection <--------------------- #
        # (
        #     elite_batch,
        #     scalarization_batch,
        #     predicted_offspring_objs,
        # ) = population.prediction_guided_selection(
        #     conf, iteration, ep, opt_graph, scalarization_template
        # )

        elite_batch, scalarization_batch = population.select_elite()
        predicted_offspring_objs = None

        logger.info("Selected Tasks:")
        for i in range(len(elite_batch)):
            logger.info(
                "id = %d, objs = %s, weight = %s",
                elite_batch[i].id,
                elite_batch[i].objs,
                scalarization_batch[i].weights,
            )

        iteration = min(iteration + rl_num_updates, total_num_updates)

        rl_num_updates = conf.tuner.update_iter

        # ----------------------> Save Results <---------------------- #
        # save ep
        # ep_dir = os.path.join(conf.tuner.save_dir, str(iteration), "ep")
        # os.makedirs(ep_dir, exist_ok=True)
        # with open(os.path.join(ep_dir, "objs.txt"), "w") as fp:
        #     for obj in ep.obj_batch:
        #         fp.write(("{:5f}" + 1 * ",{:5f}" + "\n").format(*obj))

        # save population
        population_dir = os.path.join(conf.tuner.save_dir, str(iteration), "population")
        os.makedirs(population_dir, exist_ok=True)
        with open(os.path.join(population_dir, "objs.txt"), "w") as fp:
            for sample in population.samples:
                fp.write(("{:5f}" + 1 * ",{:5f}" + "\n").format(*(sample.objs)))
        # save optgraph and node id for each sample in population
        with open(os.path.join(population_dir, "optgraph.txt"), "w") as fp:
            fp.write("{}\n".format(len(opt_graph.objs)))
            for i in range(len(opt_graph.objs)):
                fp.write(
                    ("{:5f}" + 1 * ",{:5f}" + ";{:5f}" + 1 * ",{:5f}" + ";{}\n").format(
                        *(opt_graph.weights[i]), *(opt_graph.objs[i]), opt_graph.prev[i]
                    )
                )
            fp.write("{}\n".format(len(population.samples)))
            for sample in population.samples:
                fp.write("{}\n".format(sample.optgraph_id))

        # save elites
        elite_dir = os.path.join(conf.tuner.save_dir, str(iteration), "elites")
        os.makedirs(elite_dir, exist_ok=True)
        with open(os.path.join(elite_dir, "elites.txt"), "w") as fp:
            for elite in elite_batch:
                fp.write(("{:5f}" + 1 * ",{:5f}" + "\n").format(*(elite.objs)))
        with open(os.path.join(elite_dir, "weights.txt"), "w") as fp:
            for scalarization in scalarization_batch:
                fp.write(
                    ("{:5f}" + 1 * ",{:5f}" + "\n").format(*(scalarization.weights))
                )

        if predicted_offspring_objs is not None:
            with open(os.path.join(elite_dir, "predictions.txt"), "w") as fp:
                for objs in predicted_offspring_objs:
                    fp.write(("{:5f}" + 1 * ",{:5f}" + "\n").format(*(objs)))
        with open(os.path.join(elite_dir, "offsprings.txt"), "w") as fp:
            for i in range(len(all_offspring_batch)):
                for j in range(len(all_offspring_batch[i])):
                    fp.write(
                        ("{:5f}" + 1 * ",{:5f}" + "\n").format(
                            *(all_offspring_batch[i][j].objs)
                        )
                    )

    # ----------------------> Save Final Model <----------------------

    os.makedirs(os.path.join(conf.tuner.save_dir, "final"), exist_ok=True)
    pareto = population.calculate_pareto(population.samples)

    # save ep policies & env_params
    for i, sample in enumerate(pareto):
        sample.algo.save(
            os.path.join(conf.tuner.save_dir, "final", "EP_policy_{}".format(i)),
        )

    # save all ep objectives
    with open(os.path.join(conf.tuner.save_dir, "final", "objs.txt"), "w") as fp:
        for i, sample in enumerate(pareto):
            fp.write(("{:5f}" + 1 * ",{:5f}" + "\n").format(*(sample.objs)))
