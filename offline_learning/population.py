import numpy as np
from offline_learning.sample import Sample
from offline_learning.scalarization_methods import (
    ScalarizationFunction,
    WeightedSumScalarization,
)
from utils.config import RootConfig


class Population:
    def __init__(self, conf: RootConfig):
        self.conf = conf
        self.samples: list[Sample] = []

    def update(self, sample_batch: list[Sample]):
        self.samples.extend(sample_batch)

    @staticmethod
    def calculate_pareto(samples: list[Sample]) -> list[Sample]:
        if not samples:
            return []

        pareto = []

        for i, sample in enumerate(samples):
            flag = True
            for j, other_sample in enumerate(samples):
                if i != j and Population.dominate(other_sample, sample):
                    flag = False
                    break
            if flag:
                pareto.append(sample)
        return pareto

    @staticmethod
    def compute_hypervolume(samples: list[Sample], sample: Sample):
        """Compute the hypervolume of the batch with the candidate sample added."""
        all_samples = samples + [sample]
        pareto = Population.calculate_pareto(all_samples)
        if len(pareto) < 2:
            return 0.0
        pareto.sort(key=lambda x: (x.objs[0], x.objs[1]))
        sum = 0
        for i in range(len(pareto) - 1):
            sum += (pareto[i + 1].objs[0] - pareto[i].objs[0]) * (
                pareto[i + 1].objs[1] + pareto[i].objs[1]
            )
        return sum

    @staticmethod
    def compute_sparsity(samples: list[Sample], sample: Sample):
        """Compute the sparsity of the batch with the candidate sample added."""
        all_samples = samples + [sample]
        pareto = Population.calculate_pareto(all_samples)
        if len(pareto) < 2:
            return 0.0

        pareto.sort(key=lambda x: (x.objs[0], x.objs[1]))
        sum = 0
        for i in range(len(pareto) - 1):
            sum += (pareto[i + 1].objs[0] - pareto[i].objs[0]) ** 2
        return sum / (len(pareto) - 1)

    @staticmethod
    def dominate(lhs: Sample, rhs: Sample) -> bool:
        assert (
            lhs.objs is not None and rhs.objs is not None
        ), "Objectives must be defined for comparison."
        return bool(np.all(lhs.objs >= rhs.objs) and np.any(lhs.objs > rhs.objs))

    def select_elite(self):
        N = self.conf.tuner.num_tasks
        pareto = self.calculate_pareto(self.samples)

        if len(pareto) < N:
            print(f"Pareto size = {len(pareto)} < num_tasks {N}, return all tasks")
            elite_batch = pareto
            scalarization_batch = self.get_scalarization(pareto)
        else:
            # First, select the one with the best throughput
            elite_batch = []
            selected_ids = set()

            best_throughput = 0
            best_sample_id = -1
            for i, sample in enumerate(pareto):
                if sample.objs is None:
                    raise ValueError(
                        f"Sample {sample.id} has no objectives, please check the sample generation process."
                    )
                if best_sample_id == -1 or sample.objs[0] > best_throughput:
                    best_throughput = sample.objs[0]
                    best_sample_id = i

            elite_batch.append(pareto[best_sample_id])
            selected_ids.add(best_sample_id)

            # Then, select the rest N-1 samples based on hypervolume and sparsity
            for _ in range(N - 1):
                max_metrics, best_id = -np.inf, -1
                for i in range(len(pareto)):
                    if i in selected_ids:
                        continue
                    hv = self.compute_hypervolume(elite_batch, pareto[i])
                    sparsity = self.compute_sparsity(elite_batch, pareto[i])
                    metrics = hv - sparsity
                    if metrics > max_metrics:
                        max_metrics, best_id = metrics, i

                if best_id == -1:
                    print("No more candidates to select")
                    break

                elite_batch.append(pareto[best_id])
            scalarization_batch = self.get_scalarization(elite_batch)

        return elite_batch, scalarization_batch

    @staticmethod
    def get_scalarization(samples: list[Sample]) -> list[ScalarizationFunction]:
        scalarization_batch = []
        for sample in samples:
            if sample.objs[0] == -1 and sample.objs[1] == -1:
                weights = np.array([1, 1])
            else:
                weights = np.array(
                    [1 / obj if obj > 1e-3 else 10 for obj in sample.objs.flatten()]
                )
            scalarization_batch.append(
                WeightedSumScalarization(2, weights / np.sum(weights))
            )
        return scalarization_batch
