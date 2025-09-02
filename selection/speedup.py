from typing import Any
import numpy as np
from sklearn.cluster import KMeans

from flink.yarn_connector import YarnFlinkConnector
from offline_learning.run import get_flink_connector_and_job
from selection.offline import load_observations, sample_observation
from utils.config import RootConfig


type JobObservations = list[tuple[dict[str, Any], float, float, float]]


# FILL IN HERE
history_workload_directories = []


def sim(v1: np.ndarray, v2: np.ndarray) -> float:
    # cosine similarity
    # v1: (x1, y)
    # v2: (x2, y)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_knob_response_vector(
    observations: JobObservations,
    selected_knob_vector_indices: list[int] | None = None
) -> np.ndarray:
    # observations: list of (knob_values, throughput, core, memory)
    vector = []
    for i, obs in enumerate(observations):
        if selected_knob_vector_indices is None or i in selected_knob_vector_indices:
            vector.append(obs[1])  # throughput
    return np.array(vector)


def select_knob_vectors(observations_list: list[JobObservations]) -> list[int]:
    num_knob_vectors = len(observations_list[0])
    num_jobs = len(observations_list)
    knob_vector_responses = []  # (num_knob_vectors, num_jobs)
    for i in range(num_knob_vectors):
        vec = []
        for j in range(num_jobs):
            vec.append(observations_list[j][i][1])
        knob_vector_responses.append(vec)
    
    variance = np.var(knob_vector_responses, axis=1)  # (num_knob_vectors)
    indices = np.argsort(-variance)
    selected_indices = indices[:num_jobs]
    return selected_indices.tolist()


def cluster(
    observations_list: list[JobObservations],
    num_clusters=5,
):
    num_jobs = len(observations_list)
    vecs = [get_knob_response_vector(obs) for obs in observations_list]

    X = np.asarray(vecs, dtype=float)  # (num_jobs, num_knob_vectors)

    k = int(num_clusters) if num_clusters is not None else 5
    k = max(1, min(k, num_jobs))

    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X)
    centers = km.cluster_centers_  # (k, num_dim)

    centroids: list[int] = []
    for c in range(k):
        center = centers[c]
        sims = sim(X, center)
        best_idx = int(np.argmax(sims))
        centroids.append(best_idx)

    return centroids


def assign_observation(
    observations: JobObservations,
    observations_list: list[JobObservations],
    centroids: list[int],
    selected_knob_vector_indices: list[int]
):
    # Assign an observation to the nearest centroid based on throughput
    obs_vector = get_knob_response_vector(observations, selected_knob_vector_indices)
    sims = [sim(obs_vector, np.array(observations_list[centroid][1])) for centroid in centroids]
    best_idx = int(np.argmax(sims))
    return best_idx


def main(conf: RootConfig):
    observations_list = []
    for dir in history_workload_directories:
        observations = load_observations(dir)
        observations_list.append(observations)

    selected_knob_vector_indices = select_knob_vectors(observations_list)
    centroids = cluster(observations_list)

    connector: YarnFlinkConnector
    connector, job = get_flink_connector_and_job(conf)  # type: ignore
    observations = []

    knob_vectors = [obs[0] for obs in observations_list[centroids[0]]]
    for i, knob_vector in enumerate(knob_vectors):
        if i in selected_knob_vector_indices:
            obs = sample_observation(connector, job, knob_vector, conf, conf.job.offline_source_rate)
            observations.append(obs)

    centroid = assign_observation(observations, observations_list, centroids, selected_knob_vector_indices)
    print("Selected centroid:", observations_list[centroid])
    print("Logdir: " + history_workload_directories[centroid])
