import numpy as np
import networkx as nx

from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

import pyTrajectory.config
import pyTrajectory.trajectory


def link_trajectories(data,
                      max_lag=30,
                      max_distance=150,
                      instance_position_func,
                      similarity_features=[],
                      feature_distances=[],
                      dissimilarity_weight=1,
                      min_trajectory_length=2):

    key_time_stamp = pyTrajectory.config.cfg.key_time_stamp
    key_category = pyTrajectory.config.cfg.key_category

    if type(dissimilarity_weight) in [int, float]:
        if len(similarity_features) > 0:
            dissimilarity_weight /= len(similarity_features)
        dissimilarity_weight = [dissimilarity_weight] * len(similarity_features)

    assert len(feature_distances) == len(similarity_features)
    assert len(dissimilarity_weight) == len(similarity_features)

    max_dissimilarity = [1 for _ in range(len(similarity_features))]

    active_trajectories = []
    archived_trajectories = []

    for time_stamp in tqdm(np.arange(data[key_time_stamp].min(), data[key_time_stamp].max() + 1)):

        # archive active trajectories with too high lag
        for idx, trajectory in enumerate([trajectory for trajectory in active_trajectories]):
            if time_stamp - trajectory[-1][key_time_stamp] > max_lag:
                active_trajectories.remove(trajectory)
                if len(trajectory) < min_trajectory_length:
                    # discard if too short
                    continue
                archived_trajectories.append(trajectory)

        # get current instances
        instances = data.slice_window(time_stamp, time_stamp, check_completeness=False)
        cleared_instances = []
        new_trajectories = []

        if len(instances) == 0:
            # nothing to do
            continue

        if len(active_trajectories) == 0:

            # either first frame or all other trajectories were archived
            new_trajectories = [pyTrajectory.trajectory.Trajectory([instance]) for instance in instances]
            cleared_instances = [instance for instance in instances]

        else:

            # otherwise, find optimal assignments from active trajectories to new instances (same category)
            distance_matrix = pairwise_distances(
                                    np.concatenate([np.array([instance_position_func(trajectory[-1])
                                                              for trajectory in active_trajectories]),
                                                    np.array([instance_position_func(instance) for instance in instances])]))
            category_matches = pairwise_distances(
                                    np.concatenate([np.array([trajectory[-1][key_category]
                                                              for trajectory in active_trajectories]).reshape(-1, 1),
                                                    np.array([instance[key_category] for instance in instances]).reshape(-1, 1)]))
            category_matches = category_matches == 0
            distance_matrix[~category_matches] = np.inf

            # create a graph and determine the connected components within max_distance
            num_active_trajectories = len(active_trajectories)
            distance_matrix[:num_active_trajectories, :num_active_trajectories] = np.inf
            distance_matrix[num_active_trajectories:, num_active_trajectories:] = np.inf
            G = nx.from_numpy_array(distance_matrix <= max_distance)
            components = []
            for g in nx.connected_components(G):
                components.append(np.array(sorted(g)))

            # solve assignment for each component
            for component in components:

                from_idx = component[component < num_active_trajectories]
                to_idx = component[component >= num_active_trajectories] - num_active_trajectories

                if to_idx.size == 0:
                    # no new instance to assign to
                    continue

                if from_idx.size == 0:
                    # no active trajectories to assign from, create new trajectories
                    for instance in [instance for idx, instance
                                     in enumerate(instances) if idx in to_idx]:
                        new_trajectories.append(pyTrajectory.trajectory.Trajectory([instance]))
                        cleared_instances.append(instance)
                    continue

                # n-to-n assignment
                from_positions = []
                to_positions = []

                from_features = [[] for feature in similarity_features]
                to_features = [[] for feature in similarity_features]

                for idx, trajectory in enumerate(active_trajectories):
                    if idx not in from_idx:
                        continue
                    for feature_idx, feature in enumerate(similarity_features):
                        from_features[feature_idx].append(feature(trajectory[-1]))

                for idx, instance in enumerate(instances):
                    if idx not in to_idx:
                        continue
                    for feature_idx, feature in enumerate(similarity_features):
                        to_features[feature_idx].append(feature(instance))

                # similarity for each given feature
                dissimilarity_matrices = [np.zeros((len(from_idx), len(to_idx)))
                                          for feature in similarity_features]
                for feature_idx in range(len(similarity_features)):
                    dissimilarity_matrices[feature_idx] = \
                        feature_distances[feature_idx](from_features[feature_idx],
                                                       to_features[feature_idx])
                    # update max dissimilarity
                    if dissimilarity_matrices[feature_idx].max() > max_dissimilarity[feature_idx]:
                        max_dissimilarity[feature_idx] = dissimilarity_matrices[feature_idx].max()

                # weighted cost matrix
                cost_matrix = sum([*[dissimilarity_matrix * weight / max_dissimilarity[feature_idx]
                                     for feature_idx, (dissimilarity_matrix, weight)
                                     in enumerate(zip(dissimilarity_matrices, dissimilarity_weight))]])

                row_idx, col_idx = linear_sum_assignment(cost_matrix)

                assigned_u = []
                assigned_v = []

                for u, v in zip(row_idx, col_idx):
                    # append instance to active trajectory

                    assert (u not in assigned_u) or (v not in assigned_v), 'double assignment! {} {}'.format(row_idx, col_idx)
                    assigned_u.append(u)
                    assigned_v.append(v)

                    active_trajectories[from_idx[u]].append(instances[to_idx[v]])
                    cleared_instances.append(instances[to_idx[v]])

        # clear active instances
        for instance in cleared_instances:
            instances.remove(instance)

        # create new active trajectories from unassigned instances
        for instance in instances:
            new_trajectories.append(pyTrajectory.trajectory.Trajectory([instance]))
        instances = []
        active_trajectories = active_trajectories + new_trajectories

    archived_trajectories = archived_trajectories + active_trajectories

    return archived_trajectories
