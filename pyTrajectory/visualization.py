import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections

import pyTrajectory.trajectory_operations


def get_trajectory_key(trajectory, keys=None):
    keys = ['position', 'pose', 'segmentation'] if keys is None else keys
    key = None
    while len(keys) > 0:
        test_key = keys.pop(0)
        if trajectory[test_key] is not None:
            key = test_key
    return key


def get_trajectory_range(trajectory):
    key = get_trajectory_key(trajectory)
    if key is None:
        return (None, None), (None, None)
    axes = tuple(range(len(trajectory[key].shape) - 1))
    x_min, y_min = trajectory[key].min(axis=axes)
    x_max, y_max = trajectory[key].max(axis=axes)
    return (x_min, x_max), (y_min, y_max)


def get_trajectory_alpha_from_density(trajectory):
    (x_min, x_max), (y_min, y_max) = get_trajectory_range(trajectory)
    if None in [x_min, x_max, y_min, y_max]:
        return None
    key = get_trajectory_key(trajectory, ['segmentation', 'pose', 'position'])
    if key is None:
        return None
    bin_width = 20
    if trajectory['pose'] is not None:
        keypoint_indices = np.arange(trajectory['pose'].shape[1])
        pose_segment_lengths = \
            pyTrajectory.trajectory_operations.get_pose_segment_lengths(
                trajectory, keypoint_indices)
        mean_pose_length = pose_segment_lengths.sum(axis=1).mean() / 4
    bins = [np.arange(x_min, x_max + bin_width, bin_width),
            np.arange(y_min, y_max + bin_width, bin_width)]
    if np.any(np.asarray([len(bins_d) for bins_d in bins]) > 100):
        bins = 100
    bin_counts, bin_edges = np.histogramdd(trajectory[key].reshape(-1, 2),
                                           bins=bins)
    positions_digitized = np.transpose([np.digitize(c, bins[1:-1])
            for c, bins in zip(trajectory[key].reshape(-1, 2).T, bin_edges)])
    local_counts = bin_counts[positions_digitized[:, 0], positions_digitized[:, 1]]
    alpha = np.zeros(local_counts.shape) + 0.5
    alpha[local_counts > 5] = 0.25
    alpha[local_counts > 10] = 0.1
    alpha[local_counts > 25] = 0.06
    alpha[local_counts > 50] = 0.03
    alpha[local_counts > 100] = 0.02
    return alpha


def plot_trajectory(trajectory, ax):
    alpha = get_trajectory_alpha_from_density(trajectory)
    pose_color = 0
    if trajectory['segmentation'] is not None:
        coll_segmentation = collections.PolyCollection(trajectory['segmentation'],
                                                       edgecolor=[(0, 0, 0, a) for a in alpha],
                                                       facecolor=[(0, 0, 0, a / 4) for a in alpha],
                                                       lw=0.25)
        ax.add_collection(coll_segmentation)
        pose_color = 1
    if trajectory['pose'] is not None:
        coll_pose = collections.PolyCollection(trajectory['pose'],
                                               closed=False,
                                               edgecolor=[(pose_color, 0, 0, a) for a in alpha],
                                               facecolor=(0, 0, 0, 0),
                                               lw=0.5,
                                               capstyle='round')
        ax.add_collection(coll_pose)
