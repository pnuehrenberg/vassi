import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib.colors import to_rgb
from copy import deepcopy

from .trajectory_operations import get_pose_segment_lengths
import pyTrajectory.config as config


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


def plot_trajectory(trajectory, ax, visualization_config=None, **visualization_kwargs):

    def get_value(arg):
        nonlocal visualization_config
        if arg in visualization_config:
            return visualization_config[arg]
        return tuple([*to_rgb(visualization_config[f'{arg}_color']),
                              visualization_config[f'{arg}_alpha']])

    visualization_config = visualization_config or deepcopy(config.VISUALIZATION_CONFIG)
    for arg, value in visualization_kwargs.items():
        visualization_config[arg] = value

    if get_value('plot_segmentation') and trajectory['segmentation'] is not None:
        coll_segmentation = collections.PolyCollection(trajectory['segmentation'],
                                                       edgecolor=get_value('segmentation_edge'),
                                                       facecolor=get_value('segmentation_face'),
                                                       lw=get_value('segmentation_edge_width'))
        ax.add_collection(coll_segmentation)

    if get_value('plot_pose') and trajectory['pose'] is not None:
        coll_pose = collections.PolyCollection(trajectory['pose'],
                                               closed=False,
                                               edgecolor=get_value('pose_line'),
                                               facecolor=(0, 0, 0, 0),
                                               lw=get_value('pose_line_width'),
                                               capstyle='round')
        ax.add_collection(coll_pose)

    if get_value('plot_position') and trajectory['position'] is not None:
        ax.scatter(*trajectory['position'].T,
                   marker=get_value('position_marker'),
                   s=get_value('position_size'),
                   facecolor=get_value('position_face'),
                   edgecolor=get_value('position_edge'),
                   linewidth=get_value('position_line_width'))
