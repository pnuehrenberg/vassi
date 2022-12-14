import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib import collections, patches, lines

import pyTrajectory.config


def get_instance_range(instance):
    cfg = pyTrajectory.config.cfg
    padding = cfg.figure.padding
    try:
        box = instance[cfg.key_box]
        return (box[0] - padding, box[2] + padding), \
               (box[1] - padding, box[3] + padding)
    except KeyError:
        pass
    try:
        keypoints = instance[cfg.key_keypoints][:, :2]
        x_min, y_min = keypoints.min(axis=0)
        x_max, y_max = keypoints.max(axis=0)
        return (x_min - padding, x_max + padding), \
               (y_min - padding, y_max + padding)
    except KeyError:
        raise NotImplementedError


def get_trajectory_range(trajectory):
    cfg = pyTrajectory.config.cfg
    padding = cfg.figure.padding
    try:
        boxes = trajectory[cfg.key_box]
        x_min, y_min = boxes[:, :2].min(axis=0)
        x_max, y_max = boxes[:, 2:].max(axis=0)
        return (x_min - padding, x_max + padding), \
               (y_min - padding, y_max + padding)
    except KeyError:
        pass
    try:
        keypoints = trajectory[cfg.key_keypoints][..., :2]
        x_min, y_min = keypoints.min(axis=(0, 1))
        x_max, y_max = keypoints.max(axis=(0, 1))
        return (x_min - padding, x_max + padding), \
               (y_min - padding, y_max + padding)
    except KeyError:
        raise NotImplementedError


def prepare_box(box_xyxy, **kwargs):
    xy = box_xyxy[:2]
    width, height = box_xyxy[2:] - xy
    return patches.Rectangle(xy, width, height, **kwargs)


def prepare_line_segments(data, **kwargs):
    data = np.asarray(list(zip(data[:-1], data[1:])))
    return collections.LineCollection(data, **kwargs)


def prepare_line(data, **kwargs):
    if as_segments:
        data = np.asarray(list(zip(data[:-1], data[1:])))
    return lines.Line2D(data[:, 0], data[:, 1], **kwargs)


def prepare_boxes(boxes_xyxy):
    xy = boxes_xyxy[:, :2]
    width, height = (boxes_xyxy[:, 2:] - xy).T
    return [patches.Rectangle(xy, width, height) for xy, width, height in zip(xy, width, height)]


def add_collection(ax, collection, trajectory, key, prepare_data_func=None, **kwargs):
    if prepare_data_func is None:
        data = trajectory[key][..., :2]
    else:
        data = prepare_data_func(trajectory[key])
    ax.add_collection(collection(data, **kwargs))
