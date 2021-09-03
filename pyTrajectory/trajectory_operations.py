import numpy as np

import pyTrajectory.series_math
import pyTrajectory.series_decorators


def get_pose_segments(trajectory, keypoint_indices):
    keypoint_indices = np.asarray(keypoint_indices)
    if len(keypoint_indices.shape) == 1:
        keypoint_indices = keypoint_indices[np.newaxis]
    pose_segments = np.diff(trajectory['pose'][:, keypoint_indices], axis=-2)
    pose_segments = pose_segments.reshape(len(trajectory['pose']), -1, trajectory['pose'].shape[-1])
    return pose_segments


@pyTrajectory.series_decorators.smooth_output(force_positive=True)
@pyTrajectory.series_decorators.filter_output
def get_pose_speed(trajectory, keypoint_indices):
    assert len(trajectory['pose']), 'miminum trajectory length of two instances required'
    keypoint_indices = np.asarray(keypoint_indices)
    if len(keypoint_indices.shape) == 0:
        keypoint_indices = keypoint_indices.reshape(1)
    d_time = np.diff(trajectory['time_stamp'])
    d_poses = np.diff(trajectory['pose'][:, keypoint_indices, :], axis=0)
    speed = np.sqrt(np.square(d_poses).sum(axis=-1)) / d_time[:, np.newaxis]
    speed = np.insert(speed, 0, speed[0], axis=0)
    return speed


@pyTrajectory.series_decorators.norm_output
@pyTrajectory.series_decorators.smooth_output
@pyTrajectory.series_decorators.filter_output
def get_orientation_vectors(trajectory, keypoint_indices):
    pose_segments = get_pose_segments(trajectory, keypoint_indices)
    return pyTrajectory.series_math.calculate_unit_vectors(pose_segments)


@pyTrajectory.series_decorators.smooth_output(median_filter_window_size=11, savgol_filter_window_size=11)
@pyTrajectory.series_decorators.filter_output(window_size=11)
def get_pose_segment_lengths(trajectory, keypoint_indices):
    return pyTrajectory.series_math.calculate_element_wise_magnitude(
        get_pose_segments(trajectory, keypoint_indices))


def get_noise(trajectory, window_size=11):
    noise = pyTrajectory.series_math.get_sliding_cumulative_distance(trajectory['pose'], window_size) \
            / pyTrajectory.series_math.get_sliding_distance(trajectory['pose'], window_size)
    return noise
