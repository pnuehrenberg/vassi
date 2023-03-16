import numpy as np


def format_keypoint_indices(keypoint_indices):
    # format individual keypoint indices
    keypoint_indices = np.asarray(keypoint_indices)
    if keypoint_indices.ndim == 0:
        keypoint_indices = keypoint_indices[np.newaxis]
    assert keypoint_indices.ndim == 1
    return keypoint_indices


def format_segment_keypoint_indices(keypoint_indices):
    # format segment keypoint indices
    # TODO: use in pyTrajectory.trajectory_operations.get_pose_segments
    keypoint_indices = np.asarray(keypoint_indices)
    if keypoint_indices.ndim == 1:
        # pairwise "chained" keypoint segments
        keypoint_indices = np.transpose([keypoint_indices[:-1], keypoint_indices[1:]])
    assert keypoint_indices.ndim == 2
    return keypoint_indices
