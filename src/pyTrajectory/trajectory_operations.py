import numpy as np

from .series_math import calculate_unit_vectors, \
                         calculate_element_wise_magnitude, \
                         calculate_pairwise_signed_angle_between, \
                         calculate_pairwise_unsigned_angle_between, \
                         element_wise_broadcast
from .series_decorators import smooth_output, filter_output, norm_output
from .series_operations import get_sliding_cumulative_distance, \
                               get_sliding_distance


import pyTrajectory.config
from .utils import format_segment_keypoint_indices, format_keypoint_indices


def get_posture_segments(trajectory, keypoint_indices):
    # segments between keypoints
    # keypoint indices formatted as segment_keypoint_indices
    cfg = pyTrajectory.config.cfg
    keypoint_indices = format_segment_keypoint_indices(keypoint_indices)
    posture_segments = np.diff(trajectory[cfg.key_keypoints][:, keypoint_indices], axis=-2)
    posture_segments = posture_segments.squeeze(axis=-2)
    return posture_segments


@norm_output
def get_orientation_vectors(trajectory, keypoint_indices):
    # normalized vectors between keypoints
    # keypoint indices formatted as segment_keypoint_indices
    keypoint_indices = format_segment_keypoint_indices(keypoint_indices)
    return get_posture_segments(trajectory, keypoint_indices)


def get_posture_angles(trajectory, keypoint_indices):
    segment_keypoint_indices = format_segment_keypoint_indices(keypoint_indices)
    posture_segments = get_posture_segments(trajectory, keypoint_indices)
    posture_angles = calculate_pairwise_signed_angle_between(posture_segments[:, :-1], posture_segments[:, 1:])
    return posture_angles


def get_posture_alignment(trajectory_subject, trajectory_modifier, alignment_keypoint_indices):
    # pairwise alignment (between 0 and 1) of orientation vectors
    assert len(trajectory_subject) == len(trajectory_modifier)
    orientation_vectors_subject = get_posture_segments(trajectory_subject, alignment_keypoint_indices)
    orientation_vectors_modifier = get_posture_segments(trajectory_modifier, alignment_keypoint_indices)
    alignment_angles = calculate_pairwise_unsigned_angle_between(orientation_vectors_subject,
                                                                 orientation_vectors_modifier)
    alignment = 1 - alignment_angles / np.pi
    return alignment


def get_posture_distance(trajectory_subject,
                      trajectory_modifier,
                      subject_keypoint_indices,
                      modifier_keypoint_indices,
                      pairwise=False):
    assert len(trajectory_subject) == len(trajectory_modifier)
    cfg = pyTrajectory.config.cfg
    subject_keypoint_indices = format_keypoint_indices(subject_keypoint_indices)
    modifier_keypoint_indices = format_keypoint_indices(modifier_keypoint_indices)
    keypoints_subject = trajectory_subject[cfg.key_keypoints][:, subject_keypoint_indices]
    keypoints_modifier = trajectory_modifier[cfg.key_keypoints][:, modifier_keypoint_indices]
    if subject_keypoint_indices.size == 1 \
       and np.array_equal(subject_keypoint_indices, modifier_keypoint_indices):
        pairwise = True
    if keypoints_subject.shape != keypoints_modifier.shape:
        pairwise = False
    if not pairwise:
        keypoints_subject = keypoints_subject[:, :, np.newaxis]
        keypoints_modifier = keypoints_modifier[:, np.newaxis]
    posture_distance = calculate_element_wise_magnitude(keypoints_subject - keypoints_modifier)
    return posture_distance.reshape(len(trajectory_subject), -1)


def get_target_angle(trajectory_subject,
                     trajectory_modifier,
                     orientation_keypoint_indices,
                     target_keypoint_indices):
    assert len(trajectory_subject) == len(trajectory_modifier)
    cfg = pyTrajectory.config.cfg
    orientation_keypoint_indices = format_segment_keypoint_indices(orientation_keypoint_indices)
    target_keypoint_indices = format_keypoint_indices(target_keypoint_indices)
    orientation_vectors = get_orientation_vectors(trajectory_subject, orientation_keypoint_indices)
    target_vectors = element_wise_broadcast(trajectory_modifier[cfg.key_keypoints][:, target_keypoint_indices],
                                            trajectory_subject[cfg.key_keypoints][:, orientation_keypoint_indices[:, 1]],
                                            np.subtract)
    target_angles = calculate_pairwise_unsigned_angle_between(orientation_vectors[:, np.newaxis], target_vectors)
    return target_angles.reshape(len(trajectory_subject), -1)
