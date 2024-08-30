from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .. import math
from ..data_structures import InstanceCollection, Trajectory
from ..utils import (
    KeypointPairs,
    Keypoints,
    flatten,
    perform_operation,
)


def keypoints(
    trajectory: InstanceCollection,
    *,
    keypoints: Keypoints,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x", "y"),
) -> NDArray:
    """2D coordinates of trajectory keypoints."""
    if trajectory.cfg.key_keypoints is None:
        raise ValueError("key_keypoints is not defined.")
    points = trajectory[trajectory.cfg.key_keypoints][:, tuple(keypoints), :2]
    if flat:
        return flatten(points)
    return points


def posture_segments(
    trajectory: InstanceCollection,
    *,
    keypoint_pairs: KeypointPairs,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x1", "y1", "x2", "y2"),
) -> NDArray:
    """2D line segments between trajectory keypoints."""
    if trajectory.cfg.key_keypoints is None:
        raise ValueError("key_keypoints is not defined.")
    segments = trajectory[trajectory.cfg.key_keypoints][:, tuple(keypoint_pairs), :2]
    if flat:
        return flatten(segments)
    return segments


def posture_vectors(
    trajectory: InstanceCollection,
    *,
    keypoint_pairs: KeypointPairs,
    as_unit_vectors: bool = False,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x", "y"),
) -> NDArray:
    """2D vectors between trajectory keypoints.

    Alternatively, return vectors as unit vectors with as_unit_vectors=True.
    """
    segments = posture_segments(trajectory, keypoint_pairs=keypoint_pairs)
    vectors = segments[..., 1, :] - segments[..., 0, :]
    if as_unit_vectors:
        vectors = math.unit_vector(vectors)
    if flat:
        return flatten(vectors)
    return vectors


def posture_angles(
    trajectory: Trajectory,
    *,
    trajectory_other: Optional[Trajectory] = None,
    keypoint_pairs_1: KeypointPairs,
    keypoint_pairs_2: KeypointPairs,
    element_wise: bool = False,
    flat: bool = False,
) -> NDArray:
    """Signed angles in radians between posture vectors."""
    if trajectory_other is None:
        trajectory_other = trajectory
    return perform_operation(
        math.signed_angle,
        posture_vectors(trajectory, keypoint_pairs=keypoint_pairs_1),
        posture_vectors(trajectory_other, keypoint_pairs=keypoint_pairs_2),
        element_wise,
        flat,
    )


def posture_alignment(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    keypoint_pairs_1: KeypointPairs,
    keypoint_pairs_2: KeypointPairs,
    element_wise: bool = False,
    flat: bool = False,
) -> NDArray:
    """Alignment of posture vectors, ranging beteen 0 (anti-parallel), 0.5 (orthogonal) and 1 (parallel)."""
    if trajectory_other is None:
        trajectory_other = trajectory
    unsigned_angles = perform_operation(
        math.unsigned_angle,
        posture_vectors(trajectory, keypoint_pairs=keypoint_pairs_1),
        posture_vectors(trajectory_other, keypoint_pairs=keypoint_pairs_2),
        element_wise,
        flat,
    )
    return 1 - unsigned_angles / np.pi


def keypoint_distances(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    keypoints_1: Keypoints,
    keypoints_2: Keypoints,
    element_wise: bool = False,
    flat: bool = False,
) -> NDArray:
    """Euclidean distances between keypoints."""
    if trajectory_other is None:
        trajectory_other = trajectory
    return perform_operation(
        math.euclidean_distance,
        keypoints(trajectory, keypoints=keypoints_1),
        keypoints(trajectory_other, keypoints=keypoints_2),
        element_wise,
        flat,
    )


def target_vectors(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    keypoints_1: Keypoints,
    keypoints_2: Keypoints,
    element_wise: bool = False,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x", "y"),
) -> NDArray:
    """Vectors from keypoints_1 to keypoints_2."""
    if trajectory_other is None:
        trajectory_other = trajectory
    return perform_operation(
        math.subtract,
        keypoints(trajectory, keypoints=keypoints_1),
        keypoints(trajectory_other, keypoints=keypoints_2),
        element_wise,
        flat,
    )


def target_angles(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    keypoint_pairs_1: KeypointPairs,
    keypoints_2: Keypoints,
    element_wise: bool = False,
    flat: bool = False,
) -> NDArray:
    """Signed angles in radians between posture vectors and target vectors (from posture segment end keypoints to keypoints_2)."""
    origin_keypoints = tuple([keypoint_pair[1] for keypoint_pair in keypoint_pairs_1])
    orientation_vectors = posture_vectors(trajectory, keypoint_pairs=keypoint_pairs_1)
    if not element_wise:
        orientation_vectors = np.expand_dims(orientation_vectors, axis=2)
    return perform_operation(
        math.signed_angle,
        orientation_vectors,
        target_vectors(
            trajectory,
            trajectory_other=trajectory_other,
            keypoints_1=origin_keypoints,
            keypoints_2=keypoints_2,
            element_wise=element_wise,
            flat=False,
        ),
        element_wise=element_wise,
        flat=flat,
        expand_dims_for_broadcasting=False,
    )
