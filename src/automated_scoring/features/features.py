from typing import Optional

import numpy as np

from .. import math
from ..data_structures import InstanceCollection, Trajectory
from ..utils import (
    KeypointPairs,
    Keypoints,
    flatten,
    perform_operation,
)


def keypoints(
    collection: InstanceCollection,
    *,
    keypoints: Keypoints,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x", "y"),
) -> np.ndarray:
    """
    2D coordinates of trajectory keypoints.

    Parameters:
        collection: The instance collection or trajectory to retrieve the keypoints from.
        keypoints: The indices of keypoints to retrieve.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~automated_scoring.features.decorators.as_dataframe`.

    Returns:
        The 2D coordinates of the keypoints.

    Note:
        This is currently only extracting the x and y coordinates of the keypoints.
    """
    if collection.cfg.key_keypoints is None:
        raise ValueError("key_keypoints is not defined.")
    points = collection[collection.cfg.key_keypoints][:, tuple(keypoints), :2]
    if flat:
        return flatten(points)
    return points


def position(
    collection: InstanceCollection,
    *,
    suffixes: tuple[str, ...] = ("x", "y"),
) -> np.ndarray:
    """
    Calculates the position of each instance in the collection as the mean of all keypoints.

    Parameters:
        collection: The instance collection.
        suffixes: The suffixes to use for the output when decorated with :func:`~automated_scoring.features.decorators.as_dataframe`.

    Returns:
        The 2D position of each instance in the collection.
    """
    if collection.cfg.key_keypoints is None:
        raise ValueError("key_keypoints is not defined.")
    num_keypoints = collection[collection.cfg.key_keypoints].shape[1]
    return keypoints(collection, keypoints=tuple(range(num_keypoints))).mean(axis=1)


def posture_segments(
    trajectory: InstanceCollection,
    *,
    keypoint_pairs: KeypointPairs,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x1", "y1", "x2", "y2"),
) -> np.ndarray:
    """
    Retrieves 2D line segments between trajectory keypoints.

    Parameters:
        trajectory: The trajectory or instance collection.
        keypoint_pairs: The pairs of keypoint indices to use for the segments.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~automated_scoring.features.decorators.as_dataframe`.

    Returns:
        The 2D line segments between trajectory keypoints.
    """
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
) -> np.ndarray:
    """2D vectors between trajectory keypoints.

    Alternatively, return vectors as unit vectors with as_unit_vectors=True.

    Parameters:
        trajectory: The trajectory or instance collection.
        keypoint_pairs: The pairs of keypoint indices to use for the segments.
        as_unit_vectors: Whether to return unit vectors instead of regular vectors.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~automated_scoring.features.decorators.as_dataframe`.

    Returns:
        The 2D vectors between the keypoints.
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
) -> np.ndarray:
    """
    Calculate signed angles in radians between posture vectors.

    Parameters:
        trajectory: The trajectory to calculate the angles for.
        trajectory_other: The other trajectory to calculate the angles with, if :code:`None`, falls back to :code:`trajectory`.
        keypoint_pairs_1: The keypoint pairs to use for the first angle ray (of the first trajectory).
        keypoint_pairs_2: The keypoint pairs to use for the second angle ray (of the second trajectory).
        element_wise: Whether to calculate the angles element-wise.
        flat: Whether to flatten the output along all but the first dimension.

    Returns:
        The signed angles in radians between the posture vectors.
    """
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
) -> np.ndarray:
    """
    Calculate alignment of posture vectors, ranging between 0 (anti-parallel), 0.5 (orthogonal) and 1 (parallel).

    Parameters:
        trajectory: The trajectory to calculate the alignment for.
        trajectory_other: The other trajectory to calculate the alignment with, if :code:`None`, falls back to :code:`trajectory`.
        keypoint_pairs_1: The keypoint pairs to use for the first angle ray (of the first trajectory).
        keypoint_pairs_2: The keypoint pairs to use for the second angle ray (of the second trajectory).
        element_wise: Whether to calculate the alignment element-wise.
        flat: Whether to flatten the output along all but the first dimension.

    Returns:
        The alignment of the posture vectors.
    """
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
) -> np.ndarray:
    """
    Calculate euclidean distances between keypoints.

    Parameters:
        trajectory: The trajectory to calculate the distances for.
        trajectory_other: The other trajectory to calculate the distances with, if :code:`None`, falls back to :code:`trajectory`.
        keypoints_1: The keypoint indices to use (of the first trajectory).
        keypoints_2: The keypoint indices to use as target (of the second trajectory).
        element_wise: Whether to calculate the distances element-wise.
        flat: Whether to flatten the output along all but the first dimension.

    Returns:
        The euclidean distances between the keypoints.
    """
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
) -> np.ndarray:
    """
    Retrieve vectors pointing from one set of keypoints to another.

    Parameters:
        trajectory: The trajectory to get target vectors from.
        trajectory_other: The other trajectory for the target vectors, if :code:`None`, falls back to :code:`trajectory`.
        keypoints_1: The keypoint indices to use (of the first trajectory).
        keypoints_2: The keypoint indices to use as target (of the second trajectory).
        element_wise: Whether to calculate the distances element-wise.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~automated_scoring.features.decorators.as_dataframe`.

    Returns:
        The target vectors.
    """
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
) -> np.ndarray:
    """
    Calculate signed angles in radians between posture vectors and target vectors.

    The two legs of the angles are
    - the orientation vectors of the posture segments
    - the vectors pointing from posture segment (:code:`keypoint_pairs_1`) end keypoints to the target keypoints (:code:`keypoints_2`)

    Parameters:
        trajectory: The trajectory to compute target vectors for.
        trajectory_other: The other trajectory for the target keypoints, if :code:`None`, falls back to :code:`trajectory`.
        keypoint_pairs_1: The keypoint segment indices to use (of the first trajectory).
        keypoints_2: The keypoint indices to use as target (of the second trajectory).
        element_wise: Whether to calculate the distances element-wise.
        flat: Whether to flatten the output along all but the first dimension.
    """
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
