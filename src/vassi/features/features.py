from collections.abc import Callable, Iterable

import numpy as np

from .. import batched_math, fast_math
from ..data_structures import InstanceCollection, Trajectory


def load_by_name(func_name: str) -> Callable[..., np.ndarray] | None:
    return globals().get(func_name, None)


def keypoints(
    trajectory: InstanceCollection,
    *,
    trajectory_other: InstanceCollection | None = None,
    keypoints: Iterable[int],
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x", "y"),
) -> np.ndarray:
    """
    2D coordinates of trajectory keypoints.

    Parameters:
        collection: The instance collection (e.g., a trajectory) to retrieve the keypoints from.
        keypoints: The indices of keypoints to retrieve.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The 2D coordinates of the keypoints.

    Note:
        This is currently only extracting the x and y coordinates of the keypoints.
    """
    del trajectory_other  # only for compatibility with other features
    if len(suffixes) != 2:
        raise ValueError("Expected exactly two suffixes for the coordinate dimensions.")
    if trajectory.cfg.key_keypoints is None:
        raise ValueError("key_keypoints is not defined.")
    points = trajectory[trajectory.cfg.key_keypoints][:, tuple(keypoints), :2]
    if flat:
        points = points.reshape(points.shape[0], -1)
    return points


def position(
    trajectory: InstanceCollection,
    *,
    trajectory_other: InstanceCollection | None = None,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x", "y"),
) -> np.ndarray:
    """
    Calculates the position of each instance in the collection as the mean of all keypoints.

    Parameters:
        collection: The instance collection (e.g., a trajectory).
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The 2D position of each instance in the collection.
    """
    del trajectory_other  # only for compatibility with other features
    if len(suffixes) != 2:
        raise ValueError("Expected exactly two suffixes for the coordinate dimensions.")
    if trajectory.cfg.key_keypoints is None:
        raise ValueError("key_keypoints is not defined.")
    num_keypoints = trajectory[trajectory.cfg.key_keypoints].shape[1]
    positions = keypoints(trajectory, keypoints=tuple(range(num_keypoints))).mean(
        axis=1
    )
    if flat:
        positions = positions.reshape(positions.shape[0], -1)
    return positions


def posture_segments(
    trajectory: InstanceCollection,
    *,
    trajectory_other: InstanceCollection | None = None,
    keypoint_pairs: Iterable[tuple[int, int]],
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x1", "y1", "x2", "y2"),
) -> np.ndarray:
    """
    Retrieves 2D line segments between trajectory keypoints.

    Parameters:
        trajectory: The trajectory or instance collection.
        keypoint_pairs: The pairs of keypoint indices to use for the segments.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The 2D line segments between trajectory keypoints.
    """
    del trajectory_other  # only for compatibility with other features
    if len(suffixes) != 4:
        raise ValueError(
            "Expected exactly 4 suffixes (two per segment point for the coordinate dimensions)."
        )
    if trajectory.cfg.key_keypoints is None:
        raise ValueError("key_keypoints is not defined.")
    segments = trajectory[trajectory.cfg.key_keypoints][:, tuple(keypoint_pairs), :2]
    if flat:
        segments = segments.reshape(segments.shape[0], -1)
    return segments


def posture_vectors(
    trajectory: InstanceCollection,
    *,
    trajectory_other: InstanceCollection | None = None,
    keypoint_pairs: Iterable[tuple[int, int]],
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
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The 2D vectors between the keypoints.
    """
    del trajectory_other  # only for compatibility with other features
    if len(suffixes) != 2:
        raise ValueError("Expected exactly two suffixes for the coordinate dimensions.")
    segments = posture_segments(trajectory, keypoint_pairs=keypoint_pairs)
    vectors = segments[..., 1, :] - segments[..., 0, :]
    if as_unit_vectors:
        vectors = fast_math.unit_vector(vectors)
    if flat:
        vectors = vectors.reshape(vectors.shape[0], -1)
    return vectors


def posture_angles(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    keypoint_pairs_1: Iterable[tuple[int, int]],
    keypoint_pairs_2: Iterable[tuple[int, int]],
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
    return batched_math.signed_angle(
        posture_vectors(trajectory, keypoint_pairs=keypoint_pairs_1),
        posture_vectors(trajectory_other, keypoint_pairs=keypoint_pairs_2),
        element_wise=element_wise,
        flat=flat,
    )


def posture_alignment(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    keypoint_pairs_1: Iterable[tuple[int, int]],
    keypoint_pairs_2: Iterable[tuple[int, int]],
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
    return batched_math.alignment(
        posture_vectors(trajectory, keypoint_pairs=keypoint_pairs_1),
        posture_vectors(trajectory_other, keypoint_pairs=keypoint_pairs_2),
        element_wise=element_wise,
        flat=flat,
    )


def keypoint_distances(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    keypoints_1: Iterable[int],
    keypoints_2: Iterable[int],
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
    return batched_math.euclidean_distance(
        keypoints(trajectory, keypoints=keypoints_1),
        keypoints(trajectory_other, keypoints=keypoints_2),
        element_wise=element_wise,
        flat=flat,
    )


def target_vectors(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    keypoints_1: Iterable[int],
    keypoints_2: Iterable[int],
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
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The target vectors.
    """
    if len(suffixes) != 2:
        raise ValueError("Expected exactly two suffixes for the coordinate dimensions.")
    if trajectory_other is None:
        trajectory_other = trajectory
    return batched_math.subtract(
        keypoints(trajectory, keypoints=keypoints_1),
        keypoints(trajectory_other, keypoints=keypoints_2),
        element_wise=element_wise,
        flat=flat,
    )


def target_angles(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    keypoint_pairs_1: Iterable[tuple[int, int]],
    keypoints_2: Iterable[int],
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
        element_wise: Whether to calculate the target angles element-wise.
        flat: Whether to flatten the output along all but the first dimension.
    """
    origin_keypoints = tuple([keypoint_pair[1] for keypoint_pair in keypoint_pairs_1])
    orientation_vectors = posture_vectors(
        trajectory, keypoint_pairs=keypoint_pairs_1
    )  # T x N x 2
    # we expect T x N for element_wise, and T x N x M for broadcast
    # we do not want to broadcast all orientation vectors to all target vectors
    # but only the orientation vectors to THEIR corresponding target vectors
    if element_wise:
        return batched_math.signed_angle(
            orientation_vectors,
            target_vectors(
                trajectory,
                trajectory_other=trajectory_other,
                keypoints_1=origin_keypoints,
                keypoints_2=keypoints_2,
                element_wise=True,
            ),
            element_wise=True,
            flat=flat,
        )
    result = np.zeros(
        (*orientation_vectors.shape[:2], len(tuple(keypoints_2))), dtype=float
    )
    for idx in range(len(origin_keypoints)):
        # as explained above, only the orientation vectors to THEIR corresponding target vectors
        # which requires some indexing and reshaping
        _target_vectors = target_vectors(
            trajectory,
            trajectory_other=trajectory_other,
            keypoints_1=(origin_keypoints[idx],),
            keypoints_2=keypoints_2,
            element_wise=element_wise,
        )  # T x 1 x M x 2, reshape below to T x M x 2
        _result = batched_math.broadcasted_signed_angle(
            orientation_vectors[:, idx].reshape(orientation_vectors.shape[0], 1, -1),
            _target_vectors.reshape(
                _target_vectors.shape[0], -1, _target_vectors.shape[-1]
            ),
        )
        result[:, idx] = _result[:, 0]
    if flat:
        result = result.reshape(result.shape[0], -1)
    return result
