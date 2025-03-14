import numpy as np

from .. import math
from ..data_structures import Trajectory
from ..utils import (
    KeypointPairs,
    Keypoints,
    flatten,
    pad_values,
    perform_operation,
)
from . import features


def translation(
    trajectory: Trajectory,
    *,
    step: int,
    pad_value: int | float | str = "same",
    keypoints: Keypoints,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x", "y"),
) -> np.ndarray:
    """2D translation of trajectory keypoints from t - step to t."""
    points = features.keypoints(trajectory, keypoints=keypoints)
    translation = points - math.shift(points, step)
    translation = pad_values(translation, step, pad_value)
    if flat:
        return flatten(translation)
    return translation


def velocity(
    trajectory: Trajectory,
    *,
    step: int,
    pad_value: int | float | str = "same",
    keypoints: Keypoints,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("x", "y"),
) -> np.ndarray:
    """2D velocity of trajectory keypoints (translation / step duration).

    Note that this not calculating the cumulative distance between all instances during the step,
    but the translation between the instance at t - step to the instance at step.
    """
    duration = trajectory.timestep * np.abs(step)
    velocity = translation(trajectory, step=step, keypoints=keypoints) / duration
    velocity = pad_values(velocity, step, pad_value)
    if flat:
        return flatten(velocity)
    return velocity


def speed(
    trajectory: Trajectory,
    *,
    step: int,
    pad_value: int | float | str = "same",
    keypoints: Keypoints,
    flat: bool = False,
) -> np.ndarray:
    """Speed of trajectory keypoints (magnitude of velocity)."""
    speed = math.magnitude(velocity(trajectory, step=step, keypoints=keypoints))
    speed = pad_values(speed, step, pad_value)
    if flat:
        return flatten(speed)
    return speed


def orientation_change(
    trajectory: Trajectory,
    *,
    step: int,
    pad_value: int | float | str = "same",
    keypoint_pairs: KeypointPairs,
    flat: bool = False,
) -> np.ndarray:
    """Signed angles between posture vectors of instance at t - step and instance at t."""
    orientation_vectors = features.posture_vectors(
        trajectory, keypoint_pairs=keypoint_pairs
    )
    orientation_change = math.signed_angle(
        math.shift(orientation_vectors, step),
        orientation_vectors,
    )
    orientation_change = pad_values(orientation_change, step, pad_value)
    if flat:
        return flatten(orientation_change)
    return orientation_change


def angular_speed(
    trajectory: Trajectory,
    *,
    step: int,
    pad_value: int | float | str = "same",
    keypoint_pairs: KeypointPairs,
    flat: bool = False,
) -> np.ndarray:
    """Angular speed of posture segments (orientation change / step duration)."""
    duration = trajectory.timestep * np.abs(step)
    angular_speed = (
        orientation_change(trajectory, step=step, keypoint_pairs=keypoint_pairs)
        / duration
    )
    angular_speed = pad_values(angular_speed, step, pad_value)
    if flat:
        return flatten(angular_speed)
    return angular_speed


def projected_velocity(
    trajectory: Trajectory,
    *,
    step: int,
    pad_value: int | float | str = "same",
    keypoints_1: Keypoints,
    keypoint_pairs_2: KeypointPairs,
    element_wise: bool = False,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("proj", "rej"),
) -> np.ndarray:
    """Keypoint velocity projected onto posture vectors (keypoint_pairs_2) at t - step."""
    velocity_vectors = velocity(trajectory, step=step, keypoints=keypoints_1)
    posture_vectors = features.posture_vectors(
        trajectory,
        keypoint_pairs=keypoint_pairs_2,
    )
    posture_vectors = pad_values(math.shift(posture_vectors, step), step, "same")
    projection = perform_operation(
        math.scalar_projection,
        velocity_vectors,
        posture_vectors,
        element_wise=element_wise,
        flat=flat,
    )
    rejection = perform_operation(
        math.scalar_rejection,
        velocity_vectors,
        posture_vectors,
        element_wise=element_wise,
        flat=flat,
    )
    projected_velocity = np.stack([projection, rejection], axis=-1)
    projected_velocity = pad_values(projected_velocity, step, pad_value)
    if flat:
        return flatten(projected_velocity)
    return projected_velocity


def target_velocity(
    trajectory: Trajectory,
    *,
    step: int,
    pad_value: int | float | str = "same",
    trajectory_other: Trajectory | None = None,
    keypoints_1: Keypoints,
    keypoint_pairs_2: KeypointPairs,
    element_wise: bool = False,
    origin_on_other: bool = False,
    target_on_other: bool = True,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("proj", "rej"),
) -> np.ndarray:
    """Keypoint velocity projected onto target vectors between origin and target keypoints (keypoint_pairs_2).

    Note the default values origin_on_other=False and target_on_other=True.
    By default, the origin keypoint of target vectors is on trajectory, and the target keypoint on trajectory_other.
    """
    if trajectory_other is None:
        trajectory_other = trajectory
    origin_keypoints = tuple([keypoint_pair[0] for keypoint_pair in keypoint_pairs_2])
    target_keypoints = tuple([keypoint_pair[1] for keypoint_pair in keypoint_pairs_2])
    velocity_vectors = velocity(trajectory, step=step, keypoints=keypoints_1)
    target_vectors = features.target_vectors(
        trajectory if not origin_on_other else trajectory_other,
        trajectory_other=trajectory_other if target_on_other else trajectory,
        keypoints_1=origin_keypoints,
        keypoints_2=target_keypoints,
        element_wise=True,
        flat=False,
    )
    projection = perform_operation(
        math.scalar_projection,
        velocity_vectors,
        target_vectors,
        element_wise=element_wise,
        flat=flat,
    )
    rejection = perform_operation(
        math.scalar_rejection,
        velocity_vectors,
        target_vectors,
        element_wise=element_wise,
        flat=flat,
    )
    target_velocity = np.stack([projection, rejection], axis=-1)
    target_velocity = pad_values(target_velocity, step, pad_value)
    if flat:
        return flatten(target_velocity)
    return target_velocity
