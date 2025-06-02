from typing import Optional

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
    """
    Retrieve 2D translation vectors of trajectory keypoints from :code:`t - step` to :code:`t`.

    Parameters:
        trajectory: The trajectory to retrieve the translation vectors from.
        step: The number of steps to shift the trajectory.
        pad_value: The value to pad the translation vectors with.
        keypoints: The keypoint indices to retrieve the translation vectors for.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The translation vectors of the trajectory keypoints.
    """
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
    """
    Calculate the 2D velocity of trajectory keypoints (:code:`translation / step`).

    Note that this not calculating the cumulative distance between all instances during the step,
    but the translation between the instance at :code:`t - step` to the instance at :code:`t`.

    Parameters:
        trajectory: The trajectory to calculate the velocity for.
        step: The number of timesteps to use for the velocity calculation.
        pad_value: The value to use for padding the output.
        keypoints: The keypoint indices to use for the velocity calculation.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The velocity vectors of the trajectory keypoints.
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
    """
    Speed of trajectory keypoints, calculated as magnitude of velocity.

    Parameters:
        trajectory: The trajectory to calculate the speed for.
        step: The step size to use for the velocity calculation.
        pad_value: The value to use for padding the output.
        keypoints: The keypoint indices to use for the velocity calculation.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The speed of the trajectory keypoints.
    """
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
    """
    Compute signed angles between posture vectors of instance at :code:`t - step` and instance at :code:`t`.

    Parameters:
        trajectory: The trajectory to compute the orientation change for.
        step: The number of steps to shift the trajectory by.
        pad_value: The value to pad the output with.
        keypoint_pairs: The keypoint indices to use for the orientation change calculation.
        flat: Whether to flatten the output along all but the first dimension.

    Returns:
        The orientation change between the two instances.
    """
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
    """
    Angular speed of posture segments (:code:`orientation change / step`).

    Parameters:
        trajectory: The trajectory to calculate the angular speed for.
        step: The number of steps to shift the trajectory by.
        pad_value: The value to pad the output with.
        keypoint_pairs: The keypoint indices to use for posture segments and orientation change calculation.
        flat: Whether to flatten the output along all but the first dimension.

    Returns:
        The angular speed of posture segments.
    """
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
    """
    Keypoint velocity projected onto posture vectors at :code:`t - step`.

    Parameters:
        trajectory: The trajectory to calculate the velocity for.
        step: The number of steps to shift the trajectory by.
        pad_value: The value to pad the output with.
        keypoints_1: The keypoints to calculate the velocity for.
        keypoint_pairs_2: The keypoint pairs to specify posture vectors.
        element_wise: Whether to perform the operation element-wise.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The projected velocity and (projection and rejection components).
    """
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
    trajectory_other: Optional[Trajectory] = None,
    keypoints_1: Keypoints,
    keypoint_pairs_2: KeypointPairs,
    element_wise: bool = False,
    origin_on_other: bool = False,
    target_on_other: bool = True,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("proj", "rej"),
) -> np.ndarray:
    """
    Keypoint velocity projected onto target vectors between origin and target keypoints (:code:`keypoint_pairs_2`).

    Origin and target keypoints can be on either :code:`trajectory` or :code:`trajectory_other`.

    Parameters:
        trajectory: The trajectory to calculate the target velocity for.
        step: The number of steps to shift the trajectory by.
        pad_value: The value to pad the trajectory with.
        trajectory_other: The other trajectory for the target keypoints, if :code:`None`, falls back to :code:`trajectory`.
        keypoints_1: The indices of the origin keypoints.
        keypoints_2: The indices of the target vector keypoints.
        element_wise: Whether to calculate the velocity element-wise.
        origin_on_other: Whether the origin keypoints are on the other trajectory.
        target_on_other: Whether the target keypoints are on the other trajectory.
        flat: Whether to flatten the output along all but the first dimension.
        suffixes: The suffixes to use for the output when decorated with :func:`~vassi.features.decorators.as_dataframe`.

    Returns:
        The target velocity for the given trajectory and keypoints.
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
