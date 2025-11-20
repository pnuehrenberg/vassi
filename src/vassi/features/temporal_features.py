from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np

from .. import batched_math, fast_math
from ..data_structures import Trajectory
from ..utils import pad_values
from .features import keypoints as get_keypoints
from .features import posture_vectors as get_posture_vectors
from .features import target_vectors as get_target_vectors


def load_by_name(func_name: str) -> Callable[..., np.ndarray] | None:
    return globals().get(func_name, None)


def translation(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    step: int,
    pad_value: float | Literal["same"] = "same",
    keypoints: Iterable[int],
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
    del trajectory_other  # only for compatibility with other features
    if len(suffixes) != 2:
        raise ValueError("Expected exactly two suffixes for the coordinate dimensions.")
    points = get_keypoints(trajectory, keypoints=keypoints)
    translation = points - fast_math.shift(points, step)
    translation = pad_values(translation, step, pad_value)
    if flat:
        translation = translation.reshape(translation.shape[0], -1)
    return translation


def velocity(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    step: int,
    pad_value: float | Literal["same"] = "same",
    keypoints: Iterable[int],
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
    del trajectory_other  # only for compatibility with other features
    if len(suffixes) != 2:
        raise ValueError("Expected exactly two suffixes for the coordinate dimensions.")
    duration = trajectory.timestep * np.abs(step)
    velocity = (
        translation(trajectory, step=step, pad_value=pad_value, keypoints=keypoints)
        / duration
    )
    # velocity = pad_values(velocity, step, pad_value)
    if flat:
        velocity = velocity.reshape(velocity.shape[0], -1)
    return velocity


def speed(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    step: int,
    pad_value: float | Literal["same"] = "same",
    keypoints: Iterable[int],
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
    del trajectory_other  # only for compatibility with other features
    speed = fast_math.magnitude(
        velocity(trajectory, step=step, pad_value=pad_value, keypoints=keypoints)
    )
    # speed = pad_values(speed, step, pad_value)
    if flat:
        speed = speed.reshape(speed.shape[0], -1)
    return speed


def orientation_change(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    step: int,
    pad_value: float | Literal["same"] = "same",
    keypoint_pairs: Iterable[tuple[int, int]],
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
    del trajectory_other  # only for compatibility with other features
    orientation_vectors = get_posture_vectors(trajectory, keypoint_pairs=keypoint_pairs)
    orientation_change = batched_math.element_wise_signed_angle(
        fast_math.shift(orientation_vectors, step),
        orientation_vectors,
    )
    orientation_change = pad_values(orientation_change, step, pad_value)
    if flat:
        orientation_change = orientation_change.reshape(orientation_change.shape[0], -1)
    return orientation_change


def angular_speed(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    step: int,
    pad_value: float | Literal["same"] = "same",
    keypoint_pairs: Iterable[tuple[int, int]],
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
    del trajectory_other  # only for compatibility with other features
    duration = trajectory.timestep * np.abs(step)
    angular_speed = (
        orientation_change(
            trajectory, step=step, pad_value=pad_value, keypoint_pairs=keypoint_pairs
        )
        / duration
    )
    if flat:
        angular_speed = angular_speed.reshape(angular_speed.shape[0], -1)
    return angular_speed


def projected_velocity(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    step: int,
    pad_value: float | Literal["same"] = "same",
    keypoints_1: Iterable[int],
    keypoint_pairs_2: Iterable[tuple[int, int]],
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
    del trajectory_other  # only for compatibility with other features
    if len(suffixes) != 2:
        raise ValueError(
            "Expected exactly two suffixes for the projection and rejection components."
        )
    velocity_vectors = velocity(
        trajectory, step=step, pad_value=pad_value, keypoints=keypoints_1
    )
    posture_vectors = get_posture_vectors(
        trajectory,
        keypoint_pairs=keypoint_pairs_2,
    )
    posture_vectors = pad_values(
        fast_math.shift(posture_vectors, step), step, pad_value
    )
    return batched_math.projected_components(
        velocity_vectors,
        posture_vectors,
        element_wise=element_wise,
        flat=flat,
    )


def target_velocity(
    trajectory: Trajectory,
    *,
    trajectory_other: Trajectory | None = None,
    step: int,
    pad_value: float | Literal["same"] = "same",
    keypoints_1: Iterable[int],
    keypoint_pairs_2: Iterable[tuple[int, int]],
    element_wise: bool = False,
    origin_on_other: bool = False,
    target_on_other: bool = True,
    flat: bool = False,
    suffixes: tuple[str, ...] = ("proj", "rej"),
) -> np.ndarray:
    """
    Keypoint velocity projected onto target vectors between origin and target keypoints (:code:`keypoint_pairs_2`).

    The keypoint velocity is always calculated for the keypoints of :code:`trajectory`, regardless of the :code:`origin_on_other` parameter.
    The keypoints that define the target vectors (from origin to target keypoints) can be specified to be on either :code:`trajectory` or :code:`trajectory_other`.

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
    if len(suffixes) != 2:
        raise ValueError(
            "Expected exactly two suffixes for the projection and rejection components."
        )
    if trajectory_other is None:
        trajectory_other = trajectory
    origin_keypoints = tuple([keypoint_pair[0] for keypoint_pair in keypoint_pairs_2])
    target_keypoints = tuple([keypoint_pair[1] for keypoint_pair in keypoint_pairs_2])
    velocity_vectors = velocity(
        trajectory, step=step, pad_value=pad_value, keypoints=keypoints_1
    )
    target_vectors = get_target_vectors(
        trajectory if not origin_on_other else trajectory_other,
        trajectory_other=trajectory_other if target_on_other else trajectory,
        keypoints_1=origin_keypoints,
        keypoints_2=target_keypoints,
        element_wise=True,
        flat=False,
    )
    result = batched_math.projected_components(
        velocity_vectors,
        target_vectors,
        element_wise=element_wise,
        flat=flat,
    )
    return pad_values(result, step, pad_value)
