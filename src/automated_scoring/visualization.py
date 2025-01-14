from .data_structures import Instance, Trajectory


def get_instance_range(
    instance: Instance, padding: float = 0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute the range of the instance.

    Parameters
    ----------
    instance : Instance
        The instance.
    padding : int, optional
        The padding, by default 0.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        The range of the instance.

    Raises
    ------
    NotImplementedError
        If the instance does not have a box or keypoints.
    """
    cfg = instance.cfg
    if cfg.key_box is not None:
        box = instance[cfg.key_box]
        return (
            (box[0] - padding, box[2] + padding),
            (box[1] - padding, box[3] + padding),
        )
    if cfg.key_keypoints is not None:
        keypoints = instance[cfg.key_keypoints][:, :2]
        x_min, y_min = keypoints.min(axis=0)
        x_max, y_max = keypoints.max(axis=0)
        return (
            (x_min - padding, x_max + padding),
            (y_min - padding, y_max + padding),
        )
    raise NotImplementedError(
        "instance range not implemented for instances without boxes or keypoints"
    )


def get_trajectory_range(
    trajectory: Trajectory, padding: float = 0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute the range of the trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        The trajectory.
    padding : int, optional
        The padding, by default 0.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        The range of the trajectory.

    Raises
    ------
    NotImplementedError
        If the trajectory does not have a box or keypoints.
    """
    cfg = trajectory.cfg
    if cfg.key_box is not None:
        boxes = trajectory[cfg.key_box]
        x_min, y_min = boxes[:, :2].min(axis=0)
        x_max, y_max = boxes[:, 2:].max(axis=0)
        return (
            (x_min - padding, x_max + padding),
            (y_min - padding, y_max + padding),
        )
    if cfg.key_keypoints is not None:
        keypoints = trajectory[cfg.key_keypoints][..., :2].reshape(-1, 2)
        x_min, y_min = keypoints.min(axis=0)
        x_max, y_max = keypoints.max(axis=0)
        return (
            (x_min - padding, x_max + padding),
            (y_min - padding, y_max + padding),
        )
    raise NotImplementedError(
        "trajectories range not implemented for trajectories without boxes or keypoints"
    )
