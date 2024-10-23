def get_instance_range(instance, padding=0):
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


def get_trajectory_range(trajectory, padding=0):
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
