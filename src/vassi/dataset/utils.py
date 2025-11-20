from __future__ import annotations

from ..data_structures import Trajectory


def get_num_samples(size: int | float, num_samples: int) -> int:
    if isinstance(size, float):
        if size < 0 or size > 1:
            raise ValueError("size must be between 0.0 and 1.0 if specified as float.")
        return int(size * num_samples)
    if size < 0 or size > num_samples:
        raise ValueError(
            f"size must be between 0 and num_samples ({num_samples}) if specified as integer."
        )
    return size


def check_trajectory(trajectory: Trajectory) -> Trajectory:
    if not trajectory.is_sorted:
        raise ValueError("trajectory is not sorted.")
    if not trajectory.is_complete:
        raise ValueError("trajectory is not complete.")
    return trajectory


def prepare_paired_trajectories(
    trajectory: Trajectory, trajectory_other: Trajectory
) -> tuple[Trajectory, Trajectory]:
    trajectory = check_trajectory(trajectory)
    trajectory_other = check_trajectory(trajectory_other)
    start = max(trajectory.timestamps[0], trajectory_other.timestamps[0])
    stop = min(trajectory.timestamps[-1], trajectory_other.timestamps[-1])
    if trajectory.timestamps[0] < start or trajectory.timestamps[-1] > stop:
        trajectory = trajectory.slice_window(start, stop, interpolate=False, copy=False)
    if trajectory_other.timestamps[0] < start or trajectory_other.timestamps[-1] > stop:
        trajectory_other = trajectory_other.slice_window(
            start, stop, interpolate=False, copy=False
        )
    return trajectory, trajectory_other
