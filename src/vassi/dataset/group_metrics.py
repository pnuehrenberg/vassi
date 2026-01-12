from collections.abc import Hashable, Iterable, Mapping
from typing import TypedDict, overload

import numpy as np

from ..data_structures import Trajectory
from ..data_structures.utils import get_interval_slice
from ..fast_math import magnitude
from ..type_guards import (
    is_iterable_of,
    is_tuple_of,
)
from .types import AnnotatedGroup, Dyad, Group


class GroupDistanceMatrixResult(TypedDict):
    distance_matrix: np.ndarray
    start: int | float
    stop: int | float
    actors_map: dict[Hashable, int]
    individuals_map: dict[Hashable, int]


def group_distance_matrix(
    group: Group,
    *,
    trajectories: Mapping[Hashable, Trajectory],
) -> GroupDistanceMatrixResult:
    """
    Computes a (time, actors, individuals) distance matrix for the entire group. Distances to self are set to nan.

    Returns:
        Distance Matrix
            (T, N_actors, N_individuals) filled with nan where invalid.
        start
            Minimum start timestamp across all actors.
        stop
            Maximum stop timestamp across all actors.
        actors_map
            Mapping from actor to row index.
        individuals_map
            Mapping from individual to column index.
    """
    trajectory = None
    for trajectory in trajectories.values():
        assert np.isclose(trajectory.timestamps % 1, 0, rtol=0, atol=1e-8).all(), (
            "expected timestamps to be integer or float multiples of 1 (atol=1e-8)."
        )
    if trajectory is None:
        raise ValueError("expected at least one trajectory")
    cfg = trajectory.cfg
    assert cfg.key_keypoints is not None, "expected configured keypoints"
    actors = group.actors()
    if is_iterable_of(actors, int):
        actors = sorted(actors)
    elif is_iterable_of(actors, str):
        actors = sorted(actors)
    else:
        raise ValueError("actors must be iterable of int or str")
    individuals = trajectories.keys()
    if is_iterable_of(individuals, int):
        individuals = sorted(individuals)
    elif is_iterable_of(individuals, str):
        individuals = sorted(individuals)
    else:
        raise ValueError("individuals must be iterable of int or str")
    actors_map: dict[Hashable, int] = {actor: idx for idx, actor in enumerate(actors)}
    individuals_map: dict[Hashable, int] = {
        individual: idx for idx, individual in enumerate(individuals)
    }
    start = round(
        min([trajectories[individual].timestamps[0] for individual in individuals])
    )
    stop = round(
        max([trajectories[individual].timestamps[-1] for individual in individuals])
    )
    keypoints_shape = trajectory[cfg.key_keypoints].shape[1:]
    duration = stop - start + 1
    keypoints_actors = np.full((duration, len(actors), *keypoints_shape), np.nan)
    keypoints_individuals = np.full(
        (duration, len(individuals), *keypoints_shape), np.nan
    )
    for individual in set(actors) | set(individuals):
        trajectory = trajectories[individual]
        trajectory_start, trajectory_stop = (
            round(trajectory.timestamps[0]),
            round(trajectory.timestamps[-1]),
        )
        if individual in actors:
            keypoints_actors[
                trajectory_start - start : trajectory_stop - start + 1,
                actors_map[individual],
            ] = trajectory[cfg.key_keypoints]
        if individual in individuals:
            keypoints_individuals[
                trajectory_start - start : trajectory_stop - start + 1,
                individuals_map[individual],
            ] = trajectory[cfg.key_keypoints]
    distance_matrix = (
        magnitude(
            keypoints_actors[:, :, np.newaxis] - keypoints_individuals[:, np.newaxis]
        )
        .reshape(duration, len(actors), len(individuals), keypoints_shape[-2])
        .min(axis=-1)
    )
    for actor in actors:
        distance_matrix[:, actors_map[actor], individuals_map[actor]] = np.nan
    return GroupDistanceMatrixResult(
        distance_matrix=distance_matrix,
        start=start,
        stop=stop,
        actors_map=actors_map,
        individuals_map=individuals_map,
    )


@overload
def dyad_neighbor_rank(
    group: Group,
    actor: Hashable,
    recipient: Hashable,
    *,
    trajectories: Mapping[Hashable, Trajectory],
) -> np.ndarray: ...


@overload
def dyad_neighbor_rank(
    group: Group,
    actor: Hashable,
    recipient: Hashable,
    *,
    trajectories: Mapping[Hashable, Trajectory],
    distance_matrix: np.ndarray,
    start: int | float,
    stop: int | float,
    actors_map: dict[Hashable, int],
    individuals_map: dict[Hashable, int],
) -> np.ndarray: ...


def dyad_neighbor_rank(
    group: Group,
    actor: Hashable,
    recipient: Hashable,
    *,
    trajectories: Mapping[Hashable, Trajectory],
    distance_matrix: np.ndarray | None = None,
    start: int | float | None = None,
    stop: int | float | None = None,
    actors_map: dict[Hashable, int] | None = None,
    individuals_map: dict[Hashable, int] | None = None,
) -> np.ndarray:
    """
    Computes the spatial rank mask for a single dyad.

    Handles recipient at nan distances, temporal intersection, minimum keypoint distance.
    """
    dyad = group[(actor, recipient)]
    assert isinstance(dyad, Dyad), f"Expected group of {Dyad}, not {type(dyad)}"
    if distance_matrix is not None:
        assert start is not None and stop is not None
        assert actors_map is not None and individuals_map is not None
        # corrected behavior: keyerror when actor or recipient not in respective maps
        row_idx = actors_map[actor]
        col_idx = individuals_map[recipient]
        dyad_start = round(dyad.trajectory.timestamps[0])
        dyad_stop = round(dyad.trajectory.timestamps[-1])
        slice_start = dyad_start - round(start)
        slice_stop = dyad_stop - round(start) + 1
        distance_matrix_actor = distance_matrix[slice_start:slice_stop, row_idx, :]
        distances_dyad = distance_matrix_actor[:, col_idx]
        current_rank = np.sum(distance_matrix_actor < distances_dyad[:, None], axis=1)
        current_rank[np.isnan(distances_dyad)] = -1
        return current_rank.astype(int)
    cfg = trajectories[actor].cfg
    assert cfg.key_keypoints is not None, "Keypoints must be defined in the config."
    # Use strict window from dyad
    start = dyad.trajectory.timestamps[0]
    stop = dyad.trajectory.timestamps[-1]
    keypoints_actor = dyad.trajectory[cfg.key_keypoints]
    keypoints_recipient = dyad.trajectory_other[cfg.key_keypoints]
    distances_dyad = magnitude(keypoints_actor - keypoints_recipient).min(axis=1)
    # compare with neighbors, initialize rank 0.
    current_rank = np.zeros(distances_dyad.shape, dtype=int)
    for other, trajectory_other in trajectories.items():
        if other == actor or other == recipient:
            continue
        # find temporal overlap between dyad (start/stop) and neighbor (start/stop)
        intersection_start = max(start, trajectory_other.timestamps[0])
        intersection_stop = min(stop, trajectory_other.timestamps[-1])
        if intersection_start > intersection_stop:
            continue  # No overlap
        # get slice relative to the dyads's timestamps where overlap occurs
        intersection_slice = get_interval_slice(
            dyad.trajectory.timestamps, intersection_start, intersection_stop
        )
        # slice neighbor keypoints
        keypoints_other = trajectory_other.slice_window(
            intersection_start, intersection_stop, copy=False, interpolate=False
        )[cfg.key_keypoints]
        # slice actor/dyad data to match overlap
        keypoints_actor_intersection = keypoints_actor[intersection_slice]
        distances_dyad_intersection = distances_dyad[intersection_slice]
        # calculate neighbor distance
        distance_other = magnitude(keypoints_actor_intersection - keypoints_other).min(
            axis=1
        )
        # and compare (if distances_dyad is nan, comparison is False, rank doesn't increase).
        current_rank[intersection_slice] += (
            distance_other < distances_dyad_intersection
        ).astype(int)
    # if recipient was at nan distance, the calculated rank (0) is meaningless -> assign rank -1 at nan distances
    current_rank[np.isnan(distances_dyad)] = -1
    return current_rank


def actor_activity_timeline(
    group: AnnotatedGroup,
    actor: Hashable,
    *,
    trajectories: Mapping[Hashable, Trajectory],
    category: str | Iterable[str],
) -> np.ndarray:
    """
    Computes a boolean timeline for a single actor.

    True indicates the actor is performing the target category with any recipient in the group at that timestamp.
    """
    # need the actor's full trajectory to determine the output shape/timeline
    trajectory_actor = trajectories[actor]
    activity_mask = np.zeros(len(trajectory_actor), dtype=bool)
    categories_to_encode = [category] if isinstance(category, str) else sorted(category)
    y_target = group.encode(np.asarray(categories_to_encode))
    # look for any dyad in the group where 'actor' is the actor.
    for identifier, dyad in group:
        assert is_tuple_of(identifier, Hashable), (
            f"Expected a valid dyad identifier (tuple of Hashable), got {identifier}"
        )
        if identifier[0] != actor:
            continue
        y_dyad = dyad.sample_y(indices=None, out=None)
        is_active = np.isin(y_dyad, y_target)
        if not np.any(is_active):
            continue
        # find where this dyad's window fits into the actor's global timeline
        # and accumulate activity (logical OR)
        activity_mask[
            get_interval_slice(
                trajectory_actor.timestamps,
                dyad.trajectory.timestamps[0],
                dyad.trajectory.timestamps[-1],
            )
        ] |= is_active
    return activity_mask
