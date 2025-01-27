from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...data_structures.utils import get_interval_slice
from ...features import keypoint_distances
from .. import AnnotatedGroup, Dataset, Dyad, Group
from ..types.utils import Identity


def get_proximitry_matrix(group: Group | AnnotatedGroup) -> tuple[NDArray, NDArray]:
    steps = set(trajectory.timestep for trajectory in group.trajectories.values())
    if len(steps) != 1:
        raise ValueError("all trajectories must have the same timestep")
    step = steps.pop()
    first: float = min(
        [trajectory.timestamps[0] for trajectory in group.trajectories.values()]
    )
    last: float = max(
        [trajectory.timestamps[-1] for trajectory in group.trajectories.values()]
    )
    timestamps = np.arange(first, last + step / 2, step)

    proximitry_matrix = np.zeros(
        (len(timestamps), len(group.trajectories), len(group.trajectories)), dtype=float
    )
    proximitry_matrix[:] = np.inf
    for u, (identity, trajectory) in enumerate(group.trajectories.items()):
        for v, (identity_other, trajectory_other) in enumerate(
            group.trajectories.items()
        ):
            if identity == identity_other:
                continue
            dyad = Dyad(*Dyad.prepare_trajectories(trajectory, trajectory_other))
            distances = keypoint_distances(
                dyad.trajectory,
                trajectory_other=dyad.trajectory_other,
                keypoints_1=(0,),
                keypoints_2=(0, 1, 2),
                flat=True,
            ).min(axis=1)
            proximitry_matrix[np.isin(timestamps, dyad.trajectory.timestamps), u, v] = (
                distances
            )
    return proximitry_matrix, timestamps


def _non_recipient_neighbor(
    annotation: pd.Series,
    *,
    rank: int,
    identities: Sequence[Identity],
    proximitry_matrix: NDArray,
    timestamps: NDArray,
) -> Identity:
    identities = list(identities)
    if rank >= len(identities) - 2:
        raise ValueError(
            f"rank too high. rank should be smaller than {len(identities) - 2} (number of identities - 2)"
        )
    start, stop = np.asarray(annotation[["start", "stop"]]).astype(float)
    actor = annotation["actor"]
    recipient = annotation["recipient"]
    if not isinstance(actor, int | str) or not isinstance(recipient, int | str):
        raise ValueError("actor and recipient must be valid identities (int or str)")
    actor_idx = identities.index(actor)
    neighbor_idx = np.argsort(
        proximitry_matrix[get_interval_slice(timestamps, start, stop), actor_idx],
        axis=1,
    )
    neighbor_ranks = np.array(
        [
            np.average(
                np.arange(neighbor_idx.shape[1]),
                weights=(neighbor_idx == idx).sum(axis=0),
            )
            for idx in range(neighbor_idx.shape[1])
        ]
    )
    ranked_neighbors = [
        identities[idx]
        for idx in np.argsort(neighbor_ranks)
        if identities[idx] not in (actor, recipient)
    ]
    return ranked_neighbors[rank]


def _permute_recipients_in_group(
    group: AnnotatedGroup, *, neighbor_rank: int
) -> AnnotatedGroup:
    proximitry_matrix, timestamps = get_proximitry_matrix(group)
    annotations_group = group.observations
    annotations_group = annotations_group[annotations_group["category"] != "none"]
    annotations_group["recipient"] = annotations_group.apply(
        _non_recipient_neighbor,
        args=(
            neighbor_rank,
            list(group.trajectories.keys()),
            proximitry_matrix,
            timestamps,
        ),
        axis=1,
    )
    if TYPE_CHECKING:
        assert isinstance(annotations_group, pd.DataFrame)
    return group.annotate(annotations_group, categories=group.categories)


def _permute_recipients(dataset: Dataset, *, neighbor_rank: int) -> Dataset:
    groups = {}
    for group_key in dataset.group_keys:
        group = dataset.select(group_key)
        if not isinstance(group, AnnotatedGroup):
            raise ValueError("can only replace recipients in annotated groups")
        groups[group_key] = _permute_recipients_in_group(
            group, neighbor_rank=neighbor_rank
        )
    return Dataset(groups)


@overload
def permute_recipients(dataset: Dataset, *, neighbor_rank: int) -> Dataset: ...


@overload
def permute_recipients(
    dataset: AnnotatedGroup, *, neighbor_rank: int
) -> AnnotatedGroup: ...


def permute_recipients(
    dataset: Dataset | AnnotatedGroup, *, neighbor_rank: int
) -> Dataset | AnnotatedGroup:
    if isinstance(dataset, Dataset):
        return _permute_recipients(dataset, neighbor_rank=neighbor_rank)
    elif isinstance(dataset, AnnotatedGroup):
        return _permute_recipients_in_group(dataset, neighbor_rank=neighbor_rank)
    else:
        raise TypeError("dataset_or_group must be a Dataset or AnnotatedGroup")
