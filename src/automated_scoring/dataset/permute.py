from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..data_structures.utils import get_interval_slice
from ..features import keypoint_distances
from .types import AnnotatedDataset, AnnotatedGroup, Dyad, Group
from .utils import IndividualIdentifier


def get_proximitry_matrix(group: Group | AnnotatedGroup) -> tuple[np.ndarray, np.ndarray]:
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
            dyad = Dyad(*Dyad.prepare_paired_trajectories(trajectory, trajectory_other))
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
    identities: Sequence[IndividualIdentifier],
    proximitry_matrix: np.ndarray,
    timestamps: np.ndarray,
) -> IndividualIdentifier:
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
    if TYPE_CHECKING:
        assert isinstance(annotations_group, pd.DataFrame)
    annotations_group["recipient"] = annotations_group.apply(
        _non_recipient_neighbor,
        axis=1,
        rank=neighbor_rank,
        identities=list(group.trajectories.keys()),
        proximitry_matrix=proximitry_matrix,
        timestamps=timestamps,
    )
    annotations_group = annotations_group.sort_values(
        ["recipient", "start"], ignore_index=True, inplace=False
    )
    return group.annotate(
        annotations_group,
        categories=group.categories,
        background_category=group.background_category,
    )


def _permute_recipients(
    dataset: AnnotatedDataset, *, neighbor_rank: int
) -> AnnotatedDataset:
    groups = {}
    for group_id in dataset.identifiers:
        group = dataset.select(group_id)
        if not isinstance(group, AnnotatedGroup):
            raise ValueError("can only replace recipients in annotated groups")
        groups[group_id] = _permute_recipients_in_group(
            group, neighbor_rank=neighbor_rank
        )
    return AnnotatedDataset.from_groups(groups)


def permute_recipients[T: AnnotatedDataset | AnnotatedGroup](
    sampleable: T, *, neighbor_rank: int
) -> T:
    if isinstance(sampleable, AnnotatedDataset):
        return _permute_recipients(sampleable, neighbor_rank=neighbor_rank)
    if isinstance(sampleable, AnnotatedGroup):
        return _permute_recipients_in_group(sampleable, neighbor_rank=neighbor_rank)
    raise TypeError("dataset_or_group must be a Dataset or AnnotatedGroup")
