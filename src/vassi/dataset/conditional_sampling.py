from collections.abc import Hashable, Iterable, Mapping
from typing import Literal, overload

import numpy as np

from ..data_structures import Trajectory
from ..data_structures.utils import get_interval_slice
from ..type_guards import (
    is_mapping_of,
    is_mapping_of_mappings_of,
    is_tuple_of,
)
from .group_metrics import (
    actor_activity_timeline,
    dyad_neighbor_rank,
    group_distance_matrix,
)
from .stratification import biased_stratified_subsample
from .types import (
    AnnotatedDataset,
    AnnotatedGroup,
    Dataset,
    Group,
    WithAnnotations,
    WithCategories,
)
from .utils import get_num_samples


@overload
def get_non_recipient_foreground_indices(
    element_collection: AnnotatedGroup,
    trajectories: Mapping[Hashable, Trajectory],
    *,
    frequency_per_rank: Mapping[int | Iterable[int], float],
    min_samples_per_stratum: int,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray: ...


@overload
def get_non_recipient_foreground_indices(
    element_collection: AnnotatedDataset,
    trajectories: Mapping[Hashable, Mapping[Hashable, Trajectory]],
    *,
    frequency_per_rank: Mapping[int | Iterable[int], float],
    min_samples_per_stratum: int,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray: ...


def get_non_recipient_foreground_indices(
    element_collection: AnnotatedGroup | AnnotatedDataset,
    trajectories: Mapping[Hashable, Trajectory]
    | Mapping[Hashable, Mapping[Hashable, Trajectory]],
    *,
    frequency_per_rank: Mapping[int | Iterable[int], float],
    min_samples_per_stratum: int,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    sample_description = element_collection.describe_available_samples(
        reset=False, exclude_previously_sampled=False
    )
    interval_levels = sample_description[:, 1]
    selected_indices: list[np.ndarray] = []
    for rank, frequency in frequency_per_rank.items():
        if not isinstance(rank, int):
            rank = set(rank)
        if isinstance(element_collection, AnnotatedGroup) and is_mapping_of(
            trajectories, Hashable, Trajectory
        ):
            indices = get_indices_where(
                element_collection,
                trajectories=trajectories,
                category=set(element_collection.foreground_categories),
                broadcast_actor_y_across_dyads=True,
                rank=rank,
                exclude_recipient=True,
            )
        elif isinstance(
            element_collection, AnnotatedDataset
        ) and is_mapping_of_mappings_of(trajectories, Hashable, Hashable, Trajectory):
            indices = get_indices_where(
                element_collection,
                trajectories=trajectories,
                category=set(element_collection.foreground_categories),
                broadcast_actor_y_across_dyads=True,
                rank=rank,
                exclude_recipient=True,
            )
        else:
            raise ValueError("Invalid input combination")
        _, strata_labels = np.unique(
            interval_levels[indices], axis=0, return_inverse=True
        )
        selected_indices.append(
            indices[
                biased_stratified_subsample(
                    strata_labels,
                    size=get_num_samples(frequency, indices.size),
                    min_samples_per_stratum=min_samples_per_stratum,
                    random_state=random_state,
                )
            ]
        )
    indices = np.concatenate(selected_indices, dtype=np.uint64)
    return indices


# 1. broadcast=True -> category is REQUIRED
@overload
def get_indices_where(
    element_collection: Group,
    *,
    trajectories: Mapping[Hashable, Trajectory],
    category: str | Iterable[str],
    rank: int | Iterable[int] | None = None,
    broadcast_actor_y_across_dyads: Literal[True],
    exclude_recipient: bool = False,
) -> np.ndarray: ...


# 2. broadcast=False -> category is OPTIONAL
@overload
def get_indices_where(
    element_collection: Group,
    *,
    trajectories: Mapping[Hashable, Trajectory],
    category: str | Iterable[str] | None = None,
    rank: int | Iterable[int] | None = None,
    broadcast_actor_y_across_dyads: Literal[False] = False,
    exclude_recipient: bool = False,
) -> np.ndarray: ...


# 3. broadcast=True -> category is REQUIRED
@overload
def get_indices_where(
    element_collection: Dataset,
    *,
    trajectories: Mapping[Hashable, Mapping[Hashable, Trajectory]],
    category: str | Iterable[str],
    rank: int | Iterable[int] | None = None,
    broadcast_actor_y_across_dyads: Literal[True],
    exclude_recipient: bool = False,
) -> np.ndarray: ...


# 4. broadcast=False -> category is OPTIONAL
@overload
def get_indices_where(
    element_collection: Dataset,
    *,
    trajectories: Mapping[Hashable, Mapping[Hashable, Trajectory]],
    category: str | Iterable[str] | None = None,
    rank: int | Iterable[int] | None = None,
    broadcast_actor_y_across_dyads: Literal[False] = False,
    exclude_recipient: bool = False,
) -> np.ndarray: ...


def get_indices_where(
    element_collection: Group | Dataset,
    *,
    trajectories: Mapping[Hashable, Trajectory]
    | Mapping[Hashable, Mapping[Hashable, Trajectory]],
    category: str | Iterable[str] | None = None,
    rank: int | Iterable[int] | None = None,
    broadcast_actor_y_across_dyads: bool = False,
    exclude_recipient: bool = False,
) -> np.ndarray:
    """
    Returns a flat array of indices where the specified conditions are met.
    """
    if (
        rank is not None or broadcast_actor_y_across_dyads
    ) and element_collection.get_target() != "dyad":
        raise ValueError("Expected element collection with dyads")
    if category is not None and not isinstance(element_collection, WithCategories):
        raise ValueError("Expected an annotated element collection")
    if isinstance(element_collection, Dataset):
        if not is_mapping_of_mappings_of(trajectories, Hashable, Hashable, Trajectory):
            raise ValueError("Expected mapping of mappings for Dataset trajectories")
        mask = _get_dataset_mask(
            dataset=element_collection,
            trajectories=trajectories,
            category=category,
            rank=rank,
            broadcast_actor_y_across_dyads=broadcast_actor_y_across_dyads,
            exclude_recipient=exclude_recipient,
        )
    else:
        if not is_mapping_of(trajectories, Hashable, Trajectory):
            raise ValueError("Expected mapping of trajectories for Group")
        mask = _get_group_mask(
            group=element_collection,
            trajectories=trajectories,
            category=category,
            rank=rank,
            broadcast_actor_y_across_dyads=broadcast_actor_y_across_dyads,
            exclude_recipient=exclude_recipient,
        )
    return np.argwhere(mask).ravel().astype(np.uint64)


def _get_group_mask(
    group: Group,
    *,
    trajectories: Mapping[Hashable, Trajectory],
    category: str | Iterable[str] | None,
    rank: int | Iterable[int] | None,
    broadcast_actor_y_across_dyads: bool,
    exclude_recipient: bool,
) -> np.ndarray:
    mask = np.ones(group.num_samples, dtype=bool)
    actor_broadcast_timelines: dict[Hashable, np.ndarray] = {}
    if broadcast_actor_y_across_dyads and category is not None:
        assert isinstance(group, AnnotatedGroup)
        for actor in group.actors():
            actor_broadcast_timelines[actor] = actor_activity_timeline(
                group=group,
                actor=actor,
                trajectories=trajectories,
                category=category,
            )
    offset = 0
    background = None
    y_target = None
    if category is not None and not broadcast_actor_y_across_dyads:
        assert isinstance(group, WithCategories), (
            f"Expected group to be of mixin type {WithCategories}, got {type(group)}"
        )
        categories = [category] if isinstance(category, str) else sorted(category)
        y_target = group.encode(np.asarray(categories))
    if exclude_recipient:
        assert isinstance(group, WithCategories), (
            f"Expected group to be of mixin type {WithCategories}, got {type(group)}"
        )
        background = group.category_index(group.background_category)
        if background is None:
            background = -1
    distance_matrix_result = None
    if rank is not None:
        distance_matrix_result = group_distance_matrix(group, trajectories=trajectories)
    for identifier, element in group:
        if not is_tuple_of(identifier, Hashable):
            raise ValueError("Invalid identifier")
        actor, recipient = identifier
        if actor not in group.actors():
            continue
        num_samples = element.num_samples
        element_mask_view = mask[offset : offset + num_samples]
        if category is not None and broadcast_actor_y_across_dyads:
            # Project Actor Timeline -> Dyad Timeline
            element_mask_view &= actor_broadcast_timelines[actor][
                get_interval_slice(
                    trajectories[actor].timestamps,
                    element.trajectory.timestamps[0],
                    element.trajectory.timestamps[-1],
                )
            ]
        elif category is not None:
            assert y_target is not None, "Expected y_target to be specified"
            assert isinstance(element, WithAnnotations), (
                f"Expected element to be of mixin type {WithAnnotations}, got {type(element)}"
            )
            y = element.sample_y(indices=None, out=None)
            element_mask_view &= np.isin(y, y_target)
        if rank is not None:
            assert distance_matrix_result is not None, (
                "Expected distance_matrix_result to be specified"
            )
            rank_mask = dyad_neighbor_rank(
                group=group,
                actor=actor,
                recipient=recipient,
                trajectories=trajectories,
                **distance_matrix_result,
            )
            element_mask_view &= np.isin(
                rank_mask,
                [rank] if isinstance(rank, int) else list(rank),
            )
        if exclude_recipient:
            assert isinstance(element, WithAnnotations), (
                f"Expected element to be of mixin type {WithAnnotations}, got {type(element)}"
            )
            assert background is not None, "Expected background to be specified"
            element_mask_view &= element.sample_y(indices=None, out=None) == background
        offset += num_samples
    assert offset == mask.shape[0], "Group sample count mismatch"
    return mask


def _get_dataset_mask(
    dataset: Dataset,
    *,
    trajectories: Mapping[Hashable, Mapping[Hashable, Trajectory]],
    category: str | Iterable[str] | None,
    rank: int | Iterable[int] | None,
    broadcast_actor_y_across_dyads: bool,
    exclude_recipient: bool,
) -> np.ndarray:
    mask = np.empty(dataset.num_samples, dtype=bool)
    offset = 0
    for group_id, group in dataset:
        group_mask = _get_group_mask(
            group,
            trajectories=trajectories[group_id],
            category=category,
            rank=rank,
            broadcast_actor_y_across_dyads=broadcast_actor_y_across_dyads,
            exclude_recipient=exclude_recipient,
        )
        num_samples = group.num_samples
        mask[offset : offset + num_samples] = group_mask
        offset += num_samples
    assert offset == mask.shape[0], "Dataset sample count mismatch"
    return mask
