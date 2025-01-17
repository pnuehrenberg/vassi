import random
from collections.abc import Generator, Iterable
from typing import Any, Optional, Sequence, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder

from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ...utils import ensure_generator, to_int_seed
from ._dataset_base import BaseDataset
from ._sampleable import Sampleable
from .group import AnnotatedGroup, Group
from .utils import (
    DyadIdentity,
    Identity,
    get_concatenated_dataset,
    recursive_sampleables,
)

# same as DyadIdentity, but for different purposes
ActorIdentifier = tuple[Identity, Identity]


def get_actor(key: Identity | DyadIdentity) -> Identity:
    if isinstance(key, tuple):
        return key[0]
    return key


def get_subgroup(
    group: Group,
    selected_actors: list[Identity],
    exclude: Iterable[Identity | DyadIdentity],
):
    remaining = [
        key
        for key in group.keys
        if (key in exclude) or (get_actor(key) not in selected_actors)
    ]
    if isinstance(group, AnnotatedGroup):
        return AnnotatedGroup(
            group.trajectories,
            target=group.target,
            observations=group._observations,
            categories=group._categories,
            exclude=remaining,
        )
    return Group(
        group.trajectories,
        target=group.target,
        exclude=remaining,
    )


class Dataset(BaseDataset):
    _groups_list: list[Group]
    groups: list[Group] | dict[Identity, Group]

    def __init__(self, groups: Sequence[Group] | dict[Identity, Group]) -> None:
        super().__init__()
        if len(groups) == 0:
            raise ValueError("provide at least one group.")
        if isinstance(groups, dict):
            _groups_list = list(groups.values())
        else:
            _groups_list = list(groups)
            groups = list(groups)
        is_annotated = isinstance(_groups_list[0], AnnotatedGroup)
        if any(
            [
                isinstance(group, AnnotatedGroup) != is_annotated
                for group in _groups_list[1:]
            ]
        ):
            raise ValueError(
                "groups should be either all annotated, or all not annotated."
            )
        self._groups_list = _groups_list
        self.groups = groups

    @overload
    def sample(
        self,
        feature_extractor: FeatureExtractor,
        *args,
        **kwargs,
    ) -> tuple[NDArray, NDArray | None]: ...

    @overload
    def sample(
        self,
        feature_extractor: DataFrameFeatureExtractor,
        *args,
        **kwargs,
    ) -> tuple[pd.DataFrame, NDArray | None]: ...

    def sample(
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
        show_progress: bool = True,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        return get_concatenated_dataset(
            recursive_sampleables(self, exclude=exclude),
            feature_extractor,
            sampling_type="sample",
            show_progress=show_progress,
        )

    @overload
    def subsample(
        self, feature_extractor: FeatureExtractor, *args, **kwargs
    ) -> tuple[NDArray, NDArray | None]: ...

    @overload
    def subsample(
        self, feature_extractor: DataFrameFeatureExtractor, *args, **kwargs
    ) -> tuple[pd.DataFrame, NDArray | None]: ...

    def subsample(
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        size: int | float,
        *,
        random_state: Optional[np.random.Generator | int] = None,
        stratify_by_groups: bool = True,
        categories: Optional[list[str]] = None,
        store_indices: bool = False,
        exclude_stored_indices: bool = False,
        reset_stored_indices: bool = False,
        try_even_subsampling: bool = True,
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
        show_progress: bool = True,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        if exclude is None:
            exclude = []
        return get_concatenated_dataset(
            recursive_sampleables(self, exclude=exclude),
            feature_extractor,
            size=size,
            random_state=random_state,
            stratify_by_groups=stratify_by_groups,
            store_indices=store_indices,
            exclude_stored_indices=exclude_stored_indices,
            reset_stored_indices=reset_stored_indices,
            categories=categories,
            try_even_subsampling=try_even_subsampling,
            sampling_type="subsample",
            show_progress=show_progress,
        )

    @property
    def group_keys(self) -> list[Identity]:
        # use for select
        return (
            list(self.groups.keys())
            if isinstance(self.groups, dict)
            else list(range(len(self.groups)))
        )

    def get_actors(
        self, *, exclude: Optional[Iterable[Identity | DyadIdentity]] = None
    ) -> list[ActorIdentifier]:
        exclude = list(exclude) if exclude is not None else []
        actors: list[ActorIdentifier] = []
        for group_key in self.group_keys:
            if group_key in exclude:
                continue
            group = self.select(group_key)
            for key in group.keys:
                if key in exclude:
                    continue
                identifier = group_key, get_actor(key)
                if identifier in actors:
                    continue
                actors.append(identifier)
        return actors

    @property
    def sampling_targets(self) -> list[Sampleable]:
        return [
            sampling_target
            for group in self._groups_list
            for sampling_target in group.sampling_targets
        ]

    def select(self, key: Identity) -> Group:
        if isinstance(self.groups, dict):
            return self.groups[key]
        if not isinstance(key, int):
            raise ValueError("provide integer type index to select from groups as list")
        return self.groups[key]

    @property
    def label_encoder(self) -> OneHotEncoder:
        if self._label_encoder is None:
            self._label_encoder = self._groups_list[0].label_encoder
        return self._label_encoder

    def get_observations(
        self,
        *,
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    ) -> pd.DataFrame:
        if isinstance(self.groups, list) and not all(
            [isinstance(group, AnnotatedGroup) for group in self.groups]
        ):
            raise ValueError("dataset contains non-annotated groups")
        elif isinstance(self.groups, dict) and not all(
            [isinstance(group, AnnotatedGroup) for group in self.groups.values()]
        ):
            raise ValueError("dataset contains non-annotated groups")
        from ..observations.concatenate import (
            concatenate_observations,
        )  # local import here to avoid circular import

        return concatenate_observations(self.groups, exclude=exclude)  # type: ignore (see type checking above)

    def _split(
        self,
        selected_actors: Iterable[ActorIdentifier],
        remaining_actors: Iterable[ActorIdentifier],
        *,
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    ):
        exclude = list(exclude) if exclude is not None else []
        selected_actors_by_groups: dict[Identity, list[Identity]] = {}
        remaining_actors_by_groups: dict[Identity, list[Identity]] = {}
        for group_key, identity in selected_actors:
            if group_key not in selected_actors_by_groups:
                selected_actors_by_groups[group_key] = []
            selected_actors_by_groups[group_key].append(identity)
        for group_key, identity in remaining_actors:
            if group_key not in remaining_actors_by_groups:
                remaining_actors_by_groups[group_key] = []
            remaining_actors_by_groups[group_key].append(identity)
        selected_groups = {
            group_key: get_subgroup(self.select(group_key), selected_actors, exclude)
            for group_key, selected_actors in selected_actors_by_groups.items()
        }
        remaining_groups = {
            group_key: get_subgroup(self.select(group_key), remaining_actors, exclude)
            for group_key, remaining_actors in remaining_actors_by_groups.items()
        }
        return Dataset(selected_groups), Dataset(remaining_groups)

    def split(
        self,
        size: float,
        *,
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
        random_state: Optional[np.random.Generator | int] = None,
    ) -> tuple["Dataset", "Dataset"]:
        random.seed(to_int_seed(ensure_generator(random_state)))
        if size < 0 or size > 1:
            raise ValueError("size should be between 0 and 1")
        exclude = list(exclude) if exclude is not None else []
        actors = self.get_actors(exclude=exclude)
        num_selected = int(len(actors) * size)
        num_remaining = len(actors) - num_selected
        if num_selected < 1:
            raise ValueError(
                "specified size too small. each split should contain at least one actor"
            )
        if num_remaining < 1:
            raise ValueError(
                "specified size too large. each split should contain at least one actor"
            )
        actors_selected = random.sample(actors, num_selected)
        actors_remaining = [actor for actor in actors if actor not in actors_selected]
        return self._split(actors_selected, actors_remaining, exclude=exclude)

    def k_fold(
        self,
        k: int,
        *,
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
        random_state: Optional[np.random.Generator | int] = None,
    ) -> Generator[tuple["Dataset", "Dataset"], Any, None]:
        random.seed(to_int_seed(ensure_generator(random_state)))
        actors = self.get_actors(exclude=exclude)
        num_holdout_per_fold = len(actors) // k
        if num_holdout_per_fold < 1:
            raise ValueError(
                "specified k is too large. each fold should contain at least one actor"
            )
        folds: list[tuple[list[ActorIdentifier], list[ActorIdentifier]]] = []
        actors_remaining = actors
        for _ in range(k):
            holdout = random.sample(actors_remaining, num_holdout_per_fold)
            train = [actor for actor in actors if actor not in holdout]
            folds.append((train, holdout))
            actors_remaining = [
                actor for actor in actors_remaining if actor not in holdout
            ]
        for train, holdout in folds:
            yield self._split(train, holdout, exclude=exclude)
