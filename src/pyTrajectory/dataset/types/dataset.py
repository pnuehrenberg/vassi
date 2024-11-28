import random
from collections.abc import Generator, Iterable
from typing import Any, Optional, Sequence, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ._dataset_base import BaseDataset
from ._sampleable import Sampleable
from .group import AnnotatedGroup, Group
from .utils import Identity, get_concatenated_dataset


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
        pipeline: Optional[Pipeline] = None,
        fit_pipeline: bool = True,
        exclude: Optional[list[Identity] | list[tuple[Identity, Identity]]] = None,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        if exclude is None:
            exclude = []
        return get_concatenated_dataset(
            self._groups_list,
            feature_extractor,
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
            sampling_type="sample",
            exclude=exclude,
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
        pipeline: Optional[Pipeline] = None,
        fit_pipeline: bool = True,
        random_state: Optional[np.random.Generator | int] = None,
        stratify_by_groups: bool = True,
        categories: Optional[list[str]] = None,
        store_indices: bool = False,
        exclude_stored_indices: bool = False,
        reset_stored_indices: bool = False,
        try_even_subsampling: bool = True,
        exclude: Optional[list[Identity] | list[tuple[Identity, Identity]]] = None,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        if exclude is None:
            exclude = []
        return get_concatenated_dataset(
            self._groups_list,
            feature_extractor,
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
            size=size,
            random_state=random_state,
            stratify_by_groups=stratify_by_groups,
            store_indices=store_indices,
            exclude_stored_indices=exclude_stored_indices,
            reset_stored_indices=reset_stored_indices,
            categories=categories,
            try_even_subsampling=try_even_subsampling,
            sampling_type="subsample",
            exclude=exclude,
        )

    @property
    def group_keys(self) -> list[Identity]:
        # use for select
        return (
            list(self.groups.keys())
            if isinstance(self.groups, dict)
            else list(range(len(self.groups)))
        )

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

    def get_annotations(
        self,
        *,
        exclude: Optional[list[Identity] | list[tuple[Identity, Identity]]] = None,
    ) -> pd.DataFrame:
        if isinstance(self.groups, list) and not all(
            [isinstance(group, AnnotatedGroup) for group in self.groups]
        ):
            raise ValueError("dataset contains non-annotated groups")
        elif isinstance(self.groups, dict) and not all(
            [isinstance(group, AnnotatedGroup) for group in self.groups.values()]
        ):
            raise ValueError("dataset contains non-annotated groups")
        from ..annotations.concatenate import (
            concatenate_annotations,
        )  # local import here to avoid circular import

        return concatenate_annotations(self.groups, exclude=exclude)  # type: ignore (see type checking above)

    def k_fold(
        self,
        k: int,
        exclude: Optional[Iterable[Identity | tuple[Identity, Identity]]] = None,
    ) -> Generator[tuple["Dataset", "Dataset"], Any, None]:
        def get_subgroup(group_key, selected_keys):
            group = self.select(group_key)
            exclude = [key for key in group.keys if key not in selected_keys]
            if isinstance(group, AnnotatedGroup):
                return AnnotatedGroup(
                    group.trajectories,
                    target=group.target,
                    annotations=group._annotations,
                    categories=group._categories,
                    exclude=exclude,
                )
            return Group(
                group.trajectories,
                target=group.target,
                exclude=exclude,
            )

        sampling_target_keys = []
        for group_key in self.group_keys:
            if exclude is not None and group_key in exclude:
                continue
            group = self.select(group_key)
            for key in group.keys:
                if exclude is not None and key in exclude:
                    continue
                sampling_target_keys.append((group_key, key))
        num_sampling_targetes = len(sampling_target_keys)
        num_holdout_per_fold = num_sampling_targetes // k
        if num_holdout_per_fold < 1:
            raise ValueError(
                "specified k is too large. each fold should contain at least one sampling target"
            )
        folds = []
        sampling_target_keys_remaining = sampling_target_keys
        for _ in range(k):
            holdout = random.sample(
                sampling_target_keys_remaining,
                num_holdout_per_fold,
            )
            train = [key for key in sampling_target_keys if key not in holdout]
            folds.append((train, holdout))
            sampling_target_keys_remaining = [
                key for key in sampling_target_keys_remaining if key not in holdout
            ]
        for train, holdout in folds:
            groups_train = {}
            groups_holdout = {}
            for group_key, key in train:
                if group_key not in groups_train:
                    groups_train[group_key] = []
                groups_train[group_key].append(key)
            for group_key, key in holdout:
                if group_key not in groups_holdout:
                    groups_holdout[group_key] = []
                groups_holdout[group_key].append(key)
            for group_key, selected_keys in groups_train.items():
                groups_train[group_key] = get_subgroup(group_key, selected_keys)
            for group_key, selected_keys in groups_holdout.items():
                groups_holdout[group_key] = get_subgroup(group_key, selected_keys)
            yield Dataset(groups_train), Dataset(groups_holdout)
