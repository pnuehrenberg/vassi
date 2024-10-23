from typing import Optional, Sequence, overload

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
    _groups: list[Group] | dict[Identity, Group]

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
        self._groups = groups

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
            random_state=random_state,
            stratify_by_groups=stratify_by_groups,
            store_indices=store_indices,
            exclude_stored_indices=exclude_stored_indices,
            reset_stored_indices=reset_stored_indices,
            categories=categories,
            try_even_subsampling=try_even_subsampling,
            sampling_type="sample",
            exclude=exclude,
        )

    @property
    def sampling_targets(self) -> list[Sampleable]:
        return [
            sampling_target
            for group in self._groups_list
            for sampling_target in group.sampling_targets
        ]

    def select(self, key: Identity) -> Group:
        if isinstance(self._groups, dict):
            return self._groups[key]
        if not isinstance(key, int):
            raise ValueError("provide integer type index to select from groups as list")
        return self._groups[key]

    @property
    def label_encoder(self) -> OneHotEncoder:
        if self._label_encoder is None:
            self._label_encoder = self._groups_list[0].label_encoder
        return self._label_encoder
