from itertools import permutations
from typing import Iterable, Literal, Optional, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ...data_structures import Trajectory
from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ..annotations.utils import check_annotations
from ._dataset_base import BaseDataset
from ._sampleable import AnnotatedSampleable, Sampleable
from .dyad import Dyad
from .individual import Individual
from .utils import (
    DyadIdentity,
    Identity,
    get_concatenated_dataset,
    recursive_sampleables,
)


class Group(BaseDataset):
    def __init__(
        self,
        trajectories: dict[Identity, Trajectory],
        *,
        target: Literal["individuals", "dyads"],
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    ) -> None:
        super().__init__()
        self.exclude = [] if exclude is None else list(exclude)
        if len(trajectories) == 0:
            raise ValueError("provide at least one trajectory.")
        self.trajectories = trajectories
        self._sampling_target = target
        self._sampleables: dict[Identity | DyadIdentity, Sampleable] = {}
        if target == "individuals":
            for identity in self.identities:
                if identity in self.exclude:
                    continue
                self._sampleables[identity] = Individual(self.trajectories[identity])
        elif target == "dyads":
            for actor, recipient in list(permutations(self.identities, 2)):
                if (actor, recipient) in self.exclude:
                    continue
                trajectory, trajectory_other = Sampleable.prepare_trajectories(
                    self.trajectories[actor],
                    self.trajectories[recipient],
                )
                self._sampleables[(actor, recipient)] = Dyad(
                    trajectory=trajectory,
                    trajectory_other=trajectory_other,
                )
        else:
            raise ValueError(
                "sampling target should be one of 'individuals' or 'dyads'"
            )

    @property
    def target(self) -> Literal["individuals", "dyads"]:
        assert (
            self._sampling_target == "individuals" or self._sampling_target == "dyads"
        )
        return self._sampling_target

    @property
    def identities(self) -> tuple[Identity, ...]:
        return tuple(sorted(self.trajectories.keys()))

    def annotate(
        self, annotations: pd.DataFrame, *, categories: tuple[str, ...]
    ) -> "AnnotatedGroup":
        return AnnotatedGroup(
            self.trajectories,
            target=self.target,
            annotations=annotations,
            categories=categories,
            exclude=self.exclude,
        )

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
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        return get_concatenated_dataset(
            recursive_sampleables(self, exclude=exclude),
            feature_extractor,
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
            sampling_type="sample",
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
        store_indices: bool = False,
        exclude_stored_indices: bool = False,
        reset_stored_indices: bool = False,
        categories: Optional[list[str]] = None,
        try_even_subsampling: bool = True,
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        if exclude is None:
            exclude = []
        return get_concatenated_dataset(
            recursive_sampleables(self, exclude=exclude),
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
        )

    @property
    def sampling_targets(self) -> list[Sampleable]:
        return list(self._sampleables.values())

    @property
    def keys(self) -> list[Identity | DyadIdentity]:
        # use for select
        return list(self._sampleables.keys())

    def select(self, key: Identity | DyadIdentity) -> Sampleable:
        return self._sampleables[key]

    @property
    def label_encoder(self) -> OneHotEncoder:
        raise ValueError("non annotated group can not encode labels")


class AnnotatedGroup(Group):
    def __init__(
        self,
        trajectories: dict[Identity, Trajectory],
        *,
        target: Literal["individuals", "dyads"],
        exclude: Optional[Iterable[Identity | DyadIdentity]] = None,
        annotations: pd.DataFrame,
        categories: tuple[str, ...],
    ) -> None:
        self._sampleables: dict[  # type: ignore
            Identity | DyadIdentity, AnnotatedSampleable
        ] = {}
        super().__init__(trajectories, target=target, exclude=exclude)
        required_columns = [*AnnotatedSampleable.required_columns, "actor"]
        if self.target == "dyads":
            required_columns.append("recipient")
        self._annotations = check_annotations(
            annotations,
            required_columns=required_columns,
            allow_overlapping=True,
            allow_unsorted=True,
        )
        self._categories: tuple[str, ...] = tuple()
        for key, sampleable in self._sampleables.items():
            if self.target == "dyads":
                assert isinstance(key, tuple)
                annotations = self.get_annotations(*key)
            else:
                assert isinstance(key, str | int)
                annotations = self.get_annotations(key, None)
            annotated_sampleable = self._sampleables[key].annotate(
                annotations, categories=categories
            )
            self._sampleables[key] = annotated_sampleable
            if len(self._categories) == 0:
                self._categories = annotated_sampleable.categories

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories

    def get_annotations(
        self, actor: Identity, recipient: Optional[Identity]
    ) -> pd.DataFrame:
        if self.target == "dyads" and recipient is None:
            raise ValueError("provide recipient for sampling target 'dyads'")
        elif self.target == "dyads":
            annotations = self._annotations.set_index(["actor", "recipient"])
            try:
                annotations = annotations.loc[[(actor, recipient)]]
            except KeyError:
                annotations = annotations.iloc[:0]
        else:
            annotations = self._annotations.set_index(["actor"])
            try:
                annotations = annotations.loc[[actor]]
            except KeyError:
                annotations = annotations.iloc[:0]
        return annotations.reset_index(drop=True).sort_values("start")

    @property
    def label_encoder(self) -> OneHotEncoder:
        if self._label_encoder is None:
            categories = np.asarray(self.categories).reshape(-1, 1)
            self._label_encoder = OneHotEncoder(sparse_output=False).fit(categories)
        return self._label_encoder
