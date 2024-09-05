import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Self, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ...config import Config
from ...data_structures import Trajectory
from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ...utils import warning_only
from ..annotations import check_annotations, infill_annotations
from ..sampling import split
from ._dataset_base import BaseDataset

if TYPE_CHECKING:
    from .utils import Identity


class BaseSampleable(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.subsample_idx: list[NDArray] = []

    @overload
    def extract_features(self, feature_extractor: FeatureExtractor) -> NDArray: ...

    @overload
    def extract_features(
        self, feature_extractor: DataFrameFeatureExtractor
    ) -> pd.DataFrame: ...

    def extract_features(
        self, feature_extractor: FeatureExtractor | DataFrameFeatureExtractor
    ) -> NDArray | pd.DataFrame:
        raise NotImplementedError

    def _sample_y(self) -> NDArray | None:
        return None

    def _sample_groups(self) -> NDArray | None:
        return None

    @overload
    def sample(
        self, feature_extractor: FeatureExtractor, *args, **kwargs
    ) -> tuple[NDArray, NDArray | None]: ...

    @overload
    def sample(
        self, feature_extractor: DataFrameFeatureExtractor, *args, **kwargs
    ) -> tuple[pd.DataFrame, NDArray | None]: ...

    def sample(
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        pipeline: Optional[Pipeline] = None,
        fit_pipeline: bool = True,
        exclude: Optional[
            list["Identity"] | list[tuple["Identity", "Identity"]]
        ] = None,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        if exclude is not None:
            with warning_only():
                warnings.warn("Ignoring exclude keyword argument.")
        X = self.extract_features(feature_extractor)
        y = self._sample_y()
        if pipeline is None:
            return X, y
        if isinstance(feature_extractor, DataFrameFeatureExtractor):
            pipeline.set_output(transform="pandas")
        if fit_pipeline:
            return pipeline.fit_transform(X), y
        return pipeline.transform(X), y

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
        exclude: Optional[
            list["Identity"] | list[tuple["Identity", "Identity"]]
        ] = None,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        if exclude is not None:
            with warning_only():
                warnings.warn("Ignoring exclude keyword argument.")
        if try_even_subsampling and isinstance(self, AnnotatedSampleable):
            return self.even_subsample(
                feature_extractor,
                size,
                pipeline=pipeline,
                fit_pipeline=fit_pipeline,
                random_state=random_state,
                stratify_by_groups=stratify_by_groups,
                store_indices=store_indices,
                exclude_stored_indices=exclude_stored_indices,
                reset_stored_indices=reset_stored_indices,
                categories=categories,
            )

        def test_stratify(stratify: NDArray, y: Optional[NDArray]) -> NDArray | None:
            if y is None:
                counts = np.unique(stratify, return_counts=True)[1]
            else:
                counts = np.unique(list(zip(y, stratify)), axis=0, return_counts=True)[
                    1
                ]
            if counts.min() < 2:
                # The minimum number of groups for any class cannot be less than 2.
                return None
            return stratify

        if reset_stored_indices:
            self.subsample_idx = []
        X, y = self.sample(
            feature_extractor,
            pipeline=pipeline,
            fit_pipeline=fit_pipeline,
        )
        mask = np.ones(len(X), dtype=bool)
        idx = np.arange(len(X), dtype=int)
        subsample_idx = []
        if len(self.subsample_idx) > 0:
            subsample_idx = np.concatenate(self.subsample_idx)
        exclude_idx = np.isin(idx, subsample_idx)
        if categories is not None:
            if y is None:
                raise ValueError("can only subsample annotated sampleable")
            assert isinstance(categories, list)
            mask = np.isin(y, np.asarray(categories, dtype=y.dtype))
        if exclude_stored_indices:
            mask = mask & ~exclude_idx
        if isinstance(X, pd.DataFrame):
            X = X.iloc[mask]
        else:
            X = X[mask]
        if y is not None:
            y = y[mask]
        idx = idx[mask]
        if X.shape[0] == 0:
            return X, y
        groups = None
        if stratify_by_groups:
            groups = self._sample_groups()
            if groups is not None:
                groups = groups[mask]
                groups = test_stratify(groups, y)
        if stratify_by_groups and groups is not None:
            X, y, idx = split(
                X, y=y, idx=idx, size=size, random_state=random_state, stratify=groups
            )
            if store_indices and idx is not None:
                self.subsample_idx.append(idx)
            return X, y
        stratify = y
        if stratify is not None:
            stratify = test_stratify(stratify, y)
        X, y, idx = split(
            X, y=y, idx=idx, size=size, random_state=random_state, stratify=stratify
        )
        if store_indices and idx is not None:
            self.subsample_idx.append(idx)
        return X, y


class Sampleable(BaseSampleable):
    def __init__(
        self, trajectory: Trajectory, trajectory_other: Optional[Trajectory] = None
    ) -> None:
        if not trajectory.is_sorted:
            raise ValueError("trajectory is not sorted.")
        if not trajectory.is_complete:
            raise ValueError("trajectory is not complete.")
        self.trajectory = trajectory
        if trajectory_other is not None:
            if not trajectory_other.is_sorted:
                raise ValueError("trajectory_other is not sorted.")
            if not trajectory_other.is_complete:
                raise ValueError("trajectory_other is not complete.")
            if not len(trajectory_other) == len(trajectory):
                raise ValueError("trajectories have mismatched lengths.")
            if trajectory.timestamps[0] != trajectory_other.timestamps[0]:
                raise ValueError("trajectories have mismatched timestamps.")
            if trajectory.timestamps[-1] != trajectory_other.timestamps[-1]:
                raise ValueError("trajectories have mismatched timestamps.")
        self.trajectory_other = trajectory_other

    @classmethod
    def prepare_trajectory(cls, trajectory: Trajectory):
        if not trajectory.is_sorted:
            trajectory = trajectory.sort()
        if not trajectory.is_complete:
            trajectory = trajectory.interpolate()
        return trajectory

    @classmethod
    @overload
    def prepare_trajectories(
        cls, trajectory: Trajectory, trajectory_other: None
    ) -> tuple[Trajectory, None]: ...

    @classmethod
    @overload
    def prepare_trajectories(
        cls, trajectory: Trajectory, trajectory_other: Trajectory
    ) -> tuple[Trajectory, Trajectory]: ...

    @classmethod
    def prepare_trajectories(
        cls, trajectory: Trajectory, trajectory_other: Optional[Trajectory] = None
    ) -> tuple[Trajectory, Trajectory | None]:
        trajectory = cls.prepare_trajectory(trajectory)
        if trajectory_other is not None:
            trajectory_other = cls.prepare_trajectory(trajectory_other)
        else:
            return trajectory, None
        first = max(trajectory.timestamps[0], trajectory.timestamps[0])
        last = max(trajectory.timestamps[-1], trajectory.timestamps[-1])
        trajectory = trajectory.slice_window(first, last, copy=False, interpolate=False)
        trajectory_other = trajectory_other.slice_window(first, last, interpolate=False)
        return trajectory, trajectory_other

    @property
    def cfg(self) -> Config:
        return self.trajectory.cfg

    def extract_features(  # type: ignore
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
    ) -> NDArray | pd.DataFrame:
        return feature_extractor.extract(self.trajectory, self.trajectory_other)

    def annotate(
        self, annotations: pd.DataFrame, *, categories: tuple[str, ...]
    ) -> "AnnotatedSampleable":
        return AnnotatedSampleable(
            self.trajectory,
            self.trajectory_other,
            annotations=annotations,
            categories=categories,
        )

    @property
    def sampling_targets(self) -> list[Self]:
        return [self]

    @property
    def label_encoder(self) -> OneHotEncoder:
        raise ValueError("non annotated sampleable can not encode labels")


class AnnotatedSampleable(Sampleable):
    required_columns: tuple[str, ...] = (
        "start",
        "stop",
        "category",
    )

    def __init__(
        self,
        trajectory: Trajectory,
        trajectory_other: Optional[Trajectory] = None,
        *,
        annotations: pd.DataFrame,
        categories: tuple[str, ...],
    ) -> None:
        super().__init__(trajectory, trajectory_other)
        self._annotations = check_annotations(annotations, self.required_columns)
        self._categories = ("none",)
        categories = tuple(
            category for category in categories if category not in self._categories
        )
        self._categories = tuple(sorted(self._categories + categories))

    def annotate(
        self, annotations: pd.DataFrame, *, categories: tuple[str, ...]
    ) -> Self:
        return type(self)(
            self.trajectory,
            self.trajectory_other,
            annotations=annotations,
            categories=categories,
        )

    @property
    def annotations(self) -> pd.DataFrame:
        return infill_annotations(self._annotations, self.trajectory.timestamps[-1])

    @property
    def categories(self) -> tuple[str, ...]:
        if self._categories is not None:
            return self._categories
        return tuple(np.unique(self.annotations["category"]).tolist())

    def durations(
        self, category: Optional[str] = None, *, exclude: Iterable[str] = ("none",)
    ) -> NDArray:
        annotations = self.annotations.set_index("category")
        if category is not None:
            durations = annotations.loc[[category], "duration"]
        else:
            durations = annotations.drop(list(exclude), axis="index")["duration"]
        return np.asarray(durations)

    def category_idx(self, category: str) -> NDArray:
        if category not in self.categories:
            raise KeyError
        try:
            intervals = self.annotations.set_index("category").loc[[category]]
        except KeyError:
            return np.array([], dtype=int)
        idx = np.concatenate(
            [
                np.arange(interval["start"], interval["stop"] + 1)
                for _, interval in intervals.iterrows()
            ]
        )
        idx -= self.trajectory[0][self.trajectory.cfg.key_timestamp]
        return idx[(idx >= 0) & (idx < len(self.trajectory))]

    def interval_idx(self, row: int) -> NDArray:
        interval = self.annotations.iloc[row]
        idx = np.arange(interval["start"], interval["stop"] + 1)
        idx -= self.trajectory[0][self.trajectory.cfg.key_timestamp]
        return idx[(idx >= 0) & (idx < len(self.trajectory))]

    def _sample_y(self) -> NDArray:
        category_idx = [self.category_idx(category) for category in self.categories]
        y = [
            np.repeat(category, len(_category_idx))
            for category, _category_idx in zip(self.categories, category_idx)
        ]
        category_idx = np.concatenate(category_idx)
        y = np.concatenate(y)
        return y[np.argsort(category_idx)]

    def _sample_groups(self) -> NDArray | None:
        group_idx = [self.interval_idx(row) for row in range(len(self.annotations))]
        groups = [
            np.repeat(group, len(_group_idx))
            for group, _group_idx in enumerate(group_idx)
        ]
        group_idx = np.concatenate(group_idx)
        groups = np.concatenate(groups)
        return groups[np.argsort(group_idx)]

    @overload
    def even_subsample(
        self, feature_extractor: FeatureExtractor, *args, **kwargs
    ) -> tuple[NDArray, NDArray]: ...

    @overload
    def even_subsample(
        self, feature_extractor: DataFrameFeatureExtractor, *args, **kwargs
    ) -> tuple[pd.DataFrame, NDArray]: ...

    def even_subsample(
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
    ) -> tuple[NDArray | pd.DataFrame, NDArray]:
        if reset_stored_indices:
            self.subsample_idx = []
        if categories is None:
            categories = list(self.categories)
        if isinstance(size, float) and size > 1.0:
            size = 1.0
        category_counts = [self.category_idx(category).size for category in categories]
        num_available_categories = len(
            [count for count in category_counts if count > 0]
        )
        if isinstance(size, float):
            size_per_category = int(
                size * sum(category_counts) / num_available_categories
            )
        elif isinstance(size, int):
            size_per_category = int(size / num_available_categories)
        else:
            raise ValueError("size should be int > 0 or float in the range (0.0, 1.0)")
        X = []
        y = []
        for category in categories:
            _X, _y = self.subsample(
                feature_extractor,
                size_per_category,
                pipeline=pipeline,
                fit_pipeline=fit_pipeline,
                random_state=random_state,
                stratify_by_groups=stratify_by_groups,
                store_indices=store_indices,
                exclude_stored_indices=exclude_stored_indices,
                categories=[category],
                try_even_subsampling=False,
            )
            X.append(_X)
            y.append(_y)
        X = type(feature_extractor).concatenate(*X, axis=0)
        y = np.concatenate(y, axis=0)
        return X, y

    @property
    def label_encoder(self) -> OneHotEncoder:
        if self._label_encoder is None:
            categories = np.asarray(self.categories).reshape(-1, 1)
            self._label_encoder = OneHotEncoder(sparse_output=False).fit(categories)
        return self._label_encoder
