# import warnings
from collections.abc import Iterable
from typing import Optional, Self, overload

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder

from ...config import Config
from ...data_structures import Trajectory
from ...features import DataFrameFeatureExtractor, FeatureExtractor

# from ...utils import warning_only
from ..observations.utils import check_observations, infill_observations
from ..sampling.split import split, test_stratify
from ..utils import Identifier
from ._dataset_base import BaseDataset


class BaseSampleable(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self._sampled_idx: list[NDArray] = []

    def get_sampled_idx(self, reset: bool) -> NDArray:
        if reset:
            self._sampled_idx = []
        if len(self._sampled_idx) == 0:
            return np.array([])
        return np.concatenate(self._sampled_idx)

    def store_sampled_idx(self, idx: NDArray):
        self._sampled_idx.append(idx)

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
        exclude: Optional[Iterable[Identifier]] = None,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        if exclude is not None:
            logger.warning("Ignoring exclude keyword argument.")
        X = self.extract_features(feature_extractor)
        y = self._sample_y()
        return X, y

    @overload
    def subsample(
        self, feature_extractor: FeatureExtractor, *args, **kwargs
    ) -> tuple[NDArray, NDArray | None]: ...

    @overload
    def subsample(
        self, feature_extractor: DataFrameFeatureExtractor, *args, **kwargs
    ) -> tuple[pd.DataFrame, NDArray | None]: ...

    def _stratified_subsample(
        self,
        X: pd.DataFrame | NDArray,
        y: Optional[NDArray],
        idx: Optional[NDArray],
        size: int | float,
        stratify: Optional[NDArray],
        random_state: Optional[np.random.Generator | int],
        store_indices: bool,
    ):
        X, y, idx = split(
            X,
            y=y,
            idx=idx,
            size=size,
            stratify=stratify,
            random_state=random_state,
        )
        if store_indices and idx is not None:
            self._sampled_idx.append(idx)
        return X, y

    def subsample(
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        size: int | float,
        *,
        random_state: Optional[np.random.Generator | int] = None,
        stratify_by_groups: bool = True,
        store_indices: bool = False,
        exclude_stored_indices: bool = False,
        reset_stored_indices: bool = False,
        categories: Optional[list[str]] = None,
        try_even_subsampling: bool = True,
        exclude: Optional[Iterable[Identifier]] = None,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        if exclude is not None:
            logger.warning("Ignoring exclude keyword argument.")
        if try_even_subsampling and isinstance(self, AnnotatedSampleable):
            return self.even_subsample(
                feature_extractor,
                size,
                random_state=random_state,
                stratify_by_groups=stratify_by_groups,
                store_indices=store_indices,
                exclude_stored_indices=exclude_stored_indices,
                reset_stored_indices=reset_stored_indices,
                categories=categories,
            )
        X, y = self.sample(feature_extractor)
        mask = np.ones(len(X), dtype=bool)
        idx = np.arange(len(X), dtype=int)
        exclude_idx = np.isin(idx, self.get_sampled_idx(reset_stored_indices))
        if categories is not None:
            if y is None:
                raise ValueError("can only subsample annotated sampleable")
            mask = np.isin(y, np.asarray(list(categories), dtype=y.dtype))
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
                groups = test_stratify(len(X), size, groups, y)
        if stratify_by_groups and groups is not None:
            return self._stratified_subsample(
                X, y, idx, size, groups, random_state, store_indices
            )
        stratify = y
        if stratify is not None:
            stratify = test_stratify(len(X), size, stratify, y)
        return self._stratified_subsample(
            X, y, idx, size, stratify, random_state, store_indices
        )


class Sampleable(BaseSampleable):
    def __init__(
        self, trajectory: Trajectory, trajectory_other: Optional[Trajectory] = None
    ) -> None:
        super().__init__()
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
            if trajectory.timestep != trajectory_other.timestep:
                raise ValueError("trajectories have unequal timesteps.")
            if trajectory.timestamps[0] != trajectory_other.timestamps[0]:
                raise ValueError("trajectories have mismatched timestamps.")
            if trajectory.timestamps[-1] != trajectory_other.timestamps[-1]:
                raise ValueError("trajectories have mismatched timestamps.")
        self.trajectory_other = trajectory_other

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
        if trajectory_other is None:
            return trajectory, None
        first = max(trajectory.timestamps[0], trajectory_other.timestamps[0])
        last = min(trajectory.timestamps[-1], trajectory_other.timestamps[-1])
        trajectory = trajectory.slice_window(first, last, copy=False, interpolate=False)
        trajectory_other = trajectory_other.slice_window(
            first, last, copy=False, interpolate=False
        )
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
        self, observations: pd.DataFrame, *, categories: tuple[str, ...]
    ) -> "AnnotatedSampleable":
        return AnnotatedSampleable(
            self.trajectory,
            self.trajectory_other,
            observations=observations,
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
        observations: pd.DataFrame,
        categories: tuple[str, ...],
    ) -> None:
        super().__init__(trajectory, trajectory_other)
        self._observations = check_observations(observations, self.required_columns)
        self._categories = ("none",)
        categories = tuple(
            category for category in categories if category not in self._categories
        )
        self._categories = tuple(sorted(self._categories + categories))

    def annotate(
        self, observations: pd.DataFrame, *, categories: tuple[str, ...]
    ) -> Self:
        return type(self)(
            self.trajectory,
            self.trajectory_other,
            observations=observations,
            categories=categories,
        )

    @property
    def observations(self) -> pd.DataFrame:
        return infill_observations(self._observations, self.trajectory.timestamps[-1])

    @property
    def categories(self) -> tuple[str, ...]:
        if self._categories is not None:
            return self._categories
        return tuple(np.unique(self.observations["category"]).tolist())

    def durations(
        self, category: Optional[str] = None, *, exclude: Iterable[str] = ("none",)
    ) -> NDArray:
        observations = self.observations.set_index("category")
        if category is not None:
            durations = observations.loc[[category], "duration"]
        else:
            durations = observations.drop(list(exclude), axis="index")["duration"]
        return np.asarray(durations)

    def category_idx(self, category: str) -> NDArray:
        if category not in self.categories:
            raise KeyError
        try:
            intervals = self.observations.set_index("category").loc[[category]]
        except KeyError:
            return np.array([], dtype=int)
        idx = np.concatenate(
            [
                np.arange(interval["start"], interval["stop"] + 1)
                for _, interval in intervals.iterrows()
            ]
        )
        idx -= self.trajectory.timestamps[0]
        return idx[(idx >= 0) & (idx < self.trajectory.get_interpolated_length())]

    def interval_idx(self, row: int) -> NDArray:
        interval = self.observations.iloc[row]
        idx = np.arange(interval["start"], interval["stop"] + 1)
        idx -= self.trajectory.timestamps[0]
        return idx[(idx >= 0) & (idx < self.trajectory.get_interpolated_length())]

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
        group_idx = [self.interval_idx(row) for row in range(len(self.observations))]
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
        random_state: Optional[np.random.Generator | int] = None,
        stratify_by_groups: bool = True,
        store_indices: bool = False,
        exclude_stored_indices: bool = False,
        reset_stored_indices: bool = False,
        categories: Optional[list[str]] = None,
    ) -> tuple[NDArray | pd.DataFrame, NDArray]:
        if reset_stored_indices:
            self._sampled_idx = []
        if categories is None:
            categories = list(self.categories)
        if isinstance(size, float) and size > 1.0:
            size = 1.0
        category_counts = [self.category_idx(category).size for category in categories]
        num_available_categories = len(
            [count for count in category_counts if count > 0]
        )
        if num_available_categories == 0:
            if isinstance(feature_extractor, DataFrameFeatureExtractor):
                X = pd.DataFrame()
            else:
                X = np.array([]).reshape(0, len(feature_extractor.feature_names))
            y = np.array([])
            return X, y
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
            self._label_encoder = OneHotEncoder(
                categories=[list(self.categories)],  # type: ignore (is not typed completely)
                sparse_output=False,
            ).fit(categories)
        return self._label_encoder
