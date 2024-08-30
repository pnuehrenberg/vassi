from typing import Iterable, Optional, Self, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from ...config import Config
from ...data_structures import Trajectory
from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ..annotations import check_annotations, infill_annotations
from ..sampling import SampleKMeansKwargs, sample_k_means


class BaseSampleable:
    X: Optional[NDArray | pd.DataFrame] = None
    X_subsampled: Optional[NDArray | pd.DataFrame] = None
    y: Optional[NDArray] = None
    y_subsampled: Optional[NDArray] = None

    @property
    def dataset(self) -> tuple[Optional[NDArray | pd.DataFrame], Optional[NDArray]]:
        return self.X, self.y

    @property
    def subsampled_dataset(
        self,
    ) -> tuple[Optional[NDArray | pd.DataFrame], Optional[NDArray]]:
        return self.X_subsampled, self.y_subsampled

    @overload
    def extract_features(self, feature_extractor: FeatureExtractor) -> NDArray: ...

    @overload
    def extract_features(
        self,
        feature_extractor: DataFrameFeatureExtractor,
    ) -> pd.DataFrame: ...

    def extract_features(
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
    ) -> NDArray | pd.DataFrame:
        raise NotImplementedError

    def sample(
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        pipeline: Optional[Pipeline] = None,
        fit_pipeline: bool = True,
    ) -> Self:
        raise NotImplementedError

    def subsample(
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        pipeline: Optional[Pipeline] = None,
        fit_pipeline: bool = True,
        sample_k_means_kwargs: Optional[SampleKMeansKwargs],
    ) -> Self:
        raise NotImplementedError


class Sampleable(BaseSampleable):
    trajectory: Trajectory
    trajectory_other: Optional[Trajectory]

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
        self.trajectory_other = trajectory_other

    @property
    def cfg(self) -> Config:
        return self.trajectory.cfg

    def extract_features(  # type: ignore
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
    ) -> NDArray | pd.DataFrame:
        return feature_extractor.extract(self.trajectory, self.trajectory_other)

    def sample(  # type: ignore
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        pipeline: Optional[Pipeline] = None,
        fit_pipeline: bool = True,
    ) -> Self:
        X = self.extract_features(feature_extractor)
        if pipeline is None:
            self.X = X
            return self
        if isinstance(feature_extractor, DataFrameFeatureExtractor):
            pipeline.set_output(transform="pandas")
        if fit_pipeline:
            X: NDArray | pd.DataFrame = pipeline.fit_transform(X)
        else:
            X: NDArray | pd.DataFrame = pipeline.transform(X)
        self.X = X
        return self

    def subsample(  # type: ignore
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        reset: bool = True,
        pipeline: Optional[Pipeline] = None,
        fit_pipeline: bool = True,
        sample_k_means_kwargs: Optional[SampleKMeansKwargs],
    ) -> Self:
        if reset or self.X is None:
            self.sample(
                feature_extractor,
                pipeline=pipeline,
                fit_pipeline=fit_pipeline,
            )
        if self.X is None:
            raise AssertionError
        self.X_subsampled = sample_k_means(self.X, sample_k_means_kwargs)
        return self

    def annotate(self, annotations: pd.DataFrame) -> "AnnotatedSampleable":
        return AnnotatedSampleable(
            self.trajectory,
            self.trajectory_other,
            annotations=annotations,
        )


class AnnotatedSampleable(Sampleable):
    required_columns: tuple[str, ...] = (
        "start",
        "stop",
        "category",
    )
    _annotations: pd.DataFrame
    _categories: tuple[str, ...] = ("none",)

    def __init__(
        self,
        trajectory: Trajectory,
        trajectory_other: Optional[Trajectory] = None,
        *,
        annotations: pd.DataFrame,
        categories: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__(trajectory, trajectory_other)
        self._annotations = check_annotations(annotations, self.required_columns)
        if categories is not None:
            self._categories += tuple(
                [
                    category
                    for category in categories
                    if category not in self._categories
                ]
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
        idx = idx - self.trajectory[0][self.trajectory.cfg.key_timestamp]
        return idx[(idx >= 0) & (idx < len(self.trajectory))]

    def sample(  # type: ignore
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        pipeline: Optional[Pipeline] = None,
        fit_pipeline: bool = True,
    ) -> Self:
        super().sample(feature_extractor, pipeline=pipeline, fit_pipeline=fit_pipeline)
        category_idx = [self.category_idx(category) for category in self.categories]
        y = [
            np.repeat(category, len(_category_idx))
            for category, _category_idx in zip(self.categories, category_idx)
        ]
        category_idx = np.concatenate(category_idx)
        y = np.concatenate(y)
        self.y = y[np.argsort(category_idx)]
        return self

    def subsample(  # type: ignore
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        reset: bool = True,
        pipeline: Optional[Pipeline] = None,
        fit_pipeline: bool = True,
        sample_k_means_kwargs: Optional[SampleKMeansKwargs],
    ) -> Self:
        if reset or self.X is None:
            self.sample(
                feature_extractor,
                pipeline=pipeline,
                fit_pipeline=fit_pipeline,
            )
        if self.X is None:
            raise AssertionError
        category_subsamples = []
        y_subsampled = []
        category_samples = None
        for category in self.categories:
            category_idx = self.category_idx(category)
            if len(category_idx) == 0:
                continue
            if isinstance(self.X, pd.DataFrame):
                category_samples = pd.DataFrame(self.X.iloc[category_idx])
            else:
                category_samples = self.X[category_idx]
            _category_subsamples = sample_k_means(
                category_samples, sample_k_means_kwargs
            )
            category_subsamples.append(_category_subsamples)
            y_subsampled.append(np.repeat(category, len(_category_subsamples)))
        if category_samples is None:
            raise ValueError("Empty annotations, no subsampling possible.")
        if isinstance(category_samples, pd.DataFrame):
            self.X_subsampled = pd.concat(category_subsamples, axis=0).reset_index(
                drop=True
            )
        else:
            self.X_subsampled = np.concatenate(category_subsamples, axis=0)
        self.y_subsampled = np.concatenate(y_subsampled)
        return self
