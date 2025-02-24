from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Optional, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ....features import DataFrameFeatureExtractor, FeatureExtractor
from ..._selection import select_indices
from .annotated import AnnotatedMixin

if TYPE_CHECKING:
    from loguru import Logger

    from .annotated_sampleable import AnnotatedSampleableMixin


class SampleableMixin(ABC):
    @classmethod
    @abstractmethod
    def REQUIRED_COLUMNS(cls, target=None) -> tuple[str, ...]: ...

    @abstractmethod
    def _size(self) -> int: ...

    @property
    def size(self) -> int:
        return self._size()

    def __len__(self) -> int:
        return self.size

    @abstractmethod
    def _sample_X(
        self,
        extractor: FeatureExtractor | DataFrameFeatureExtractor,
    ) -> pd.DataFrame | NDArray: ...

    @overload
    def sample_X(
        self,
        extractor: FeatureExtractor,
    ) -> NDArray: ...

    @overload
    def sample_X(
        self,
        extractor: DataFrameFeatureExtractor,
    ) -> pd.DataFrame: ...

    def sample_X(
        self,
        extractor: FeatureExtractor | DataFrameFeatureExtractor,
    ) -> pd.DataFrame | NDArray:
        return self._sample_X(extractor)

    @overload
    def sample(
        self,
        extractor: FeatureExtractor,
    ) -> tuple[NDArray, NDArray | None]: ...

    @overload
    def sample(
        self,
        extractor: DataFrameFeatureExtractor,
    ) -> tuple[pd.DataFrame, NDArray | None]: ...

    def sample(
        self,
        extractor: FeatureExtractor | DataFrameFeatureExtractor,
    ) -> tuple[pd.DataFrame | NDArray, NDArray | None]:
        X = self.sample_X(extractor)
        y = None
        if isinstance(self, AnnotatedMixin):
            y = self.sample_y()
        return X, y

    @abstractmethod
    def _get_available_indices(
        self,
        *,
        reset_previous_indices: bool,
        exclude_previous_indices: bool,
    ) -> tuple[NDArray, NDArray | None, Sequence[NDArray | None], dict]: ...

    @abstractmethod
    def _select_samples(
        self,
        extractor: FeatureExtractor | DataFrameFeatureExtractor,
        indices: NDArray,
        splits: Optional[dict],
        *,
        store_indices: bool,
    ) -> tuple[pd.DataFrame | NDArray, NDArray | None]: ...

    @overload
    def subsample(
        self,
        extractor: FeatureExtractor,
        size: int | float | Mapping[str | tuple[str, ...], int | float],
        *,
        random_state: Optional[int | np.random.Generator],
        stratify: bool,
        reset_previous_indices: bool,
        exclude_previous_indices: bool,
        store_indices: bool,
        log: Optional["Logger"],
    ) -> tuple[NDArray, NDArray | None]: ...

    @overload
    def subsample(
        self,
        extractor: DataFrameFeatureExtractor,
        size: int | float | Mapping[str | tuple[str, ...], int | float],
        *,
        random_state: Optional[int | np.random.Generator],
        stratify: bool,
        reset_previous_indices: bool,
        exclude_previous_indices: bool,
        store_indices: bool,
        log: Optional["Logger"],
    ) -> tuple[pd.DataFrame, NDArray | None]: ...

    def subsample(
        self,
        extractor: FeatureExtractor | DataFrameFeatureExtractor,
        size: int | float | Mapping[str | tuple[str, ...], int | float],
        *,
        random_state: Optional[int | np.random.Generator],
        stratify: bool,
        reset_previous_indices: bool,
        exclude_previous_indices: bool,
        store_indices: bool,
        log: Optional["Logger"],
    ) -> tuple[pd.DataFrame | NDArray, NDArray | None]:
        available_indices, y, stratification_levels, splits = (
            self._get_available_indices(
                reset_previous_indices=reset_previous_indices,
                exclude_previous_indices=exclude_previous_indices,
            )
        )
        selected_indices = select_indices(
            available_indices,
            y,
            size=size,
            random_state=random_state,
            stratify=stratify,
            stratification_levels=stratification_levels,
            categories=(self.categories if isinstance(self, AnnotatedMixin) else None),
            log=log,
        )
        return self._select_samples(
            extractor,
            selected_indices,
            splits,
            store_indices=store_indices,
        )

    @abstractmethod
    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
        background_category: str,
    ) -> "AnnotatedSampleableMixin": ...
