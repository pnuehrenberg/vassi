from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ....features import DataFrameFeatureExtractor, FeatureExtractor
from .annotated import AnnotatedMixin
from .sampleable import SampleableMixin

if TYPE_CHECKING:
    from loguru import Logger


class AnnotatedSampleableMixin(SampleableMixin, AnnotatedMixin):
    if TYPE_CHECKING:

        @overload
        def sample(
            self,
            extractor: FeatureExtractor,
        ) -> tuple[NDArray, NDArray]: ...

        @overload
        def sample(
            self,
            extractor: DataFrameFeatureExtractor,
        ) -> tuple[pd.DataFrame, NDArray]: ...

        def sample(
            self,
            extractor: FeatureExtractor | DataFrameFeatureExtractor,
        ) -> tuple[pd.DataFrame | NDArray, NDArray]: ...

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
        ) -> tuple[NDArray, NDArray]: ...

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
        ) -> tuple[pd.DataFrame, NDArray]: ...

        def subsample(
            self,
            extractor: FeatureExtractor | DataFrameFeatureExtractor,
            size: int | float | Mapping[str | tuple[str, ...], int | float],
            *,
            random_state: int | np.random.Generator | None,
            stratify: bool,
            reset_previous_indices: bool,
            exclude_previous_indices: bool,
            store_indices: bool,
            log: Optional[Logger],
        ) -> tuple[pd.DataFrame | NDArray, NDArray]: ...
