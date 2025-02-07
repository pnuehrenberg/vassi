from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder

from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ..utils import Identifier

if TYPE_CHECKING:
    from loguru import Logger

    from ._sampleable import AnnotatedSampleable, Sampleable


class BaseDataset:
    _label_encoder: OneHotEncoder | None

    def __init__(self) -> None:
        self._label_encoder = None

    @overload
    def sample(
        self,
        feature_extractor: FeatureExtractor,
        *,
        exclude: Optional[Iterable[Identifier]] = None,
        log: Optional[Logger],
    ) -> tuple[NDArray, NDArray | None]: ...

    @overload
    def sample(
        self,
        feature_extractor: DataFrameFeatureExtractor,
        *,
        exclude: Optional[Iterable[Identifier]] = None,
        log: Optional[Logger],
    ) -> tuple[pd.DataFrame, NDArray | None]: ...

    def sample(
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        exclude: Optional[Iterable[Identifier]] = None,
        log: Optional[Logger],
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        raise NotImplementedError

    @overload
    def subsample(
        self,
        feature_extractor: FeatureExtractor,
        size: int | float,
        *,
        random_state: Optional[np.random.Generator | int] = None,
        stratify_by_groups: bool = True,
        store_indices: bool = False,
        exclude_stored_indices: bool = False,
        reset_stored_indices: bool = False,
        categories: Optional[list[str]] = None,
        try_even_subsampling: bool = True,
        exclude: Optional[list[Identifier]] = None,
        log: Optional[Logger],
    ) -> tuple[NDArray, NDArray | None]: ...

    @overload
    def subsample(
        self,
        feature_extractor: DataFrameFeatureExtractor,
        size: int | float,
        *,
        random_state: Optional[np.random.Generator | int] = None,
        stratify_by_groups: bool = True,
        store_indices: bool = False,
        exclude_stored_indices: bool = False,
        reset_stored_indices: bool = False,
        categories: Optional[list[str]] = None,
        try_even_subsampling: bool = True,
        exclude: Optional[list[Identifier]] = None,
        log: Optional[Logger],
    ) -> tuple[pd.DataFrame, NDArray | None]: ...

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
        exclude: Optional[list[Identifier]] = None,
        log: Optional[Logger],
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        raise NotImplementedError

    @property
    def sampling_targets(self) -> list["Sampleable"] | list["AnnotatedSampleable"]:
        raise NotImplementedError

    @property
    def label_encoder(self) -> OneHotEncoder:
        raise NotImplementedError

    def encode(self, y: NDArray, *, one_hot: bool = False) -> NDArray[np.integer]:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_encoded = self.label_encoder.transform(y)
        assert isinstance(y_encoded, np.ndarray)
        if one_hot:
            return y_encoded
        return np.argmax(y_encoded, axis=1)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(str(category) for category in self.label_encoder.categories_[0])
