from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..._selection import select_indices
from .annotated import AnnotatedMixin

if TYPE_CHECKING:
    from loguru import Logger

    from ....features import BaseExtractor, F
    from .annotated_sampleable import AnnotatedSampleableMixin


class SamplingFunction(Protocol):
    def __call__(
        self,
        sampleable: "SampleableMixin",
        extractor: BaseExtractor[F],
        *args: ...,
        random_state: Optional[np.random.Generator | int],
        log: Optional[Logger],
        **kwargs: ...,
    ) -> tuple[F, NDArray]: ...


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
        extractor: BaseExtractor[F],
    ) -> F: ...

    def sample_X(
        self,
        extractor: BaseExtractor[F],
    ) -> F:
        return self._sample_X(extractor)

    def sample(
        self,
        extractor: BaseExtractor[F],
    ) -> tuple[F, NDArray | None]:
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
        extractor: BaseExtractor[F],
        indices: NDArray,
        splits: Optional[dict],
        *,
        store_indices: bool,
    ) -> tuple[F, NDArray | None]: ...

    def subsample(
        self,
        extractor: BaseExtractor[F],
        size: int | float | Mapping[str | tuple[str, ...], int | float],
        *,
        random_state: Optional[int | np.random.Generator],
        stratify: bool,
        reset_previous_indices: bool,
        exclude_previous_indices: bool,
        store_indices: bool,
        log: Optional["Logger"],
    ) -> tuple[F, NDArray | None]:
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
