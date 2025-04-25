from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Optional, Protocol

import loguru
import numpy as np
import pandas as pd

from ..._selection import select_indices
from .annotated import AnnotatedMixin

if TYPE_CHECKING:
    from ....features import BaseExtractor, Shaped
    from .annotated_sampleable import AnnotatedSampleableMixin


class SamplingFunction(Protocol):
    """
    Protocol for functions that sample data from a sampleable.

    Parameters:
        sampleable (:class:`~automated_scoring.dataset.types.mixins.sampleable.SampleableMixin`): The sampleable to sample from.
        extractor (:class:`~automated_scoring.features.feature_extractor.BaseExtractor`): The extractor to use for sampling.
        *args: Additional arguments to use within the function.
        random_state (:class:`~numpy.random.Generator` | int | None): The random state to use for sampling.
        log (loguru.Logger | None): The logger to use for logging.
        **kwargs: Additional keyword arguments to use within the function.
    """

    def __call__[F: Shaped](
        self,
        sampleable: "SampleableMixin",
        extractor: BaseExtractor[F],
        *args: ...,
        random_state: Optional[np.random.Generator | int],
        log: Optional[loguru.Logger],
        **kwargs: ...,
    ) -> tuple[F, np.ndarray]: ...


class SampleableMixin(ABC):
    """
    Mixin for sampleable objects.
    """

    @classmethod
    @abstractmethod
    def REQUIRED_COLUMNS(
        cls, target: Optional[Literal["individual", "dyad"]] = None
    ) -> tuple[str, ...]: ...

    @abstractmethod
    def _size(self) -> int: ...

    @property
    def size(self) -> int:
        """
        Return the total number of available samples.

        Also accessible via the :code:`len` function.
        """
        return self._size()

    def __len__(self) -> int:
        return self.size

    @abstractmethod
    def _sample_X[F: Shaped](
        self,
        extractor: BaseExtractor[F],
    ) -> F: ...

    def sample_X[F: Shaped](
        self,
        extractor: BaseExtractor[F],
    ) -> F:
        """
        Extract features for all available samples.

        Args:
            extractor: The extractor to use for sampling.

        Returns:
            The extracted features.
        """
        return self._sample_X(extractor)

    def sample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
    ) -> tuple[F, np.ndarray | None]:
        """
        Extract features and labels (if also :class:`~automated_scoring.dataset.types.mixins.annotated.AnnotatedMixin`) for all available samples.

        Args:
            extractor: The extractor to use for sampling.

        Returns:
            The extracted features and labels.
        """
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
    ) -> tuple[np.ndarray, np.ndarray | None, Sequence[np.ndarray | None], dict]: ...

    @abstractmethod
    def _select_samples[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        indices: np.ndarray,
        splits: Optional[dict],
        *,
        store_indices: bool,
    ) -> tuple[F, np.ndarray | None]: ...

    def subsample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        size: int | float | Mapping[str | tuple[str, ...], int | float],
        *,
        random_state: Optional[int | np.random.Generator] = None,
        stratify: bool = True,
        reset_previous_indices: bool = False,
        exclude_previous_indices: bool = False,
        store_indices: bool = False,
        log: Optional[loguru.Logger] = None,
    ) -> tuple[F, np.ndarray | None]:
        """
        Extract features and labels (if also :class:`~automated_scoring.dataset.types.mixins.annotated.AnnotatedMixin`) for a subset of samples.

        Args:
            extractor: The extractor to use for sampling.
            size: The number of samples to extract.
            random_state: The random state to use for sampling.
            stratify: Whether to stratify the samples.
            stratification_levels: The stratification levels to use.
            reset_previous_indices: Whether to reset the previous indices.
            exclude_previous_indices: Whether to exclude the previous indices.
            store_indices: Whether to store the indices.
            log: The logger to use for logging.

        Returns:
            The extracted features and labels.
        """
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
    ) -> "AnnotatedSampleableMixin":
        """
        Annotate the sampleable with observations.

        Parameters:
            observations: The observations to annotate the sampleable with.
            categories: The categories of the observations.
            background_category: The background category of the observations.

        Returns:
            The annotated sampleable.
        """
        ...
