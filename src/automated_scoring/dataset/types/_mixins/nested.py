from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Self,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ....features import DataFrameFeatureExtractor, FeatureExtractor
from ...utils import (
    Identifier,
    IndividualIdentifier,
    SubjectIdentifier,
)
from .annotated import AnnotatedMixin

if TYPE_CHECKING:
    from .sampleable import SampleableMixin


class NestedSampleableMixin(ABC):
    _target: Literal["individual", "dyad"]
    _sampleables: dict[Identifier, "SampleableMixin"]
    _iter_current: int

    def _size(self) -> int:
        return sum(len(sampleable) for sampleable in self._sampleables.values())

    @property
    def target(self) -> Literal["individual", "dyad"]:
        return self._target

    @property
    def identifiers(self) -> tuple[Identifier, ...]:
        return self._get_identifiers()

    @property
    @abstractmethod
    def individuals(
        self,
    ) -> tuple[IndividualIdentifier, ...] | tuple[SubjectIdentifier, ...]: ...

    @abstractmethod
    def _get_identifiers(self) -> tuple[Identifier, ...]: ...

    def _sample_y(self, *, exclude: Optional[Sequence[Identifier]]) -> NDArray:
        if not isinstance(self, AnnotatedMixin):
            raise ValueError("can only sample y for annotated sampleables")
        if exclude is None:
            exclude = ()
        y = []
        for identifier, sampleable in self:
            if identifier in exclude:
                continue
            if TYPE_CHECKING:
                assert isinstance(sampleable, AnnotatedMixin)
            y.append(sampleable._sample_y(exclude=exclude))
        return np.concatenate(y)

    def __iter__(self) -> Self:
        self._iter_current = 0
        return self

    def __next__(self) -> tuple[Identifier, SampleableMixin]:
        if self._iter_current >= len(self.identifiers):
            raise StopIteration
        identifier = self.identifiers[self._iter_current]
        self._iter_current += 1
        return identifier, self.select(identifier)

    def _sample_X(
        self,
        extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        exclude: Optional[Sequence[Identifier]],
    ) -> pd.DataFrame | NDArray:
        if exclude is None:
            exclude = ()
        X = []
        for identifier, sampleable in self:
            if identifier in exclude:
                continue
            X.append(sampleable.sample_X(extractor, exclude=exclude))
        if isinstance(extractor, DataFrameFeatureExtractor):
            return pd.concat(X, axis=0, ignore_index=True)
        return np.concatenate(X, axis=0)

    def _get_available_indices(
        self,
        *,
        reset_previous_indices: bool,
        exclude_previous_indices: bool,
        exclude: Optional[Sequence[Identifier]],
    ) -> tuple[NDArray, NDArray | None, Sequence[NDArray | None], dict]:
        def update_splits(splits: dict, offset: int) -> dict:
            if all([key in splits for key in ["min", "max", "offset"]]):
                splits["offset"] += offset
                return splits
            for key, _splits in splits.items():
                if TYPE_CHECKING:
                    assert isinstance(_splits, dict)
                splits[key] = update_splits(_splits, offset)
            return splits

        if exclude is None:
            exclude = ()
        indices: list[NDArray] = []
        y: list[NDArray] = []
        stratification_levels: list[list[NDArray | None]] = []
        splits = []
        for identifier, sampleable in self:
            if identifier in exclude:
                continue
            (
                indices_sampleable,
                y_sampleable,
                stratification_levels_sampleable,
                splits_sampleable,
            ) = sampleable._get_available_indices(
                reset_previous_indices=reset_previous_indices,
                exclude_previous_indices=exclude_previous_indices,
                exclude=exclude,
            )
            indices.append(indices_sampleable)
            if y_sampleable is not None:
                y.append(y_sampleable)
            stratification_levels_sampleable = [
                *stratification_levels_sampleable,
                np.repeat(
                    0, len(indices_sampleable)
                ),  # add new stratification level for each hierarchical level of nesting
            ]
            stratification_levels.append(stratification_levels_sampleable)
            splits.append(splits_sampleable)
        splits_updated = {}
        if (
            len(
                set(
                    [
                        len(stratification_levels_sampleable)
                        for stratification_levels_sampleable in stratification_levels
                    ]
                )
            )
            != 1
        ):
            raise ValueError(
                "stratification levels must be the same for all sampleables"
            )
        num_stratification_levels = len(stratification_levels[0])
        for idx, identifier in enumerate(self.identifiers):
            offset = 0
            if idx > 0:
                offset = int(max(indices[idx - 1])) + 1
            indices[idx] = indices[idx] + offset
            splits_updated[identifier] = update_splits(splits[idx], offset)
            for stratification_level_idx in range(num_stratification_levels):
                stratification_offset = 0
                current_stratification = stratification_levels[idx][
                    stratification_level_idx
                ]
                if current_stratification is None:
                    continue
                if idx > 0:
                    previous_stratification = stratification_levels[idx - 1][
                        stratification_level_idx
                    ]
                    if previous_stratification is None:
                        raise ValueError(
                            "stratification levels must be provided for either all or none of the sampleables"
                        )
                    stratification_offset = int(max(previous_stratification)) + 1
                current_stratification += stratification_offset
                stratification_levels[idx][stratification_level_idx] = (
                    current_stratification
                )
        stratification_levels_concatenated = []
        for stratification_level_idx in range(num_stratification_levels):
            if any(
                [
                    stratification_levels[idx][stratification_level_idx] is None
                    for idx in range(len(stratification_levels))
                ]
            ):
                continue
            concatenated_stratification: list[NDArray] = []
            for idx in range(len(stratification_levels)):
                stratification = stratification_levels[idx][stratification_level_idx]
                if TYPE_CHECKING:
                    assert isinstance(stratification, np.ndarray)
                concatenated_stratification.append(stratification)
            stratification_levels_concatenated.append(
                np.concatenate(concatenated_stratification)
            )
        return (
            np.concatenate(indices),
            np.concatenate(y) if len(y) == len(indices) else None,
            stratification_levels_concatenated,
            splits_updated,
        )

    def _select_samples(
        self,
        extractor: FeatureExtractor | DataFrameFeatureExtractor,
        indices: NDArray,
        splits: Optional[dict],
        *,
        store_indices: bool,
        exclude: Optional[Sequence[Identifier]],
    ) -> tuple[pd.DataFrame | NDArray, NDArray | None]:
        if splits is None:
            raise ValueError("splits must be specified for nested sampleables")
        if exclude is None:
            exclude = ()
        X, y = [], []
        num_features = None
        for identifier, sampleable in self:
            if identifier in exclude:
                continue
            _X, _y = sampleable._select_samples(
                extractor,
                indices,
                splits[identifier],
                store_indices=store_indices,
                exclude=exclude,
            )
            if num_features is None and _X.shape[1] != 0:
                num_features = _X.shape[1]
            X.append(_X)
            y.append(_y)
        if isinstance(extractor, DataFrameFeatureExtractor):
            X = pd.concat(X, axis=0, ignore_index=True)
        else:
            X = [(_X.reshape(0, num_features) if _X.size == 0 else _X) for _X in X]
            X = np.concatenate(X)
        if any([_y is None for _y in y]):
            y = None
        else:
            y = np.concatenate(y)
        return X, y

    def select(self, identifier: Identifier) -> "SampleableMixin":
        return self._sampleables[identifier]
