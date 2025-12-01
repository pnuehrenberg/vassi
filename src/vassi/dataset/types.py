from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from itertools import permutations
from pathlib import Path
from typing import (
    Literal,
    Self,
    overload,
    override,
)

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,  # pyright: ignore[reportUnknownVariableType]
)

from .._utils import check_memory_for_array
from ..data_structures import Trajectory
from ..features import BaseExtractor, Shaped
from ..io.h5 import load_trajectories
from ..type_guards import is_tuple_of
from ..utils import to_int_seed
from ..warnings import warn
from .observations import densify_observations
from .stratification import biased_stratified_subsample
from .utils import (
    check_trajectory,
    get_num_samples,
    prepare_paired_trajectories,
)


def _stratified_subsample[F: Shaped](
    element: Base,
    extractor: BaseExtractor[F],
    *,
    size: int | float,
    min_samples_per_stratum: int,
    random_state: int | np.random.Generator | None,
    reset: bool,
    exclude_previously_sampled: bool,
    store_indices: bool,
    ensure_sampling_at: np.ndarray | None,
    out: np.ndarray | None,
) -> tuple[F, np.ndarray | None]:
    random_state = np.random.default_rng(random_state)
    available_indices, interval_levels, _ = element.describe_available_samples(
        reset=reset,
        exclude_previously_sampled=exclude_previously_sampled,
    )
    num_available_samples = len(available_indices)
    num_samples = get_num_samples(size, num_available_samples)
    if ensure_sampling_at is not None:
        num_samples -= len(ensure_sampling_at)
        if num_samples < 0:
            raise ValueError(
                f"Cannot ensure sampling of {len(ensure_sampling_at)} indices from {num_available_samples} available indices with size={size}"
            )
        mask = ~np.isin(available_indices, ensure_sampling_at)
        available_indices = available_indices[mask]
        interval_levels = interval_levels[mask]
    _, strata_labels = np.unique(interval_levels, axis=0, return_inverse=True)
    subsampled_indices = available_indices[
        biased_stratified_subsample(
            strata_labels,
            size=num_samples,
            min_samples_per_stratum=min_samples_per_stratum,
            random_state=random_state,
        )
    ]
    if ensure_sampling_at is not None:
        subsampled_indices = np.concatenate([subsampled_indices, ensure_sampling_at])
    return element.sample(
        extractor,
        indices=np.sort(subsampled_indices),
        store_indices=store_indices,
        out=out,
    )


def _stratified_subsample_by_categories[F: Shaped](
    element: WithAnnotations,
    extractor: BaseExtractor[F],
    *,
    size: Mapping[str | Iterable[str], int | float],
    min_samples_per_stratum: int,
    random_state: int | np.random.Generator | None,
    reset: bool,
    exclude_previously_sampled: bool,
    store_indices: bool,
    ensure_sampling_at: np.ndarray | None,
    out: np.ndarray | None,
) -> tuple[F, np.ndarray]:
    random_state = np.random.default_rng(random_state)
    # validate input
    subsampling_categories: set[str] = set()
    for condition in size:
        if isinstance(condition, str):
            if condition in subsampling_categories:
                raise ValueError(f"Duplicate category: {condition}")
            subsampling_categories.add(condition)
            continue
        condition = set(condition)
        if (
            len(duplicate_categories := condition.intersection(subsampling_categories))
            > 0
        ):
            raise ValueError(f"Duplicate categories: {duplicate_categories}")
        subsampling_categories |= condition
    undefined_categories = subsampling_categories - element.categories
    if len(undefined_categories) > 0:
        raise ValueError(
            f"Cannot subsample undefined categories: {undefined_categories}"
        )
    if not isinstance(element, Base):
        raise TypeError(
            f"invalid use of {type(element)}, this mixin should only be used in subclasses of {Base}"
        )
    available_indices, interval_levels, _ = element.describe_available_samples(
        reset=reset,
        exclude_previously_sampled=exclude_previously_sampled,
    )
    _y = element.sample_y(indices=None)
    if ensure_sampling_at is not None:
        mask = ~np.isin(available_indices, ensure_sampling_at)
        categories = sorted(element.categories)
        for idx, count in zip(*np.unique(_y[~mask], return_counts=True)):
            if categories[idx] not in subsampling_categories:
                raise ValueError(
                    f"Cannot ensure sampling of {count} indices for category {categories[idx]} when category is not included in size"
                )
        available_indices = available_indices[mask]
        interval_levels = interval_levels[mask]
    y = _y[available_indices]
    subsampled_indices: list[np.ndarray] = []
    for _condition, subsampling_size in size.items():
        if isinstance(_condition, str):
            condition = [_condition]
        else:
            condition = list(_condition)
        num_available_samples = np.isin(_y, element.encode(np.asarray(condition))).sum()
        try:
            num_samples = get_num_samples(subsampling_size, num_available_samples)
        except ValueError as e:
            raise ValueError(
                f"Invalid subsampling size for condition: {_condition}"
            ) from e
        if ensure_sampling_at is not None:
            num_ensured_for_condition: int = np.isin(
                _y[ensure_sampling_at], element.encode(np.asarray(condition))
            ).sum()
            num_samples -= num_ensured_for_condition
            if num_samples < 0:
                raise ValueError(
                    f"Cannot ensure sampling of {num_ensured_for_condition} indices for {_condition} from {num_available_samples} available indices with size={subsampling_size}"
                )
        mask = np.isin(y, element.encode(np.asarray(condition)))
        _, strata_labels = np.unique(
            np.concatenate([y[mask].reshape(-1, 1), interval_levels[mask]], axis=1),
            axis=0,
            return_inverse=True,
        )
        _subsampled_indices = available_indices[mask][
            biased_stratified_subsample(
                strata_labels,
                size=num_samples,
                min_samples_per_stratum=min_samples_per_stratum,
                random_state=random_state,
            )
        ]
        subsampled_indices.append(_subsampled_indices)
    if ensure_sampling_at is not None:
        subsampled_indices.append(ensure_sampling_at)
    return element.sample(
        extractor,
        indices=np.sort(np.concatenate(subsampled_indices)),
        store_indices=store_indices,
        out=out,
    )


class WithCategories:
    _categories: set[str] | None = None
    _background_category: str | None = None

    def with_categories(self, categories: set[str], *, background_category: str):
        if self._categories is not None and categories != self._categories:
            raise ValueError(
                f"Categories are already set to {self._categories}, cannot change to {categories}"
            )
        if (
            self._background_category is not None
            and background_category != self._background_category
        ):
            raise ValueError(
                f"Background category is already set to {self._background_category}, cannot change to {background_category}"
            )
        self._categories = categories
        self._background_category = background_category

    @property
    def categories(self) -> frozenset[str]:
        if self._categories is None:
            raise ValueError(
                f"Categories are not set, call with_categories on {type(self)}"
            )
        return frozenset(self._categories)

    @property
    def background_category(self) -> str:
        if self._background_category is None:
            raise ValueError(
                f"Background category is not set, call with_categories on {type(self)}"
            )
        return self._background_category

    @property
    def foreground_categories(self) -> frozenset[str]:
        if self._categories is None:
            raise ValueError(
                f"Categories are not set, call with_categories on {type(self)}"
            )
        return self.categories - {self.background_category}

    def encode(self, labels: np.ndarray) -> np.ndarray:
        assert labels.ndim == 1
        categories = sorted(self.categories)
        assert np.isin(labels, categories + [self.background_category]).all()
        labels_encoded = np.full(len(labels), -1, dtype=int)
        idx, label_idx = np.where(
            labels[:, np.newaxis] == np.asarray(categories)[np.newaxis]
        )
        labels_encoded[idx] = label_idx
        return labels_encoded


class Base(ABC):
    def __init__(self):
        self.previously_sampled_indices: list[np.ndarray] = []

    @property
    @abstractmethod
    def num_samples(self) -> int: ...

    @abstractmethod
    def describe_available_samples(
        self,
        *,
        reset: bool,
        exclude_previously_sampled: bool,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        # Returns a tuple of available_indices, interval_levels, num_intervals_per_level
        ...

    @abstractmethod
    def sample_X[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        out: np.ndarray | None,
    ) -> F: ...

    @abstractmethod
    def sample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        store_indices: bool,
        out: np.ndarray | None,
    ) -> tuple[F, np.ndarray | None]: ...

    @abstractmethod
    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: set[str],
        background_category: str,
    ) -> WithAnnotations: ...

    def subsample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        size: int | float,
        min_samples_per_stratum: int,
        random_state: int | np.random.Generator | None,
        reset: bool,
        exclude_previously_sampled: bool,
        store_indices: bool,
        ensure_sampling_at: np.ndarray | None,
        out: np.ndarray | None,
    ) -> tuple[F, np.ndarray | None]:
        return _stratified_subsample(
            self,
            extractor,
            size=size,
            min_samples_per_stratum=min_samples_per_stratum,
            random_state=random_state,
            reset=reset,
            exclude_previously_sampled=exclude_previously_sampled,
            store_indices=store_indices,
            ensure_sampling_at=ensure_sampling_at,
            out=out,
        )


class RequiresBase:
    _base: tuple[Base, type[Base]] | None = None

    @property
    def base(self) -> tuple[Base, type[Base]]:
        if self._base is not None:
            return self._base
        invalid_type_message = f"invalid use of {type(self)}, this mixin class should only be used in subclasses of {Base}"
        if not isinstance(self, Base):
            raise TypeError(invalid_type_message)
        base = None
        for _base in type(self).__bases__:
            if not issubclass(_base, Base):
                continue
            base = _base
            break
        if base is None:
            raise TypeError(invalid_type_message)
        self._base = self, base
        return self._base


class WithAnnotations(RequiresBase, WithCategories, ABC):
    _columns: frozenset[str] | None = None
    _conditional_columns: dict[str, dict[int | str, set[str]]] | None = None
    _observations: pd.DataFrame | None = None
    _y: np.ndarray | None = None

    def __init_subclass__(
        cls,
        columns: set[str],
        conditional_columns: dict[str, dict[int | str, set[str]]] | None = None,
    ):
        cls._columns = frozenset(columns)
        cls._conditional_columns = conditional_columns

    def with_annotations(
        self,
        observations: pd.DataFrame,
        *,
        categories: set[str],
        background_category: str,
        prepare_annotations: bool,
    ) -> None:
        self.with_categories(categories, background_category=background_category)
        self._y = None
        required_conditional_columns: set[str] = set()
        if self._conditional_columns is not None:
            for attr, conditions in self._conditional_columns.items():
                value = getattr(self, attr, None)
                for condition, columns in conditions.items():
                    if value != condition:
                        continue
                    required_conditional_columns |= columns
        self._columns = self.columns | required_conditional_columns
        if not prepare_annotations:
            self._observations = None
            return
        self._observations = self._prepare_observations(observations)

    @property
    def columns(self) -> frozenset[str]:
        if self._columns is None:
            raise ValueError(
                f"Columns are not set, call with_annotations on {type(self)}"
            )
        return self._columns

    @property
    def conditional_columns(self) -> dict[str, dict[int | str, set[str]]] | None:
        return self._conditional_columns

    @abstractmethod
    def _prepare_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        missing_columns = self.columns - set(observations.columns)
        if len(missing_columns) > 0:
            raise ValueError(f"Observations are missing columns: {missing_columns}")
        return observations[sorted(self.columns)]

    @abstractmethod
    def sample_y(
        self,
        *,
        indices: np.ndarray | None,
    ) -> np.ndarray: ...

    def sample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        store_indices: bool,
        out: np.ndarray | None,
    ) -> tuple[F, np.ndarray]:
        y = self.sample_y(indices=indices)
        self, base_type = self.base
        x, _ = (
            base_type.sample(  # TODO: _ may already hold y, in which case it is sampled twice!
                self, extractor, indices=indices, store_indices=store_indices, out=out
            )
        )
        return (x, y)

    @property
    def observations(self) -> pd.DataFrame:
        if self._observations is None:
            raise ValueError(
                f"Observations are not set, call with_annotations on {type(self)}"
            )
        return self._observations

    @property
    def category_counts(self) -> dict[str, int]:
        """Counts of each category in the sampleable."""
        counts: dict[str, int] = {}
        for category in sorted(self.categories):
            mask = self.observations["category"] == category
            if not mask.any():
                counts[category] = 0
                continue
            counts[category] = int(
                np.sum(
                    1
                    + np.asarray(self.observations.loc[mask, "stop"])
                    - np.asarray(self.observations.loc[mask, "start"])
                )
            )
        return counts

    def subsample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        size: int | float | Mapping[str | Iterable[str], int | float],
        min_samples_per_stratum: int,
        random_state: int | np.random.Generator | None,
        reset: bool,
        exclude_previously_sampled: bool,
        store_indices: bool,
        ensure_sampling_at: np.ndarray | None,
        out: np.ndarray | None,
    ) -> tuple[F, np.ndarray]:
        if isinstance(size, int | float):
            self, _ = self.base
            x, y = _stratified_subsample(
                self,
                extractor,
                size=size,
                min_samples_per_stratum=min_samples_per_stratum,
                random_state=random_state,
                reset=reset,
                exclude_previously_sampled=exclude_previously_sampled,
                store_indices=store_indices,
                ensure_sampling_at=ensure_sampling_at,
                out=out,
            )
            assert isinstance(y, np.ndarray), (
                "Expected y to be a numpy array for an annotated dataset type"
            )
            return x, y
        return _stratified_subsample_by_categories(
            self,
            extractor,
            size=size,
            min_samples_per_stratum=min_samples_per_stratum,
            random_state=random_state,
            reset=reset,
            exclude_previously_sampled=exclude_previously_sampled,
            store_indices=store_indices,
            ensure_sampling_at=ensure_sampling_at,
            out=out,
        )


class ElementMixin(RequiresBase):
    # this mixin class provides common functionality for elemental dataset types
    # (e.g., individual and dyad) and should be used in combination with some subclass of Base

    def describe_available_samples(
        self, *, reset: bool, exclude_previously_sampled: bool
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        self, _ = self.base
        num_samples = self.num_samples
        indices = np.arange(num_samples)
        if reset:
            self.previously_sampled_indices.clear()
        if exclude_previously_sampled and len(self.previously_sampled_indices) > 0:
            indices = np.setdiff1d(
                indices, np.concatenate(self.previously_sampled_indices)
            )
        return indices, np.zeros((len(indices), 1), dtype=int), (1,)

    def sample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        store_indices: bool,
        out: np.ndarray | None,
    ) -> tuple[F, np.ndarray | None]:
        self, _ = self.base
        if store_indices:
            if indices is None:
                indices = np.arange(self.num_samples)
            self.previously_sampled_indices.append(indices)
        return self.sample_X(extractor, indices=indices, out=out), None


class AnnotatedElementMixin(
    WithAnnotations, ABC, columns={"start", "stop", "category"}
):
    @override
    def sample_y[F: Shaped](self, *, indices: np.ndarray | None) -> np.ndarray:
        if self._y is None:
            observations = self.observations
            start = np.asarray(observations["start"])
            stop = np.asarray(observations["stop"])
            category = self.encode(np.asarray(observations["category"]))
            self._y: np.ndarray | None = np.repeat(category, stop - start + 1)
        if indices is not None:
            return self._y[indices]
        return self._y

    def describe_available_samples(
        self, *, reset: bool, exclude_previously_sampled: bool
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        observations = self.observations
        num_intervals = len(observations)
        start = np.asarray(observations["start"])
        stop = np.asarray(observations["stop"])
        self, base_type = self.base
        available_indices, _, _ = base_type.describe_available_samples(
            self, reset=reset, exclude_previously_sampled=exclude_previously_sampled
        )  # interval_levels, num_intervals_per_level are np.zeros and (1, ) and will be replaced below
        intervals = np.repeat(np.arange(num_intervals), stop - start + 1)[
            available_indices
        ]
        return available_indices, intervals.reshape(-1, 1), (num_intervals,)


class Individual(ElementMixin, Base):
    @overload
    def __init__(self, trajectory: Trajectory) -> None: ...

    @overload
    def __init__(self, trajectory: None = None, *, individual: Individual) -> None: ...

    def __init__(
        self,
        trajectory: Trajectory | None = None,
        *,
        individual: Individual | None = None,
    ):
        super().__init__()
        if trajectory is not None:
            trajectory = check_trajectory(trajectory)
        elif individual is not None:
            trajectory = individual.trajectory
        else:
            raise ValueError("Either trajectory or individual must be provided")
        self.trajectory: Trajectory = trajectory

    @property
    @override
    def num_samples(self) -> int:
        return len(self.trajectory)

    @override
    def sample_X[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        out: np.ndarray | None,
    ) -> F:
        return extractor.extract(self.trajectory, indices=indices, out=out)

    @override
    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: set[str],
        background_category: str,
    ) -> AnnotatedIndividual:
        return AnnotatedIndividual(
            individual=self,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )


class AnnotatedIndividual(
    AnnotatedElementMixin, Individual, columns={"category", "start", "stop"}
):
    @overload
    def __init__(
        self,
        trajectory: Trajectory,
        *,
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectory: None = None,
        *,
        individual: Individual,
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ) -> None: ...

    def __init__(
        self,
        trajectory: Trajectory | None = None,
        *,
        individual: Individual | None = None,
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ):
        if trajectory is not None:
            super().__init__(trajectory)
        elif individual is not None:
            super().__init__(individual=individual)
        else:
            raise ValueError("Either trajectory or individual must be provided")
        self.with_annotations(
            observations,
            categories=categories,
            background_category=background_category,
            prepare_annotations=True,
        )

    @override
    def _prepare_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        # this cannot be moved to AnnotatedElementMixin, because it depends on the trajectory
        observations = super()._prepare_observations(observations)
        observations = densify_observations(
            observations,
            time_range=tuple(self.trajectory.timestamps[[0, -1]]),
            background_category=self.background_category,
        )
        # the below only finishes successfully if categories are valid
        _ = self.encode(np.asarray(observations["category"]))
        return observations


class Dyad(ElementMixin, Base):
    @overload
    def __init__(
        self,
        trajectory: Trajectory,
        trajectory_other: Trajectory,
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectory: None = None,
        trajectory_other: None = None,
        *,
        dyad: Dyad,
    ) -> None: ...

    def __init__(
        self,
        trajectory: Trajectory | None = None,
        trajectory_other: Trajectory | None = None,
        *,
        dyad: Dyad | None = None,
    ):
        super().__init__()
        if trajectory is not None and trajectory_other is not None:
            trajectory, trajectory_other = prepare_paired_trajectories(
                trajectory, trajectory_other
            )
            self.trajectory: Trajectory = trajectory
            self.trajectory_other: Trajectory = trajectory_other
        elif dyad is not None:
            self.trajectory = dyad.trajectory
            self.trajectory_other = dyad.trajectory_other
        else:
            raise ValueError(
                "Either trajectory and trajectory_other or dyad must be provided"
            )

    @property
    @override
    def num_samples(self) -> int:
        return len(self.trajectory)

    @override
    def sample_X[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        out: np.ndarray | None,
    ) -> F:
        return extractor.extract(
            self.trajectory,
            self.trajectory_other,
            indices=indices,
            out=out,
        )

    @override
    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: set[str],
        background_category: str,
    ) -> AnnotatedDyad:
        time_range = tuple(self.trajectory.timestamps[[0, -1]])
        observations = densify_observations(
            observations,
            time_range=time_range,
            background_category=background_category,
        )
        return AnnotatedDyad(
            dyad=self,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )


class AnnotatedDyad(AnnotatedElementMixin, Dyad, columns={"category", "start", "stop"}):
    @overload
    def __init__(
        self,
        trajectory: Trajectory,
        trajectory_other: Trajectory,
        *,
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectory: None = None,
        trajectory_other: None = None,
        *,
        dyad: Dyad,
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ) -> None: ...

    def __init__(
        self,
        trajectory: Trajectory | None = None,
        trajectory_other: Trajectory | None = None,
        *,
        dyad: Dyad | None = None,
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ):
        if trajectory is not None and trajectory_other is not None:
            super().__init__(trajectory, trajectory_other)
        elif dyad is not None:
            super().__init__(dyad=dyad)
        else:
            raise ValueError(
                "Either trajectory and trajectory_other or individual must be provided"
            )
        self.with_annotations(
            observations,
            categories=categories,
            background_category=background_category,
            prepare_annotations=True,
        )

    @override
    def _prepare_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        # same as for AnnotatedIndividual
        observations = super()._prepare_observations(observations)
        observations = densify_observations(
            observations,
            time_range=tuple(self.trajectory.timestamps[[0, -1]]),
            background_category=self.background_category,
        )

        # the below only finishes successfully if categories are valid
        _ = self.encode(np.asarray(observations["category"]))
        return observations


def _split_indices(
    elements: Sequence[Base],
    indices: np.ndarray | None,
) -> list[None] | list[np.ndarray]:
    if indices is None:
        element_indices: list[None] | list[np.ndarray] = [
            None for _ in range(len(elements))
        ]
    else:
        bounds = [0] + np.cumsum([element.num_samples for element in elements]).tolist()
        element_indices = []
        for lower, upper in zip(bounds[:-1], bounds[1:]):
            element_indices.append(
                indices[(indices >= lower) & (indices < upper)] - lower
            )
    return element_indices


class ElementCollection[I: Hashable, E: Base](Base, ABC):
    elements: dict[I, E] | None = None
    identifier_validator: Callable[..., I]
    element_validator: Callable[..., E]

    def __init__(
        self,
        elements: Mapping[I, E],
        *,
        identifier_validator: Callable[[object], I],
        element_validator: Callable[[object], E],
    ):
        super().__init__()
        self.identifier_validator = identifier_validator
        self.element_validator = element_validator
        self.elements = {
            self.identifier_validator(identifier): self.element_validator(element)
            for identifier, element in sorted(elements.items())
        }

    def __getitem__(self, identifier: ...) -> E:
        if self.elements is None:
            raise ValueError(f"{type(self)} is not initialized")
        identifier = self.identifier_validator(identifier)
        return self.elements[identifier]

    def __setitem__(self, identifier: ..., element: ...) -> None:
        if self.elements is None:
            raise ValueError(f"{type(self)} is not initialized")
        identifier = self.identifier_validator(identifier)
        element = self.element_validator(element)
        self.elements[identifier] = element

    def __iter__(self) -> Iterator[tuple[I, E]]:
        if self.elements is None:
            raise ValueError(f"{type(self)} is not initialized")
        return iter(self.elements.items())

    def __len__(self) -> int:
        if self.elements is None:
            raise ValueError(f"{type(self)} is not initialized")
        return len(self.elements)

    def identifiers(self) -> list[I]:
        if self.elements is None:
            raise ValueError(f"{type(self)} is not initialized")
        return list(self.elements.keys())

    @property
    @override
    def num_samples(self) -> int:
        return sum(element.num_samples for _, element in self)

    @override
    def describe_available_samples(
        self, *, reset: bool, exclude_previously_sampled: bool
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        available_indices: list[np.ndarray] = []
        interval_levels: list[np.ndarray] = []
        offset_per_level = None
        index_offset = 0
        for idx, (_, element) in enumerate(self):
            (
                _available_indices,
                _interval_levels,
                _num_intervals_per_level,
            ) = element.describe_available_samples(
                reset=reset, exclude_previously_sampled=exclude_previously_sampled
            )
            _available_indices += index_offset
            if offset_per_level is None:
                offset_per_level = np.zeros(_interval_levels.shape[1], dtype=int)
            _interval_levels += offset_per_level
            _interval_levels = np.concatenate(
                [
                    np.repeat(idx, _interval_levels.shape[0]).reshape(-1, 1),
                    _interval_levels,
                ],
                axis=1,
            )
            available_indices.append(_available_indices)
            interval_levels.append(_interval_levels)
            index_offset += element.num_samples
            offset_per_level += _num_intervals_per_level
        if offset_per_level is None:
            # empty
            return (
                np.array([], dtype=int),
                np.array([], dtype=int).reshape((0, 1)),
                (0,),
            )
        num_intervals_per_level = (len(self), *offset_per_level.tolist())
        return (
            np.concatenate(available_indices, axis=0),
            np.concatenate(interval_levels, axis=0),
            num_intervals_per_level,
        )

    @override
    def sample_X[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        out: np.ndarray | None,
    ) -> F:
        if indices is not None:
            if indices.ndim != 1:
                raise ValueError("indices must be a 1D array")
            elif np.isdtype(indices.dtype, "bool"):
                indices = np.argwhere(indices).ravel()
            elif not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("indices must be an integer array")
            num_samples = len(indices)
        else:
            num_samples = self.num_samples
        if out is None:
            shape = (num_samples, extractor.num_features)
            check_memory_for_array(shape)
            out = np.zeros(shape)
        elif out.shape != (num_samples, extractor.num_features):
            raise ValueError(
                f"Output array has incorrect shape, expected {(num_samples, extractor.num_features)} and not {out.shape}"
            )
        element_indices = _split_indices([element for _, element in self], indices)
        sample_idx = 0
        for _indices, (_, element) in zip(element_indices, self):
            if _indices is not None and _indices.size == 0:
                continue
            if _indices is None:
                num_element_samples = element.num_samples
            else:
                num_element_samples = len(_indices)
            _ = element.sample_X(
                extractor,
                indices=_indices,
                out=out[sample_idx : sample_idx + num_element_samples],
            )
            sample_idx += num_element_samples
        assert sample_idx == num_samples
        return extractor.finalize(out)

    @override
    def sample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        store_indices: bool,
        out: np.ndarray | None,
    ) -> tuple[F, np.ndarray | None]:
        if store_indices:
            if indices is None:
                indices = np.arange(self.num_samples)
            self.previously_sampled_indices.append(indices)
        return self.sample_X(extractor, indices=indices, out=out), None


def _value_validator[VT](value_type: type[VT], value: ...) -> VT:
    if not isinstance(value, value_type):
        raise ValueError(f"value must be of type {value_type} according to first item")
    return value


def _key_validator[KT: Hashable](key_type: type[KT], key: ...) -> KT:
    if not isinstance(key, key_type):
        raise ValueError(f"key must be of type {key_type} according to first item")
    return key


def _tuple_key_validator[KT: Hashable](key_type: type[KT], key: ...) -> tuple[KT, KT]:
    if not is_tuple_of(key, key_type):
        raise ValueError(f"key must be of type {key_type} according to first item")
    return key


def get_validators[KT: Hashable, VT, K, V](
    elements: Mapping[K, V],
    *,
    minimal_value_type: type[VT],
) -> tuple[Callable[[object], Hashable], Callable[[object], VT]]:
    if len(elements) == 0:
        raise ValueError("elements must not be empty")
    key, value = next(iter(elements.items()))
    if not isinstance(value, minimal_value_type):
        raise ValueError(f"element values must be instances of {minimal_value_type}")
    value_validator = partial(_value_validator, type(value))
    if not isinstance(key, tuple):
        if not isinstance(key, Hashable):
            raise ValueError(f"key must be instance of {Hashable}")
        key_validator = partial(_key_validator, type(key))
        return key_validator, value_validator
    key_item = key[0]  # pyright: ignore[reportUnknownVariableType]
    if not isinstance(key_item, Hashable):
        raise ValueError(f"key must be a tuple of {Hashable}")
    key_validator = partial(_tuple_key_validator, type(key_item))
    return key_validator, value_validator


def actor_if_dyad(identifier: Hashable) -> Hashable:
    if is_tuple_of(identifier, Hashable):
        return identifier[0]
    return identifier


class Group(ElementCollection[Hashable, Individual | Dyad]):
    @overload
    def __init__(
        self,
        trajectories: Mapping[Hashable, Trajectory],
        *,
        target: Literal["individual", "dyad"],
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectories: None = None,
        *,
        group: Group,
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectories: None = None,
        *,
        elements: Mapping[Hashable, Individual | Dyad],
        target: Literal["individual", "dyad"],
    ) -> None: ...

    def __init__(
        self,
        trajectories: Mapping[Hashable, Trajectory] | None = None,
        *,
        group: Group | None = None,
        elements: Mapping[Hashable, Individual | Dyad] | None = None,
        target: Literal["individual", "dyad"] | None = None,
    ):
        if trajectories is not None and target == "individual":
            elements = {
                identifier: Individual(trajectory)
                for identifier, trajectory in sorted(trajectories.items())
            }
            identifier_validator, element_validator = get_validators(
                elements,
                minimal_value_type=Individual,
            )
        elif trajectories is not None and target == "dyad":
            elements = {
                (identifier, identifier_other): Dyad(trajectory, trajectory_other)
                for (
                    (identifier, trajectory),
                    (identifier_other, trajectory_other),
                ) in permutations(sorted(trajectories.items()), 2)
            }
            identifier_validator, element_validator = get_validators(
                elements,
                minimal_value_type=Dyad,
            )
        elif group is not None:
            if group.elements is None:
                elements = {}
            else:
                elements = {
                    identifier: element
                    for identifier, element in group.elements.items()
                }
            identifier_validator = group.identifier_validator
            element_validator = group.element_validator
        elif elements is not None and target is not None:
            identifier_validator, element_validator = get_validators(
                elements,
                minimal_value_type=Individual if target == "individual" else Dyad,
            )
        else:
            raise ValueError(
                "Provide either group, or initialize from elements (individuals or dyads) or trajectories and a valid target"
            )

        super().__init__(
            elements,
            identifier_validator=identifier_validator,
            element_validator=element_validator,
        )

    def get_target(self) -> Literal["individual", "dyad"]:
        if self.elements is None:
            raise ValueError(f"{type(self)} is not initialized")
        _, element = next(iter(self.elements.items()))
        return "individual" if isinstance(element, Individual) else "dyad"

    def actors(self) -> set[Hashable]:
        return set([actor_if_dyad(identifier) for identifier, _ in self])

    @override
    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: set[str],
        background_category: str,
    ) -> AnnotatedGroup:
        return AnnotatedGroup(
            group=self,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )

    def select(self, identifiers: set[Hashable]) -> Self:
        return type(self)(
            elements={identifier: self[identifier] for identifier in identifiers},
            target=self.get_target(),
        )


class AnnotatedGroup(
    WithAnnotations,
    Group,
    columns={"start", "stop", "category", "actor"},
    conditional_columns={"_target": {"dyad": {"recipient"}}},
):
    @overload
    def __init__(
        self,
        trajectories: Mapping[Hashable, Trajectory],
        *,
        target: Literal["individual", "dyad"],
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectories: None = None,
        *,
        group: Group,
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectories: None = None,
        *,
        elements: Mapping[Hashable, AnnotatedIndividual | AnnotatedDyad] | None = None,
        target: Literal["individual", "dyad"] | None = None,
    ) -> None: ...

    def __init__(
        self,
        trajectories: Mapping[Hashable, Trajectory] | None = None,
        *,
        group: Group | None = None,
        elements: Mapping[Hashable, AnnotatedIndividual | AnnotatedDyad] | None = None,
        target: Literal["individual", "dyad"] | None = None,
        observations: pd.DataFrame | None = None,
        categories: set[str] | None = None,
        background_category: str | None = None,
    ):
        if trajectories is not None and target is not None:
            super().__init__(trajectories, target=target)
        elif group is not None:
            super().__init__(group=group)
        elif elements is not None and target is not None:
            categories_per_element = {
                element.categories for element in elements.values()
            }
            background_category_per_element = {
                element.background_category for element in elements.values()
            }
            if len(categories_per_element) != 1:
                raise ValueError("All elements must have the same categories")
            if len(background_category_per_element) != 1:
                raise ValueError("All elements must have the same background category")
            super().__init__(elements=elements, target=target)
            self._target: Literal["individual", "dyad"] = target
            self.with_annotations(
                pd.DataFrame(),
                categories=set(categories_per_element.pop()),
                background_category=background_category_per_element.pop(),
                prepare_annotations=False,
            )
            self._observations: pd.DataFrame | None = self._concatenate_observations()
            return
        else:
            raise ValueError("Either trajectories and target or group must be provided")
        if observations is None or categories is None or background_category is None:
            raise ValueError(
                "Observations, categories, and background category must be provided"
            )
        self._target = target if target is not None else self.get_target()
        self.with_annotations(
            observations,
            categories=categories,
            background_category=background_category,
            prepare_annotations=True,
        )

    @override
    def __getitem__(self, identifier: Hashable) -> AnnotatedIndividual | AnnotatedDyad:
        element = super().__getitem__(identifier)
        if not isinstance(element, AnnotatedIndividual | AnnotatedDyad):
            raise TypeError(
                f"Expected {AnnotatedIndividual} or {AnnotatedDyad}, got {type(element)}"
            )
        return element

    @override
    def __iter__(
        self,
    ) -> Iterator[tuple[Hashable, AnnotatedIndividual | AnnotatedDyad]]:
        for identifier, _ in super().__iter__():
            yield identifier, self[identifier]

    def _concatenate_observations(self) -> pd.DataFrame:
        identifier_columns = (
            ["actor"] if self._target == "individual" else ["actor", "recipient"]
        )
        all_observations = [element.observations for _, element in self]
        identifiers = np.repeat(
            np.asarray(self.identifiers()),
            [len(observations) for observations in all_observations],
            axis=0,
        )
        observations = pd.concat(all_observations, axis=0, ignore_index=True)
        observations[identifier_columns] = identifiers
        return observations

    @override
    def _prepare_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        if self.elements is None:
            raise ValueError(f"{type(self)} is not initialized")
        observations = super()._prepare_observations(observations)
        identifier_columns = (
            ["actor"] if self._target == "individual" else ["actor", "recipient"]
        )
        observations = observations.set_index(identifier_columns)
        for identifier, element in self.elements.items():
            try:
                observations_element = observations.loc[[identifier]]
            except KeyError:
                observations_element = observations.iloc[:0]
            try:
                self[identifier] = element.annotate(
                    observations_element,
                    categories=set(self.categories),
                    background_category=self.background_category,
                )
            except ValueError as e:
                raise ValueError(
                    f"Error annotating {type(element).__name__} {identifier}"
                ) from e
        return self._concatenate_observations()

    @override
    def sample_y[F: Shaped](self, *, indices: np.ndarray | None) -> np.ndarray:
        if self._y is None:
            self._y: np.ndarray | None = np.concatenate(
                [element.sample_y(indices=None) for _, element in self]
            )
        if indices is not None:
            return self._y[indices]
        return self._y

    @override
    def sample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        store_indices: bool,
        out: np.ndarray | None,
    ) -> tuple[F, np.ndarray]:
        X, _ = super().sample(
            extractor, indices=indices, store_indices=store_indices, out=out
        )
        y = self.sample_y(indices=indices)
        return X, y


class Dataset(ElementCollection[Hashable, Group]):
    @overload
    def __init__(
        self,
        trajectories: Mapping[Hashable, Mapping[Hashable, Trajectory]],
        *,
        target: Literal["individual", "dyad"],
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectories: None = None,
        *,
        dataset: Dataset,
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectories: None = None,
        *,
        groups: Mapping[Hashable, Group],
    ) -> None: ...

    def __init__(
        self,
        trajectories: Mapping[Hashable, Mapping[Hashable, Trajectory]] | None = None,
        *,
        dataset: Dataset | None = None,
        groups: Mapping[Hashable, Group] | None = None,
        target: Literal["individual", "dyad"] | None = None,
    ):
        if trajectories is not None and target is not None:
            elements = {
                identifier: Group(_trajectories, target=target)
                for identifier, _trajectories in sorted(trajectories.items())
            }
            identifier_validator, element_validator = get_validators(
                elements,
                minimal_value_type=Group,
            )
        elif dataset is not None:
            if dataset.elements is None:
                elements = {}
            else:
                elements = {
                    identifier: element
                    for identifier, element in dataset.elements.items()
                }
            identifier_validator = dataset.identifier_validator
            element_validator = dataset.element_validator
        elif groups is not None:
            elements = {
                identifier: group for identifier, group in sorted(groups.items())
            }
            identifier_validator, element_validator = get_validators(
                elements,
                minimal_value_type=Group,
            )
        else:
            raise ValueError(
                "Provide either dataset or groups or initialize with trajectories and a valid target"
            )
        super().__init__(
            elements,
            identifier_validator=identifier_validator,
            element_validator=element_validator,
        )

    def get_target(self) -> Literal["individual", "dyad"]:
        if self.elements is None:
            raise ValueError(f"{type(self)} is not initialized")
        _, element = next(iter(self.elements.items()))
        return element.get_target()

    @override
    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: set[str],
        background_category: str,
    ) -> AnnotatedDataset:
        return AnnotatedDataset(
            dataset=self,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )

    def _split(
        self,
        selected_actors: set[tuple[Hashable, Hashable]],
    ) -> tuple[Self, Self]:
        selected_groups: dict[Hashable, Group] = {}
        remaining_groups: dict[Hashable, Group] = {}
        for identifier, group in self:
            actors = group.actors()
            selected = {
                actor
                for (_group_identifier, actor) in selected_actors
                if _group_identifier == identifier and actor in actors
            }
            remaining = actors - selected
            if len(selected) > 0:
                selected_groups[identifier] = group.select(
                    {
                        identifier
                        for identifier, _ in group
                        if actor_if_dyad(identifier) in selected
                        # condition on recipient for not "subset_actors_only"
                    }
                )
            if len(remaining) > 0:
                remaining_groups[identifier] = group.select(
                    {
                        identifier
                        for identifier, _ in group
                        if actor_if_dyad(identifier) in remaining
                        # same as above applies here
                    }
                )
        return type(self)(groups=selected_groups), type(self)(groups=remaining_groups)

    def actors(self) -> set[tuple[Hashable, Hashable]]:
        return {
            (identifier, actor)
            for identifier, group in self
            for actor in group.actors()
        }

    def exclude(
        self,
        actors: set[Hashable | tuple[Hashable, Hashable]],
    ) -> Self:
        simple_identifiers: set[tuple[Hashable, Hashable]] = set()
        specific_identifiers: set[tuple[Hashable, Hashable]] = set()
        for identifier in actors:
            if isinstance(identifier, tuple):
                specific_identifiers.add(identifier)  # pyright: ignore[reportUnknownArgumentType]
                continue
            simple_identifiers |= {
                (group_identifier, identifier)
                for group_identifier in self.identifiers()
            }
        current_actors = self.actors()
        excluded_actors = (simple_identifiers | specific_identifiers) & current_actors
        if len(excluded_actors) == 0:
            return type(self)(groups={identifier: group for identifier, group in self})
        return self._split(current_actors - excluded_actors)[0]

    def split(
        self,
        size: int | float,
        random_state: int | np.random.Generator | None,
    ) -> tuple[Self, Self]:
        random_state = np.random.default_rng(random_state)
        if isinstance(size, float) and (size < 0 or size > 1):
            raise ValueError(
                "size should be within (0.0, 1.0) interval (exclusive) if float"
            )
        actors = sorted(self.actors())
        if isinstance(size, float):
            size = int(size * len(actors))
        if size < 1 or size > len(actors) - 1:
            raise ValueError(
                f"size should be within [1, {len(actors)}] (inclusive) if int"
            )
        try:
            selection = train_test_split(  # pyright: ignore[reportUnknownVariableType]
                actors,
                train_size=size,
                random_state=to_int_seed(random_state),
                stratify=[actor[0] for actor in actors],
            )[0]
        except ValueError:
            selection = train_test_split(  # pyright: ignore[reportUnknownVariableType]
                actors,
                train_size=size,
                random_state=to_int_seed(random_state),
            )[0]

        # the below is only for type checking purposes (sklearn is not fully typed)
        selected_actors: set[tuple[Hashable, Hashable]] = set()
        for actor in selection:  # pyright: ignore[reportUnknownVariableType]
            if not is_tuple_of(actor, Hashable):
                raise ValueError(f"Invalid actor {actor}")
            selected_actors.add(actor)

        return self._split(selected_actors)

    def k_fold(
        self,
        k: int,
        random_state: int | np.random.Generator | None,
    ) -> Iterator[tuple[Self, Self]]:
        random_state = np.random.default_rng(random_state)
        actors = sorted(self.actors())
        try:
            kf = StratifiedKFold(
                n_splits=k, shuffle=True, random_state=to_int_seed(random_state)
            )
            for selected, _ in kf.split(actors, [actor[0] for actor in actors]):  # pyright: ignore[reportUnknownMemberType]
                yield self._split(
                    {actors[idx] for idx in selected},
                )
        except ValueError:
            kf = KFold(n_splits=k, shuffle=True, random_state=to_int_seed(random_state))
            for selected, _ in kf.split(actors):  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                yield self._split(
                    {actors[idx] for idx in selected},  # pyright: ignore[reportUnknownVariableType]
                )

    @classmethod
    def load(
        cls,
        trajectory_file: Path | str,
        *,
        target: Literal["individual", "dyad"],
        **_: ...,
    ) -> Self:
        return cls(
            load_trajectories(trajectory_file),
            target=target,
        )


class AnnotatedDataset(
    WithAnnotations,
    Dataset,
    columns={"start", "stop", "category", "actor", "group"},
    conditional_columns={"_target": {"dyad": {"recipient"}}},
):
    @overload
    def __init__(
        self,
        trajectories: Mapping[Hashable, Mapping[Hashable, Trajectory]],
        *,
        target: Literal["individual", "dyad"],
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectories: None = None,
        *,
        dataset: Dataset,
        observations: pd.DataFrame,
        categories: set[str],
        background_category: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        trajectories: None = None,
        *,
        groups: Mapping[Hashable, AnnotatedGroup],
    ) -> None: ...

    def __init__(
        self,
        trajectories: Mapping[Hashable, Mapping[Hashable, Trajectory]] | None = None,
        *,
        dataset: Dataset | None = None,
        groups: Mapping[Hashable, AnnotatedGroup] | None = None,
        target: Literal["individual", "dyad"] | None = None,
        observations: pd.DataFrame | None = None,
        categories: set[str] | None = None,
        background_category: str | None = None,
    ):
        if trajectories is not None and target is not None:
            super().__init__(trajectories, target=target)
        elif dataset is not None:
            super().__init__(dataset=dataset)
        elif groups is not None:
            super().__init__(groups=groups)
            categories_per_group = {group.categories for group in groups.values()}
            background_category_per_group = {
                group.background_category for group in groups.values()
            }
            if len(categories_per_group) != 1:
                raise ValueError("All groups must have the same categories")
            if len(background_category_per_group) != 1:
                raise ValueError("All groups must have the same background category")
            self._target = super().get_target()
            self.with_annotations(
                pd.DataFrame(),
                categories=set(categories_per_group.pop()),
                background_category=background_category_per_group.pop(),
                prepare_annotations=False,
            )
            self._observations: pd.DataFrame | None = self._concatenate_observations()
            return
        else:
            raise ValueError(
                "Either trajectories and target or dataset must be provided"
            )
        if observations is None or categories is None or background_category is None:
            raise ValueError(
                "Observations, categories, and background category must be provided"
            )
        self._target: Literal["individual", "dyad"] = (
            target if target is not None else self.get_target()
        )
        self.with_annotations(
            observations,
            categories=categories,
            background_category=background_category,
            prepare_annotations=True,
        )

    @override
    def __getitem__(self, identifier: Hashable) -> AnnotatedGroup:
        element = super().__getitem__(identifier)
        if not isinstance(element, AnnotatedGroup):
            raise TypeError(f"Expected {AnnotatedGroup}, got {type(element)}")
        return element

    @override
    def __iter__(
        self,
    ) -> Iterator[tuple[Hashable, AnnotatedGroup]]:
        for identifier, _ in super().__iter__():
            yield identifier, self[identifier]

    def _concatenate_observations(self) -> pd.DataFrame:
        all_observations = [element.observations for _, element in self]
        identifiers = np.repeat(
            np.asarray(self.identifiers()),
            [len(observations) for observations in all_observations],
            axis=0,
        )
        observations = pd.concat(all_observations, axis=0, ignore_index=True)
        observations["group"] = identifiers
        return observations

    @override
    def _prepare_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        if self.elements is None:
            raise ValueError(f"{type(self)} is not initialized")
        observations = super()._prepare_observations(observations)
        observations = observations.set_index("group")
        for identifier, element in self.elements.items():
            try:
                observations_element = observations.loc[[identifier]]
            except KeyError:
                observations_element = observations.iloc[:0]
            try:
                self[identifier] = element.annotate(
                    observations_element,
                    categories=set(self.categories),
                    background_category=self.background_category,
                )
            except ValueError as e:
                raise ValueError(
                    f"Error annotating {type(element).__name__} {identifier}"
                ) from e
        return self._concatenate_observations()

    @override
    def sample_y[F: Shaped](self, *, indices: np.ndarray | None) -> np.ndarray:
        if self._y is None:
            self._y: np.ndarray | None = np.concatenate(
                [element.sample_y(indices=None) for _, element in self]
            )
        if indices is not None:
            return self._y[indices]
        return self._y

    @override
    def sample[F: Shaped](
        self,
        extractor: BaseExtractor[F],
        *,
        indices: np.ndarray | None,
        store_indices: bool,
        out: np.ndarray | None,
    ) -> tuple[F, np.ndarray]:
        X, _ = super().sample(
            extractor, indices=indices, store_indices=store_indices, out=out
        )
        y = self.sample_y(indices=indices)
        return X, y

    @override
    @classmethod
    def load(
        cls,
        trajectory_file: Path | str,
        *,
        target: Literal["individual", "dyad"],
        observation_file: Path | str,
        categories: set[str] | None = None,
        background_category: str,
        **_: ...,
    ) -> Self:
        observations = pd.read_csv(observation_file)  # pyright: ignore[reportUnknownMemberType]
        if categories is None:
            categories = set(observations["category"]) | {background_category}
            warn(
                f"Loading categories ({', '.join(sorted(categories))}) from observations file, specify categories argument if incomplete."
            )
        return cls(
            load_trajectories(trajectory_file),
            target=target,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )
