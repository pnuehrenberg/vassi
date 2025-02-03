from itertools import permutations
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Self, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder

from ...data_structures import Trajectory
from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ..observations.utils import check_observations
from ..utils import Identifier, IndividualIdentifier, get_actor
from ._dataset_base import BaseDataset
from ._sampleable import AnnotatedSampleable, Sampleable
from .dyad import Dyad
from .individual import Individual
from .utils import (
    get_concatenated_dataset,
    recursive_sampleables,
)


class Group(BaseDataset):
    def __init__(
        self,
        trajectories: dict[IndividualIdentifier, Trajectory],
        *,
        target: Literal["individuals", "dyads"],
        exclude: Optional[Iterable[Identifier]] = None,
    ) -> None:
        super().__init__()
        self.exclude = [] if exclude is None else list(exclude)
        if len(trajectories) == 0:
            raise ValueError("provide at least one trajectory.")
        self.trajectories = trajectories
        self._sampling_target = target
        self._sampleables: dict[Identifier, Sampleable] = {}
        if target == "individuals":
            for identity in self.individuals:
                if identity in self.exclude:
                    continue
                self._sampleables[identity] = Individual(self.trajectories[identity])
        elif target == "dyads":
            for identifier in self.potential_identifieres:
                if TYPE_CHECKING:
                    assert isinstance(identifier, tuple)
                actor, recipient = identifier
                if (actor, recipient) in self.exclude:
                    continue
                trajectory, trajectory_other = Sampleable.prepare_trajectories(
                    self.trajectories[actor],
                    self.trajectories[recipient],
                )
                self._sampleables[(actor, recipient)] = Dyad(
                    trajectory=trajectory,
                    trajectory_other=trajectory_other,
                )
        else:
            raise ValueError(
                "sampling target should be one of 'individuals' or 'dyads'"
            )

    @property
    def target(self) -> Literal["individuals", "dyads"]:
        assert (
            self._sampling_target == "individuals" or self._sampling_target == "dyads"
        )
        return self._sampling_target

    @property
    def individuals(self) -> tuple[IndividualIdentifier, ...]:
        return tuple(sorted(self.trajectories.keys()))

    def annotate(
        self, observations: pd.DataFrame, *, categories: tuple[str, ...]
    ) -> "AnnotatedGroup":
        return AnnotatedGroup(
            self.trajectories,
            target=self.target,
            observations=observations,
            categories=categories,
            exclude=self.exclude,
        )

    @overload
    def sample(
        self,
        feature_extractor: FeatureExtractor,
        *args,
        **kwargs,
    ) -> tuple[NDArray, NDArray | None]: ...

    @overload
    def sample(
        self,
        feature_extractor: DataFrameFeatureExtractor,
        *args,
        **kwargs,
    ) -> tuple[pd.DataFrame, NDArray | None]: ...

    def sample(
        self,
        feature_extractor: FeatureExtractor | DataFrameFeatureExtractor,
        *,
        exclude: Optional[Iterable[Identifier]] = None,
        show_progress: bool = True,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        return get_concatenated_dataset(
            recursive_sampleables(self, exclude=exclude),
            feature_extractor,
            sampling_type="sample",
            show_progress=show_progress,
        )

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
        random_state: Optional[np.random.Generator | int] = None,
        stratify_by_groups: bool = True,
        store_indices: bool = False,
        exclude_stored_indices: bool = False,
        reset_stored_indices: bool = False,
        categories: Optional[list[str]] = None,
        try_even_subsampling: bool = True,
        exclude: Optional[Iterable[Identifier]] = None,
        show_progress: bool = True,
    ) -> tuple[NDArray | pd.DataFrame, NDArray | None]:
        if exclude is None:
            exclude = []
        return get_concatenated_dataset(
            recursive_sampleables(self, exclude=exclude),
            feature_extractor,
            size=size,
            random_state=random_state,
            stratify_by_groups=stratify_by_groups,
            store_indices=store_indices,
            exclude_stored_indices=exclude_stored_indices,
            reset_stored_indices=reset_stored_indices,
            categories=categories,
            try_even_subsampling=try_even_subsampling,
            sampling_type="subsample",
            show_progress=show_progress,
        )

    @property
    def sampling_targets(self) -> list[Sampleable]:
        return list(self._sampleables.values())

    @property
    def potential_identifieres(self) -> list[Identifier]:
        if self.target == "individuals":
            identifiers = self.individuals
        else:
            identifiers = list(permutations(self.individuals, 2))
        return sorted(identifiers)

    @property
    def identifiers(self) -> list[Identifier]:
        # use for select
        return sorted(list(self._sampleables.keys()))

    def select(self, key: Identifier) -> Sampleable:
        return self._sampleables[key]

    @property
    def label_encoder(self) -> OneHotEncoder:
        raise ValueError("non annotated group can not encode labels")

    def _copy(self, *, exclude: Iterable[Identifier]) -> Self:
        return type(self)(
            self.trajectories,
            target=self.target,
            exclude=exclude,
        )

    def get_subgroup(
        self,
        selected_individuals: Iterable[IndividualIdentifier],
        *,
        exclude: Iterable[Identifier],
    ) -> Self:
        remaining = [
            sampleable_id
            for sampleable_id in self.potential_identifieres
            if (sampleable_id in exclude)
            or (get_actor(sampleable_id) not in selected_individuals)
        ]
        return self._copy(exclude=remaining)


class AnnotatedGroup(Group):
    def __init__(
        self,
        trajectories: dict[IndividualIdentifier, Trajectory],
        *,
        target: Literal["individuals", "dyads"],
        exclude: Optional[Iterable[Identifier]] = None,
        observations: pd.DataFrame,
        categories: tuple[str, ...],
    ) -> None:
        self._sampleables: dict[  # type: ignore
            Identifier, AnnotatedSampleable
        ] = {}
        super().__init__(trajectories, target=target, exclude=exclude)
        required_columns = [*AnnotatedSampleable.required_columns, "actor"]
        if self.target == "dyads":
            required_columns.append("recipient")
        observations = check_observations(
            observations,
            required_columns=required_columns,
            allow_overlapping=True,
            allow_unsorted=True,
        )
        if "group" in observations.columns:
            observations = observations.drop(columns=["group"])
        self._observations = observations
        self._categories: tuple[str, ...] = tuple()
        for key, sampleable in self._sampleables.items():
            if self.target == "dyads":
                assert isinstance(key, tuple)
                observations = self.get_observations(*key)
            else:
                assert isinstance(key, str | int)
                observations = self.get_observations(key, None)
            annotated_sampleable = self._sampleables[key].annotate(
                observations, categories=categories
            )
            self._sampleables[key] = annotated_sampleable
            if len(self._categories) == 0:
                self._categories = annotated_sampleable.categories

    def _copy(self, *, exclude: Iterable[Identifier]) -> Self:
        return type(self)(
            self.trajectories,
            target=self.target,
            observations=self._observations,
            categories=self._categories,
            exclude=exclude,
        )

    @property
    def observations(self) -> pd.DataFrame:
        return self._observations.copy()

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories

    def get_observations(
        self, actor: IndividualIdentifier, recipient: Optional[IndividualIdentifier]
    ) -> pd.DataFrame:
        if self.target == "dyads" and recipient is None:
            raise ValueError("provide recipient for sampling target 'dyads'")
        elif self.target == "dyads":
            observations = self.observations.set_index(["actor", "recipient"])
            try:
                observations = observations.loc[[(actor, recipient)]]
            except KeyError:
                observations = observations.iloc[:0]
        else:
            observations = self.observations.set_index(["actor"])
            try:
                observations = observations.loc[[actor]]
            except KeyError:
                observations = observations.iloc[:0]
        return observations.reset_index(drop=True).sort_values("start")

    @property
    def label_encoder(self) -> OneHotEncoder:
        if self._label_encoder is None:
            categories = np.asarray(self.categories).reshape(-1, 1)
            self._label_encoder = OneHotEncoder(sparse_output=False).fit(categories)
        return self._label_encoder
