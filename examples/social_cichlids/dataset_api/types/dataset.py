from collections.abc import Generator, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Self,
    overload,
)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from automated_scoring.classification.results import IndividualIdentifier
from automated_scoring.dataset import (
    GroupIdentifier,
)
from automated_scoring.dataset.types.dataset import SubjectIdentifier
from automated_scoring.utils import ensure_generator, to_int_seed

from ._mixins import (
    AnnotatedMixin,
    AnnotatedSampleableMixin,
    NestedSampleableMixin,
    SampleableMixin,
)
from .group import Group


def include(
    individual: SubjectIdentifier,
    exclude: Sequence[GroupIdentifier | IndividualIdentifier | SubjectIdentifier],
) -> bool:
    exclude = list(exclude)
    if individual in exclude:
        return False
    if individual[0] in exclude:
        return False
    if individual[1] in exclude:
        return False
    return True


class Dataset(NestedSampleableMixin, SampleableMixin):
    def __init__(
        self,
        groups: Mapping[GroupIdentifier, Group],
    ):
        self._sampleables = {identifier: group for identifier, group in groups.items()}

    @overload
    @classmethod
    def REQUIRED_COLUMNS(
        cls, target: Literal["individual"]
    ) -> tuple[
        Literal["group"],
        Literal["actor"],
        Literal["category"],
        Literal["start"],
        Literal["stop"],
    ]: ...

    @overload
    @classmethod
    def REQUIRED_COLUMNS(
        cls, target: Literal["dyad"]
    ) -> tuple[
        Literal["group"],
        Literal["actor"],
        Literal["recipient"],
        Literal["category"],
        Literal["start"],
        Literal["stop"],
    ]: ...

    @classmethod
    def REQUIRED_COLUMNS(
        cls, target=None
    ) -> (
        tuple[
            Literal["group"],
            Literal["actor"],
            Literal["category"],
            Literal["start"],
            Literal["stop"],
        ]
        | tuple[
            Literal["group"],
            Literal["actor"],
            Literal["recipient"],
            Literal["category"],
            Literal["start"],
            Literal["stop"],
        ]
    ):
        if target == "individual":
            return ("group", "actor", "category", "start", "stop")
        elif target == "dyad":
            return ("group", "actor", "recipient", "category", "start", "stop")
        else:
            raise ValueError("target argument must be either 'individual' or 'dyad'")

    @property
    def individuals(self) -> tuple[SubjectIdentifier, ...]:
        individuals: list[SubjectIdentifier] = []
        for identifier in self.identifiers:
            group = self.select(identifier)
            if TYPE_CHECKING:
                assert isinstance(group, Group)
                assert isinstance(identifier, GroupIdentifier)
            individuals.extend(
                [(identifier, individual) for individual in group.individuals]
            )
        return tuple(sorted(individuals))

    def _get_identifiers(self) -> tuple[GroupIdentifier, ...]:
        identifiers = []
        for identifier in self._sampleables.keys():
            if TYPE_CHECKING:
                assert isinstance(identifier, GroupIdentifier)
            identifiers.append(identifier)
        return tuple(sorted(identifiers))

    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
        background_category: str,
    ) -> "AnnotatedDataset":
        groups: dict[GroupIdentifier, Group] = {}
        for identifier in self.identifiers:
            group = self.select(identifier)
            if TYPE_CHECKING:
                assert isinstance(identifier, GroupIdentifier)
                assert isinstance(group, Group)
            groups[identifier] = group
        return AnnotatedDataset(
            groups,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )

    def _finalize_init(self, observations: pd.DataFrame) -> None:
        if not isinstance(self, AnnotatedMixin):
            return
        observations = observations.loc[
            np.isin(observations["category"], self.categories)
        ]
        for identifier in self.identifiers:
            group = self.select(identifier)
            observations_group = observations.loc[observations["group"] == identifier]
            self._sampleables[identifier] = group.annotate(
                observations=observations_group,
                categories=self.categories,
                background_category=self.background_category,
            )

    def _get_observations(self) -> pd.DataFrame:
        observations = []
        for identifier in self.identifiers:
            group = self.select(identifier)
            if not isinstance(group, AnnotatedMixin):
                raise ValueError("unannotated groups do not have observations")
            observations_group = group._get_observations()
            observations_group["group"] = identifier
            observations.append(observations_group)
        observations = pd.concat(observations, axis=0, ignore_index=True)[
            list(self.REQUIRED_COLUMNS(self.target))
        ]
        if TYPE_CHECKING:
            assert isinstance(observations, pd.DataFrame)
        return observations

    @classmethod
    def _empty_like(cls, group: Group) -> Self:
        return cls({})

    @classmethod
    def from_groups(
        cls,
        groups: Mapping[GroupIdentifier, Group],
    ) -> Self:
        if len(groups) < 1:
            raise ValueError("groups must contain at least one group")
        new = cls._empty_like(list(groups.values())[0])
        for identifier, group in groups.items():
            new._sampleables[identifier] = groups[identifier]
        return new

    def _make_split(
        self,
        *,
        individuals_selected: tuple[SubjectIdentifier, ...],
        individuals_remaining: tuple[SubjectIdentifier, ...],
        subset_actors_only: bool,
    ) -> tuple[Self, Self]:
        groups_selected = {}
        groups_remaining = {}
        for identifier in self.identifiers:
            group = self.select(identifier)
            if TYPE_CHECKING:
                assert isinstance(group, Group)
            selected = tuple(
                individual[1]
                for individual in individuals_selected
                if individual[0] == identifier
            )
            if len(selected) > 0:
                groups_selected[identifier] = type(group).from_group(
                    group,
                    individuals=selected,
                    subset_actors_only=subset_actors_only,
                )
            remaining = tuple(
                individual[1]
                for individual in individuals_remaining
                if individual[0] == identifier
            )
            if len(remaining) > 0:
                groups_remaining[identifier] = type(group).from_group(
                    group,
                    individuals=remaining,
                    subset_actors_only=subset_actors_only,
                )
        return (
            type(self).from_groups(groups_selected),
            type(self).from_groups(groups_remaining),
        )

    def split(
        self,
        size: int | float,
        *,
        random_state: int | None | np.random.Generator,
        subset_actors_only: bool = True,
        exclude: Optional[
            Sequence[GroupIdentifier | IndividualIdentifier | SubjectIdentifier]
        ],
    ) -> tuple[Self, Self]:
        if exclude is None:
            exclude = ()
        random_state = ensure_generator(random_state)
        individuals = [
            individual
            for individual in self.individuals
            if include(individual, exclude)
        ]
        if isinstance(size, float) and (size < 0 or size > 1):
            raise ValueError(
                "size should be within (0.0, 1.0) interval (exclusive) if float"
            )
        if isinstance(size, float):
            size = int(size * len(individuals))
        if isinstance(size, int) and (size < 1 or size > len(individuals) - 1):
            raise ValueError(
                f"size should be within [1, {len(individuals)}] (inclusive) if int"
            )
        individuals_selected: Sequence[SubjectIdentifier] = []
        try:
            split_selected, _ = train_test_split(
                individuals,
                train_size=size,
                random_state=to_int_seed(random_state),
                stratify=[individual[0] for individual in individuals],
            )
        except ValueError:
            split_selected, _ = train_test_split(
                individuals,
                train_size=size,
                random_state=to_int_seed(random_state),
            )
        for individual in sorted(np.asarray(split_selected).tolist()):
            individuals_selected.append(tuple(individual))
        individuals_selected = tuple(individuals_selected)
        individuals_remaining = tuple(
            sorted(set(individuals) - set(individuals_selected))
        )
        return self._make_split(
            individuals_selected=individuals_selected,
            individuals_remaining=individuals_remaining,
            subset_actors_only=subset_actors_only,
        )

    def k_fold(
        self,
        k: int,
        *,
        random_state: int | None | np.random.Generator,
        subset_actors_only: bool = True,
        exclude: Optional[
            Sequence[GroupIdentifier | IndividualIdentifier | SubjectIdentifier]
        ],
    ) -> Generator[tuple[Self, Self], None, None]:
        if exclude is None:
            exclude = ()
        random_state = ensure_generator(random_state)
        individuals = [
            individual
            for individual in self.individuals
            if include(individual, exclude)
        ]
        try:
            kf = StratifiedKFold(
                n_splits=k, shuffle=True, random_state=to_int_seed(random_state)
            )
            for selected, remaining in kf.split(
                individuals, [individual[0] for individual in individuals]
            ):
                yield self._make_split(
                    individuals_selected=tuple(individuals[idx] for idx in selected),
                    individuals_remaining=tuple(individuals[idx] for idx in remaining),
                    subset_actors_only=subset_actors_only,
                )
        except ValueError:
            kf = KFold(n_splits=k, shuffle=True, random_state=to_int_seed(random_state))
            for selected, remaining in kf.split(individuals):
                yield self._make_split(
                    individuals_selected=tuple(individuals[idx] for idx in selected),
                    individuals_remaining=tuple(individuals[idx] for idx in remaining),
                    subset_actors_only=subset_actors_only,
                )


class AnnotatedDataset(Dataset, AnnotatedSampleableMixin):
    def __init__(
        self,
        groups: Mapping[GroupIdentifier, Group],
        *,
        observations: pd.DataFrame,
        categories: tuple[str, ...],
        background_category: str,
    ):
        AnnotatedMixin.__init__(
            self,
            categories=categories,
            background_category=background_category,
        )
        Dataset.__init__(self, groups)
        self._finalize_init(observations)

    @classmethod
    def _empty_like(cls, group: Group) -> Self:
        if not isinstance(group, AnnotatedMixin):
            raise ValueError("groups must be annotated")
        observations = pd.DataFrame()
        observations = observations.set_axis(
            labels=cls.REQUIRED_COLUMNS(group.target)
        ).T
        return cls(
            {},
            observations=observations,
            categories=group.categories,
            background_category=group.background_category,
        )
