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

from ...utils import to_int_seed
from ..observations.utils import check_observations, with_duration
from ..utils import GroupIdentifier, IndividualIdentifier, SubjectIdentifier
from .group import Group
from .mixins import (
    AnnotatedMixin,
    AnnotatedSampleableMixin,
    NestedSampleableMixin,
    SampleableMixin,
)


def include(
    individual: SubjectIdentifier,
    exclude: Sequence[GroupIdentifier | IndividualIdentifier | SubjectIdentifier],
) -> bool:
    """
    Check if an individual (i.e., subject, individual of a group in a dataset) should be included.

    Parameters:
        individual: The individual to check.
        exclude: A sequence of identifiers to exclude.

    Returns:
        If the individual should be included.
    """
    exclude = list(exclude)
    if individual in exclude:
        return False
    if individual[0] in exclude:
        return False
    if individual[1] in exclude:
        return False
    return True


class Dataset(NestedSampleableMixin, SampleableMixin):
    """
    A dataset is a collection of groups (:class:`~automated_scoring.dataset.types.group.Group`),
    each of which is a collection of individuals (:class:`~automated_scoring.dataset.types.individual.Individual`) or
    dyads (:class:`~automated_scoring.dataset.types.dyad.Dyad`).

    Parameters:
        groups: A mapping of group identifiers to groups.
        target: The target type of the dataset.
    """

    def __init__(
        self,
        groups: Mapping[GroupIdentifier, Group],
        *,
        target: Literal["individual", "dyad"],
    ):
        self._target = target
        for group in groups.values():
            if group.target != target:
                raise ValueError(
                    f"Groups must all be of the same target type '{target}' (got '{group.target}')"
                )
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
        cls, target: Optional[Literal["individual", "dyad"]] = None
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
        """
        Returns the required columns for annotations with the given target.

        Parameters:
            target: The target type for the annotations.

        Returns:
            The required columns for annotations.
        """
        if target == "individual":
            return ("group", "actor", "category", "start", "stop")
        elif target == "dyad":
            return ("group", "actor", "recipient", "category", "start", "stop")
        else:
            raise ValueError("target argument must be either 'individual' or 'dyad'")

    def __next__(self) -> tuple[GroupIdentifier, Group]:
        identifier, group = super().__next__()
        if TYPE_CHECKING:
            assert isinstance(group, Group)
            assert isinstance(identifier, GroupIdentifier)
        return identifier, group

    @property
    def individuals(self) -> tuple[SubjectIdentifier, ...]:
        """Returns a tuple of all subjects (individuals in groups) in the dataset."""
        individuals: list[SubjectIdentifier] = []
        for identifier, group in self:
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
        """
        Annotates the dataset with the given observations.

        Parameters:
            observations: The observations.
            categories: Categories of the observations.
            background_category: The background category of the observations.

        Returns:
            The annotated dataset.
        """
        groups: dict[GroupIdentifier, Group] = {}
        for identifier, group in self:
            groups[identifier] = group
        return AnnotatedDataset(
            groups,
            target=self.target,
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
        observations = check_observations(
            observations,
            required_columns=self.REQUIRED_COLUMNS(self.target),
            allow_overlapping=True,
            allow_unsorted=True,
        )
        for identifier, group in self:
            observations_group = observations.loc[observations["group"] == identifier]
            self._sampleables[identifier] = group.annotate(
                observations=observations_group,
                categories=self.categories,
                background_category=self.background_category,
            )

    @with_duration
    def _get_observations(self) -> pd.DataFrame:
        observations: list[pd.DataFrame] = []
        for identifier, group in self:
            group = self.select(identifier)
            if not isinstance(group, AnnotatedMixin):
                raise ValueError("unannotated groups do not have observations")
            observations_group = group.observations
            observations_group["group"] = identifier
            observations.append(observations_group)
        return pd.concat(observations, axis=0, ignore_index=True)

    @classmethod
    def _empty_like(cls, group: Group) -> Self:
        return cls({}, target=group.target)

    @classmethod
    def from_groups(
        cls,
        groups: Mapping[GroupIdentifier, Group],
    ) -> Self:
        """
        Create a new dataset from a groups.

        Args:
            groups: The groups to include in the dataset.

        Returns:
            The dataset.
        """
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
        for identifier, group in self:
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

    def exclude_individuals(
        self,
        individuals: Sequence[IndividualIdentifier | SubjectIdentifier],
        *,
        subset_actors_only: bool = True,
    ) -> Self:
        """
        Exclude individuals from the dataset.

        Args:
            individuals: The individuals to exclude.
            subset_actors_only: Whether to exclude only actors if :code:`target="dyad"`. This drops all dyads involving the excluded individuals as actors. Otherwise, all dyads that involve excluded individuals (as either actor or recipient) are dropped.

        Returns:
            The dataset with the excluded individuals.
        """
        individuals_selected = tuple(
            individual
            for individual in self.individuals
            if include(individual, individuals)
        )
        individuals_remaining = tuple(
            sorted(set(self.individuals) - set(individuals_selected))
        )
        return self._make_split(
            individuals_selected=individuals_selected,
            individuals_remaining=individuals_remaining,
            subset_actors_only=subset_actors_only,
        )[0]

    def split(
        self,
        size: int | float,
        *,
        random_state: int | None | np.random.Generator,
        subset_actors_only: bool = True,
    ) -> tuple[Self, Self]:
        """
        Split the dataset into two subsets.

        Args:
            size: The size of the first subset. If float, it should be within (0.0, 1.0) interval (exclusive).
            random_state: The random state for reproducibility.
            subset_actors_only: Whether to only include actors in the split.

        Returns:
            A tuple of two subsets.

        See also:
            :meth:`exclude_individuals` for more details on the :code:`subset_actors_only` parameter.
        """
        random_state = np.random.default_rng(random_state)
        if isinstance(size, float) and (size < 0 or size > 1):
            raise ValueError(
                "size should be within (0.0, 1.0) interval (exclusive) if float"
            )
        individuals = self.individuals
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
        for individual in split_selected:
            individuals_selected.append(tuple(individual))
        individuals_selected = tuple(sorted(individuals_selected))
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
    ) -> Generator[tuple[Self, Self], None, None]:
        """
        Yields a generator of k-fold splits.

        Args:
            k: The number of folds.
            random_state: The random state to use for splitting.
            subset_actors_only: Whether to only include actors in the split.

        Yields:
            A generator of k-fold splits.

        See also:
            :meth:`exclude_individuals` for more details on the :code:`subset_actors_only` parameter.
        """
        random_state = np.random.default_rng(random_state)
        individuals = self.individuals
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
    """
    Annotated dataset.

    Parameters:
        groups: The groups of individuals in the dataset.
        target: The target of the dataset.
        observations: The observations of the dataset.
        categories: The categories of the dataset.
        background_category: The background category of the dataset.

    """

    def __init__(
        self,
        groups: Mapping[GroupIdentifier, Group],
        *,
        target: Literal["individual", "dyad"],
        observations: pd.DataFrame,
        categories: tuple[str, ...],
        background_category: str,
    ):
        AnnotatedMixin.__init__(
            self,
            categories=categories,
            background_category=background_category,
        )
        Dataset.__init__(self, groups, target=target)
        self._finalize_init(observations)

    @classmethod
    def _empty_like(cls, group: Group) -> Self:
        if not isinstance(group, AnnotatedMixin):
            raise ValueError("groups must be annotated")
        observations = pd.DataFrame(
            columns=pd.Index(cls.REQUIRED_COLUMNS(group.target))
        )
        return cls(
            {},
            target=group.target,
            observations=observations,
            categories=group.categories,
            background_category=group.background_category,
        )
