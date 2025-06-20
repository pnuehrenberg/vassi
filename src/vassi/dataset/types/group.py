from collections.abc import Iterable, Mapping, Sequence
from itertools import permutations
from typing import (
    TYPE_CHECKING,
    Literal,
    Self,
    cast,
    overload,
)

import numpy as np
import pandas as pd

from ...data_structures import Trajectory
from ..observations.utils import with_duration
from ..utils import DyadIdentifier, Identifier, IndividualIdentifier, get_actor
from ._base_sampleable import BaseSampleable
from .dyad import Dyad
from .individual import Individual
from .mixins import (
    AnnotatedMixin,
    AnnotatedSampleableMixin,
    NestedSampleableMixin,
    SampleableMixin,
)


class Group(NestedSampleableMixin, SampleableMixin):
    """
    A group is a collection of individuals (:class:`~vassi.dataset.types.individual.Individual`) or
    dyads (:class:`~vassi.dataset.types.dyad.Dyad`), depending on the target.

    Parameters:
        trajectories: A mapping of individual identifiers to trajectories.
        target: The target type of the dataset.
    """

    def __init__(
        self,
        trajectories: Iterable[Trajectory] | Mapping[IndividualIdentifier, Trajectory],
        *,
        target: Literal["individual", "dyad"],
    ):
        _trajectories: list[Trajectory] = (
            list(trajectories.values())
            if isinstance(trajectories, Mapping)
            else list(trajectories)
        )
        _individuals: list[IndividualIdentifier] = (
            list(range(len(_trajectories)))
            if not isinstance(trajectories, Mapping)
            else list(
                cast(Mapping[IndividualIdentifier, Trajectory], trajectories).keys()
            )  # cast seems necessary here for pyright, as keys is inferred incorrectly
        )
        self.trajectories = {
            individual: trajectory
            for individual, trajectory in zip(_individuals, _trajectories)
        }
        self._target = target
        self._sampleables = {}
        for identifier in self.potential_identifiers:
            if self.target == "individual":
                if TYPE_CHECKING:
                    assert isinstance(identifier, IndividualIdentifier)
                self._sampleables[identifier] = Individual(
                    self.trajectories[identifier]
                )
            elif self.target == "dyad":
                if TYPE_CHECKING:
                    assert isinstance(identifier, tuple)
                self._sampleables[identifier] = Dyad(
                    *Dyad.prepare_paired_trajectories(
                        self.trajectories[identifier[0]],
                        self.trajectories[identifier[1]],
                    )
                )
            else:
                raise ValueError("target must be either 'individual' or 'dyad'")

    @classmethod
    def _empty_like(cls, group: Self) -> Self:
        return cls([], target=group.target)

    @classmethod
    def from_group(
        cls,
        group: Self,
        *,
        individuals: Sequence[IndividualIdentifier],
        subset_actors_only: bool,
    ) -> Self:
        """
        Create a new group as a subset of the original group.

        Parameters:
            group: The original group to create a subset from.
            individuals: The individuals to include in the new group.
            subset_actors_only: Only applicable when :code:`target="dyad"`. Whether to drop dyads with non-included individuals as actors or recipients, or only when they are actors.

        Returns:
            A new group containing only the specified individuals.
        """

        def get_actor(
            identifier: IndividualIdentifier | DyadIdentifier,
        ) -> IndividualIdentifier:
            if isinstance(identifier, IndividualIdentifier):
                return identifier
            return identifier[0]

        new = cls._empty_like(group)
        if subset_actors_only:
            new.trajectories = group.trajectories
        else:
            new.trajectories = {
                individual: group.trajectories[individual]
                for individual in group.individuals
                if individual in individuals
            }
        for identifier in new.potential_identifiers:
            if get_actor(identifier) not in individuals:
                continue
            if identifier not in group.identifiers:
                # was excluded on group init
                continue
            new._sampleables[identifier] = group.select(identifier)
        return new

    @overload
    @classmethod
    def REQUIRED_COLUMNS(
        cls, target: Literal["individual"]
    ) -> tuple[
        Literal["actor"], Literal["category"], Literal["start"], Literal["stop"]
    ]: ...

    @overload
    @classmethod
    def REQUIRED_COLUMNS(
        cls, target: Literal["dyad"]
    ) -> tuple[
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
        tuple[Literal["actor"], Literal["category"], Literal["start"], Literal["stop"]]
        | tuple[
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
            return ("actor", "category", "start", "stop")
        elif target == "dyad":
            return ("actor", "recipient", "category", "start", "stop")
        else:
            raise ValueError("target argument must be either 'individual' or 'dyad'")

    @property
    def individuals(self) -> tuple[IndividualIdentifier, ...]:
        """Returns the identifiers of individuals in the group."""
        individuals = tuple(sorted(self.trajectories))
        return tuple(
            individual
            for individual in individuals
            if any(
                [individual == get_actor(identifier) for identifier in self.identifiers]
            )
        )

    @property
    def potential_identifiers(
        self,
    ) -> tuple[IndividualIdentifier, ...] | tuple[DyadIdentifier, ...]:
        """Returns the identifiers of individuals or all potential identifiers of dyads in the group (all combinations of individuals)."""
        individuals = tuple(sorted(self.trajectories))
        if self.target == "individual":
            return individuals
        return tuple(permutations(individuals, 2))

    def _get_identifiers(
        self,
    ) -> tuple[IndividualIdentifier, ...] | tuple[DyadIdentifier, ...]:
        individual_identifiers: list[IndividualIdentifier] = []
        dyad_identifiers: list[DyadIdentifier] = []
        for identifier in tuple(self._sampleables):
            if self.target == "individual":
                if TYPE_CHECKING:
                    assert isinstance(identifier, str | int)
                individual_identifiers.append(identifier)
            else:
                if TYPE_CHECKING:
                    assert isinstance(identifier, tuple)
                dyad_identifiers.append(identifier)
        return (
            tuple(individual_identifiers)
            if self.target == "individual"
            else tuple(dyad_identifiers)
        )

    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
        background_category: str,
    ) -> "AnnotatedGroup":
        """
        Annotates the group with the given observations.

        Parameters:
            observations: The observations.
            categories: Categories of the observations.
            background_category: The background category of the observations.

        Returns:
            The annotated group.
        """
        return AnnotatedGroup(
            self.trajectories,
            target=self.target,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )

    def __next__(self) -> tuple[Identifier, BaseSampleable]:
        identifier, sampleable = super().__next__()
        if TYPE_CHECKING:
            assert isinstance(sampleable, BaseSampleable)
        return identifier, sampleable

    def _finalize_init(self, observations: pd.DataFrame) -> None:
        if not isinstance(self, AnnotatedMixin):
            return
        identifier_columns = (
            ["actor", "recipient"] if self.target == "dyad" else ["actor"]
        )
        observations = observations.loc[
            np.isin(observations["category"], self.categories)
        ]
        for identifier, sampleable in self:
            observations_sample = observations.loc[
                (
                    observations[identifier_columns]
                    == (identifier if isinstance(identifier, tuple) else (identifier,))
                ).all(axis=1)
            ]
            self._sampleables[identifier] = sampleable.annotate(
                observations=observations_sample,
                categories=self.categories,
                background_category=self.background_category,
            )

    @with_duration
    def _get_observations(self) -> pd.DataFrame:
        observations: list[pd.DataFrame] = []
        identifier_columns = (
            ["actor", "recipient"] if self.target == "dyad" else ["actor"]
        )
        for identifier, sampleable in self:
            if not isinstance(sampleable, AnnotatedMixin):
                raise ValueError("unannotated sampleables do not have observations")
            observations_sampleable = sampleable.observations
            if not isinstance(identifier, tuple):
                identifier = (identifier,)
            for column, value in zip(identifier_columns, identifier):
                observations_sampleable[column] = value
            observations.append(observations_sampleable)
        return pd.concat(observations, axis=0, ignore_index=True)


class AnnotatedGroup(Group, AnnotatedSampleableMixin):
    """
    Annotated group.

    Parameters:
        trajectories: Individual trajectories to be included in the group.
        target: The target of the group.
        observations: Observations for the group.
        categories: Categories of the observations.
        background_category: Background category of the observations.
    """

    def __init__(
        self,
        trajectories: Iterable[Trajectory] | Mapping[IndividualIdentifier, Trajectory],
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
        Group.__init__(self, trajectories, target=target)
        self._finalize_init(observations)

    @classmethod
    def _empty_like(cls, group: Group) -> Self:
        if not isinstance(group, cls):
            raise ValueError("group must be of type {}".format(cls.__name__))
        observations = pd.DataFrame(
            columns=pd.Index(cls.REQUIRED_COLUMNS(group.target))
        )
        return cls(
            [],
            target=group.target,
            observations=observations,
            categories=group.categories,
            background_category=group.background_category,
        )
