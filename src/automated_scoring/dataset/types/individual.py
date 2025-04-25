from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ...data_structures import Trajectory
from ._base_sampleable import BaseSampleable
from .mixins import (
    AnnotatedMixin,
    AnnotatedSampleableMixin,
)

if TYPE_CHECKING:
    from ...features import BaseExtractor, Shaped


class Individual(BaseSampleable):
    """
    An individual is a sampleable that holds one trajectory.

    Parameters:
        trajectory: The trajectory of the individual.
    """

    def _sample_X[F: Shaped](
        self,
        extractor: BaseExtractor[F],
    ) -> F:
        return extractor.extract(self.trajectory)

    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
        background_category: str,
    ) -> "AnnotatedIndividual":
        """
        Annotates the individual with the given observations.

        Parameters:
            observations: The observations.
            categories: Categories of the observations.
            background_category: The background category of the observations.

        Returns:
            The annotated individual.
        """
        return AnnotatedIndividual(
            trajectory=self.trajectory,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )


class AnnotatedIndividual(Individual, AnnotatedSampleableMixin):
    """
    Annotated individual.

    Parameters:
        trajectory: The trajectory of the individual.
        observations: The observations corresponding to the trajectory.
        categories: Categories of the observations.
        background_category: The background category of the observations.
    """

    def __init__(
        self,
        trajectory: Trajectory,
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
        Individual.__init__(self, trajectory)
        self._finalize_init(observations)
