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
        return AnnotatedIndividual(
            trajectory=self.trajectory,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )


class AnnotatedIndividual(Individual, AnnotatedSampleableMixin):
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
