import pandas as pd
from numpy.typing import NDArray

from ...data_structures import Trajectory
from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ._base_sampleable import BaseSampleable
from ._mixins import (
    AnnotatedMixin,
    AnnotatedSampleableMixin,
)


class Individual(BaseSampleable):
    def _sample_X(
        self,
        extractor: FeatureExtractor | DataFrameFeatureExtractor,
    ) -> pd.DataFrame | NDArray:
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
