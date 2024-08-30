from typing import Optional

import pandas as pd

from ...data_structures import Trajectory
from .sampleable import AnnotatedSampleable, Sampleable


class Individual(Sampleable):
    def __init__(self, trajectory: Trajectory) -> None:
        super().__init__(trajectory)

        def annotate(
            self,
            annotations: pd.DataFrame,
            categories: Optional[tuple[str, ...]] = None,
        ) -> "AnnotatedIndividual":
            return AnnotatedIndividual(
                self.trajectory,
                annotations=annotations,
                categories=categories,
            )


class AnnotatedIndividual(AnnotatedSampleable):
    def __init__(
        self,
        trajectory: Trajectory,
        annotations: pd.DataFrame,
        categories: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__(trajectory, annotations=annotations, categories=categories)
