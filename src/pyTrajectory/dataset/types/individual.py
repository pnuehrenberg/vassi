import pandas as pd

from ...data_structures import Trajectory
from ._sampleable import AnnotatedSampleable, Sampleable


class Individual(Sampleable):
    trajectory_other: None

    def __init__(self, trajectory: Trajectory) -> None:
        super().__init__(trajectory)

        def annotate(
            self,
            annotations: pd.DataFrame,
            *,
            categories: tuple[str, ...],
        ) -> "AnnotatedIndividual":
            return AnnotatedIndividual(
                self.trajectory,
                annotations=annotations,
                categories=categories,
            )


class AnnotatedIndividual(AnnotatedSampleable):
    trajectory_other: None

    def __init__(
        self,
        trajectory: Trajectory,
        annotations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
    ) -> None:
        super().__init__(trajectory, annotations=annotations, categories=categories)
