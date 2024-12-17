import pandas as pd

from ...data_structures import Trajectory
from ._sampleable import AnnotatedSampleable, Sampleable


class Individual(Sampleable):
    trajectory_other: None

    def __init__(self, trajectory: Trajectory) -> None:
        super().__init__(trajectory)

        def annotate(
            self,
            observations: pd.DataFrame,
            *,
            categories: tuple[str, ...],
        ) -> "AnnotatedIndividual":
            return AnnotatedIndividual(
                self.trajectory,
                observations=observations,
                categories=categories,
            )


class AnnotatedIndividual(AnnotatedSampleable):
    trajectory_other: None

    def __init__(
        self,
        trajectory: Trajectory,
        observations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
    ) -> None:
        super().__init__(trajectory, observations=observations, categories=categories)
