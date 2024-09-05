import pandas as pd

from ...data_structures import Trajectory
from ._sampleable import AnnotatedSampleable, Sampleable


class Dyad(Sampleable):
    trajectory_other: Trajectory

    def __init__(self, trajectory: Trajectory, trajectory_other: Trajectory) -> None:
        super().__init__(trajectory, trajectory_other)

    def annotate(
        self,
        annotations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
    ) -> "AnnotatedDyad":
        return AnnotatedDyad(
            self.trajectory,
            self.trajectory_other,
            annotations=annotations,
            categories=categories,
        )


class AnnotatedDyad(AnnotatedSampleable):
    trajectory_other: Trajectory

    def __init__(
        self,
        trajectory: Trajectory,
        trajectory_other: Trajectory,
        annotations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
    ) -> None:
        super().__init__(
            trajectory, trajectory_other, annotations=annotations, categories=categories
        )
