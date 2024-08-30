from typing import Optional

import pandas as pd

from ...data_structures import Trajectory
from .sampleable import AnnotatedSampleable, Sampleable


class Dyad(Sampleable):
    def __init__(self, trajectory: Trajectory, trajectory_other: Trajectory) -> None:
        super().__init__(trajectory, trajectory_other)

        def annotate(
            self,
            annotations: pd.DataFrame,
            categories: Optional[tuple[str, ...]] = None,
        ) -> "AnnotatedDyad":
            return AnnotatedDyad(
                self.trajectory,
                self.trajectory_other,
                annotations=annotations,
                categories=categories,
            )


class AnnotatedDyad(AnnotatedSampleable):
    def __init__(
        self,
        trajectory: Trajectory,
        trajectory_other: Trajectory,
        annotations: pd.DataFrame,
        categories: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            trajectory, trajectory_other, annotations=annotations, categories=categories
        )
