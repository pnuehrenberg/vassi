import pandas as pd
from numpy.typing import NDArray

from ...data_structures import Trajectory
from ...features import DataFrameFeatureExtractor, FeatureExtractor
from ._base_sampleable import BaseSampleable
from ._mixins import (
    AnnotatedMixin,
    AnnotatedSampleableMixin,
)


class Dyad(BaseSampleable):
    def __init__(
        self,
        trajectory: Trajectory,
        trajectory_other: Trajectory,
    ):
        super().__init__(trajectory)
        self.trajectory_other = self._check_paired_trajectories(trajectory_other)

    @classmethod
    def prepare_paired_trajectories(
        cls, trajectory: Trajectory, trajectory_other: Trajectory
    ) -> tuple[Trajectory, Trajectory]:
        start = max(trajectory.timestamps[0], trajectory_other.timestamps[0])
        stop = min(trajectory.timestamps[-1], trajectory_other.timestamps[-1])
        if trajectory.timestamps[0] < start or trajectory.timestamps[-1] > stop:
            trajectory = trajectory.slice_window(
                start, stop, interpolate=False, copy=False
            )
        if (
            trajectory_other.timestamps[0] < start
            or trajectory_other.timestamps[-1] > stop
        ):
            trajectory_other = trajectory_other.slice_window(
                start, stop, interpolate=False, copy=False
            )
        return trajectory, trajectory_other

    def _check_paired_trajectories(self, trajectory_other: Trajectory) -> Trajectory:
        trajectory_other = self._check_trajectory(trajectory_other)
        if self.trajectory.timestep != trajectory_other.timestep:
            raise ValueError("trajectories have unequal timesteps.")
        if self.trajectory.timestamps[0] != trajectory_other.timestamps[0]:
            raise ValueError("trajectories have mismatched timestamps.")
        if self.trajectory.timestamps[-1] != trajectory_other.timestamps[-1]:
            raise ValueError("trajectories have mismatched timestamps.")
        return trajectory_other

    def _sample_X(
        self,
        extractor: FeatureExtractor | DataFrameFeatureExtractor,
    ) -> pd.DataFrame | NDArray:
        return extractor.extract(self.trajectory, self.trajectory_other)

    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
        background_category: str,
    ) -> "AnnotatedDyad":
        return AnnotatedDyad(
            self.trajectory,
            self.trajectory_other,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )


class AnnotatedDyad(Dyad, AnnotatedSampleableMixin):
    def __init__(
        self,
        trajectory: Trajectory,
        trajectory_other: Trajectory,
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
        Dyad.__init__(self, trajectory, trajectory_other)
        self._finalize_init(observations)
