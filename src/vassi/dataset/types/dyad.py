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


class Dyad(BaseSampleable):
    """
    A dyad is a sampleable pair of trajectories (:class:`~vassi.data_structures.trajectory.Trajectory`).

    Parameters:
        trajectory: The first trajectory in the dyad (i.e., the actor of social behaviors).
        trajectory_other: The second trajectory in the dyad (i.e., the recipient).
    """

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
        """
        Helper function to prepare paired trajectories for a dyad.

        The trajectories are temporally aligned with slicing (see :meth:`~vassi.data_structures.trajectory.Trajectory.slice_window`) and returned as a view.
        Note that the trajectories should already be complete and sorted.

        Parameters:
            trajectory: The trajectory of the actor (first individual of the dyad)
            trajectory_other: The trajectory of the recipient (second individual of the dyad)

        Returns:
            The aligned trajectories.

        See also:
            :class:`~vassi.data_structures.trajectory.Trajectory` for more details on handling trajectory data.
        """
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

    def _sample_X[F: Shaped](
        self,
        extractor: BaseExtractor[F],
    ) -> F:
        return extractor.extract(self.trajectory, self.trajectory_other)

    def annotate(
        self,
        observations: pd.DataFrame,
        *,
        categories: tuple[str, ...],
        background_category: str,
    ) -> "AnnotatedDyad":
        """
        Annotates the dyad with the given observations.

        Parameters:
            observations: The observations.
            categories: Categories of the observations.
            background_category: The background category of the observations.

        Returns:
            The annotated dyad.
        """
        return AnnotatedDyad(
            self.trajectory,
            self.trajectory_other,
            observations=observations,
            categories=categories,
            background_category=background_category,
        )


class AnnotatedDyad(Dyad, AnnotatedSampleableMixin):
    """
    Annotated dyad.

    Parameters:
        trajectory: The trajectory of the actor.
        trajectory_other: The other trajectory of the recipient.
        observations: Observations for the dyad.
        categories: Categories of the observations.
        background_category: Background category of the observations.
    """

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
