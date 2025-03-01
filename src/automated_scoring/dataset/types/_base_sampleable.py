from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Optional, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...data_structures import Trajectory
from .._selection import get_available_indices
from ..observations.utils import (
    check_observations,
    ensure_single_index,
    infill_observations,
    to_y,
)
from ._mixins import (
    AnnotatedMixin,
    SampleableMixin,
)

if TYPE_CHECKING:
    from ...features import BaseExtractor, F


class BaseSampleable(SampleableMixin):
    def __init__(
        self,
        trajectory: Trajectory,
    ):
        self._previous_indices: list[NDArray] = []
        self._observations: pd.DataFrame | None = None
        self.trajectory = self._check_trajectory(trajectory)

    @classmethod
    def REQUIRED_COLUMNS(
        cls, target=None
    ) -> tuple[Literal["category"], Literal["start"], Literal["stop"]]:
        return ("category", "start", "stop")

    def _size(self) -> int:
        return len(self.trajectory)

    def _check_trajectory(self, trajectory: Trajectory) -> Trajectory:
        if not trajectory.is_sorted:
            raise ValueError("trajectory is not sorted.")
        if not trajectory.is_complete:
            raise ValueError("trajectory is not complete.")
        return trajectory

    def _finalize_init(self, observations: pd.DataFrame) -> None:
        if not isinstance(self, AnnotatedMixin):
            return
        observations = observations.loc[
            np.isin(observations["category"], self.categories)
        ]
        observations = ensure_single_index(
            observations,
            index_columns=(),
            drop=True,
        )
        observations = check_observations(
            observations,
            required_columns=self.REQUIRED_COLUMNS(),
            allow_overlapping=False,
            allow_unsorted=False,
        )
        observations = infill_observations(
            observations, observation_stop=self.trajectory.timestamps[-1]
        )
        self._observations = observations

    def _get_observations(self) -> pd.DataFrame:
        if self._observations is None:
            raise ValueError("not AnnotatedMixin, _finalize_init must be called first")
        return self._observations

    def _sample_y(self) -> NDArray:
        if not isinstance(self, AnnotatedMixin):
            raise ValueError(
                "only implemented for AnnotatedMixin objects inheriting from AnnotatedMixin"
            )
        return to_y(
            self.observations,
            start=self.trajectory.timestamps[0],
            stop=self.trajectory.timestamps[-1],
        )

    def _get_available_indices(
        self,
        *,
        reset_previous_indices: bool,
        exclude_previous_indices: bool,
    ) -> tuple[
        NDArray,
        NDArray | None,
        Sequence[NDArray | None],
        dict[Literal["min", "max", "offset"], int],
    ]:
        indices = np.arange(self.size)
        y = None
        intervals = None
        if isinstance(self, AnnotatedMixin):
            y = self.sample_y()
            intervals = self.observations.copy()
            intervals["category"] = np.arange(len(intervals), dtype=int)
            intervals = to_y(
                intervals,
                start=self.trajectory.timestamps[0],
                stop=self.trajectory.timestamps[-1],
                dtype=int,
            )
        if reset_previous_indices:
            self._previous_indices.clear()
        y_available, intervals_available = y, intervals
        if exclude_previous_indices and len(self._previous_indices) > 0:
            indices, y_available, intervals_available = get_available_indices(
                self._previous_indices, indices=indices, y=y, intervals=intervals
            )
        return (
            indices,
            y_available,
            [intervals_available],
            {"min": 0, "max": self.size - 1, "offset": 0},
        )

    def _select_samples(
        self,
        extractor: BaseExtractor[F],
        indices: NDArray,
        splits: Optional[dict],
        *,
        store_indices: bool,
    ) -> tuple[F, NDArray | None]:
        if splits is not None:
            if not all([key in splits for key in ["min", "max", "offset"]]):
                raise ValueError(
                    "invalid splits dictionary, should contain min, max and offset keys"
                )
            indices = indices - splits["offset"]
            indices = indices[(indices >= splits["min"]) & (indices < splits["max"])]
        if indices.size == 0:
            if store_indices:
                self._previous_indices.append(indices)
            y = None
            if isinstance(self, AnnotatedMixin):
                y = np.array([])
            return extractor.empty(), y
        X = self.sample_X(extractor)
        y = None
        if isinstance(self, AnnotatedMixin):
            y = self.sample_y()
        if store_indices:
            self._previous_indices.append(indices)
        if isinstance(X, pd.DataFrame):
            X = X.iloc[indices]
        elif isinstance(X, np.ndarray):
            X = X[indices]
        else:
            raise TypeError("unsupported sample type")
        if y is not None:
            y = y[indices]
        if TYPE_CHECKING:
            X = cast(F, X)
        return X, y
