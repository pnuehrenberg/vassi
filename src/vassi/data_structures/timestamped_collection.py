from typing import Mapping, Optional, Self

import numpy as np

from .. import config
from . import utils
from .collection import InstanceCollection


class TimestampedInstanceCollection(InstanceCollection):
    """
    Represents a collection of timestamped instances.

    This class provides direct access to timestamps and adds functionality specific to timestamped data, such as sorting by timestamp
    and slicing based on time windows. It ensures that a timestamp key is defined in the configuration.

    Parameters:
        data: The instance data for the collection.
        cfg : The configuration object.
        validate_on_init: Whether to validate the data during initialization.

    Raises:
        ValueError: If the timestamp key is not defined in the trajectory keys during initialization.
    """

    def __init__(
        self,
        *,
        data: Optional[Mapping[str, np.ndarray]] = None,
        cfg: Optional[config.Config] = None,
        validate_on_init: bool = True,
    ) -> None:
        super().__init__(data=data, cfg=cfg, validate_on_init=validate_on_init)
        try:
            _ = self.key_timestamp
        except AssertionError:
            raise ValueError(
                "cannot initialize with undefined timestamp key (key has to be present in trajectory_keys)."
            )

    @property
    def key_timestamp(self) -> str:
        """Returns the key used to access the timestamp within each instance."""
        assert self.cfg.key_timestamp is not None
        assert self.cfg.key_timestamp in self.cfg.trajectory_keys
        return self.cfg.key_timestamp

    @property
    def timestamps(self) -> np.ndarray:
        """Returns the timestamps of the instances in the collection."""
        return self[self.key_timestamp]

    @property
    def is_sorted(self) -> bool:
        """Checks if the timestamps in the collection are sorted in ascending order."""
        timestamps = self.timestamps
        # trajectories do not allow duplicate timestamps, so >= is valid
        return bool(np.all(timestamps[1:] >= timestamps[:-1]))

    def sort(self, copy: bool = True) -> Self:
        """
        Sorts the collection by timestamp.

        Parameters:
            copy: Whether to return a copy of the sorted collection.

        Returns:
            The sorted collection.

        Raises:
            ValueError: If :code:`copy=False` and the collection is a view or has existing views.
        """
        if not copy and len(self._view_of) > 0:
            base = ", ".join([str(base) for base in self._view_of])
            raise ValueError(f"can not safely sort a view of {base}, copy first.")
        elif not copy and len(self._views) > 0:
            view = ", ".join([str(view) for view in self._views])
            raise ValueError(
                f"can not safely sort because views exist ({view}), copy first."
            )
        if self.is_sorted:
            if not copy:
                return self
            return self.copy()
        sort_idx = np.argsort(self.timestamps)
        data = {key: value[sort_idx] for key, value in self.items()}
        if not copy:
            self.data = data
            return self
        return self.init_other(data=data)

    def get_slice(
        self,
        start: int | float,
        stop: int | float,
    ) -> slice:
        """
        Converts a time window to a slice object for accessing data within the window.

        Parameters:
            start: The start time of the window (inclusive).
            stop: The end time of the window (inclusive).

        Returns:
            The slice object for accessing data within the window.

        Raises:
            ValueError: If the collection has zero length or is not sorted.
            utils.OutOfInterval: If the start or stop time is outside the timestamp range.
        """
        if self.length == 0:
            raise ValueError("window slicing requires non zero-length collection")
        if not self.is_sorted:
            raise ValueError(
                "window slicing requires sorted collection, call sort first."
            )
        timestamps = self.timestamps
        first = timestamps.min()
        last = timestamps.max()
        if start < first:
            raise utils.OutOfInterval(
                f"start: {start} not in timestamp range: [{first} {last}]"
            )
        if stop > last:
            raise utils.OutOfInterval(
                f"stop: {stop} not in timestamp range: [{first} {last}]"
            )
        return utils.get_interval_slice(timestamps, start, stop)

    def slice_window(
        self,
        start: int | float,
        stop: int | float,
    ) -> Self:
        """
        Slices the collection based on a specified time window.

        Parameters:
            start: The start time of the window (inclusive).
            stop: The end time of the window (inclusive).

        Returns:
            The sliced collection based on the specified time window.
        """
        return self[self.get_slice(start, stop)]
