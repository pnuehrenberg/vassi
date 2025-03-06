from typing import Mapping, Optional, Self

import numpy as np
from numpy.typing import NDArray

from .. import config
from . import utils
from .collection import InstanceCollection


class TimestampedInstanceCollection(InstanceCollection):
    """
    Represents a collection of timestamped instances, inheriting from InstanceCollection.

    This class extends `InstanceCollection` and adds functionality specific to timestamped data, such as sorting by timestamp and slicing based on time windows. It ensures that a timestamp key is defined in the configuration.

    Parameters
    ----------
    data : Mapping of str and numpy.ndarray, optional
        A dictionary containing the data for the collection (default is None).
    cfg : config.Config, optional
        The configuration object (default is None).
    validate_on_init : bool, optional
        Whether to validate the data during initialization (default is True).

    Raises
    ------
    ValueError
        If the timestamp key is not defined in the trajectory keys during initialization.
    """

    def __init__(
        self,
        *,
        data: Optional[Mapping[str, NDArray]] = None,
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
        """
        Returns the key used to access the timestamp within each instance.

        Returns
        -------
        str
            The key used to access the timestamp.
        """
        assert self.cfg.key_timestamp is not None
        assert self.cfg.key_timestamp in self.cfg.trajectory_keys
        return self.cfg.key_timestamp

    @property
    def timestamps(self) -> NDArray[np.int64 | np.float64]:
        """
        Returns the timestamps of the instances in the collection.

        Returns
        -------
        NDArray[np.int64 | np.float64]
            An array containing the timestamps of the instances.
        """
        return self[self.key_timestamp]

    @property
    def is_sorted(self) -> bool:
        """
        Checks if the timestamps in the collection are sorted in ascending order.

        Returns
        -------
        bool
            True if the timestamps are sorted, False otherwise.
        """
        timestamps = self.timestamps
        # trajectories do not allow duplicate timestamps, so >= is valid
        return bool(np.all(timestamps[1:] >= timestamps[:-1]))

    def sort(self, copy: bool = True) -> Self:
        """
        Sorts the collection by timestamp.

        Parameters
        ----------
        copy : bool, optional
            Whether to return a copy of the sorted collection, defaults to True.

        Returns
        -------
        Self
            The sorted collection (or a copy if `copy` is True).

        Raises
        ------
        ValueError
            If `copy` is False and the collection is a view or has existing views.
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
        return self._init_other(data=data)

    def _window_to_slice(
        self,
        start: int | float,
        stop: int | float,
    ) -> slice:
        """
        Converts a time window to a slice object for accessing data within the window.

        Parameters
        ----------
        start : int or float
            The start time of the window (inclusive).
        stop : int or float
            The end time of the window (inclusive).

        Returns
        -------
        slice
            A slice object representing the indices of the timestamps within the specified window.

        Raises
        ------
        ValueError
            If the collection has zero length or is not sorted.
        utils.OutOfInterval
            If the start or stop time is outside the timestamp range.
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

        This method allows for extracting a subset of the data within a given time range. The slicing operation is performed using internal indexing logic.

        Parameters
        ----------
        start : int | float
            The start time of the window (inclusive).
        stop : int | float
            The end time of the window (exclusive).

        Returns
        -------
        Self
            A new TimestampedInstanceCollection containing the sliced data.
        """
        return self[self._window_to_slice(start, stop)]
