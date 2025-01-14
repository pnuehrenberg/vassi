from typing import Mapping, Optional, Self

import numpy as np
from numpy.typing import NDArray

from .. import config
from . import utils
from .collection import InstanceCollection


class TimestampedInstanceCollection(InstanceCollection):
    """
    Data structure for collections of instances with timestamps.

    Parameters
    ----------
    data: Mapping[str, NDArray], optional
        Data of the instances.

    cfg: Config, optional
        Configuration of the instances.

    validate_on_init: bool, optional
        Whether to validate the data on initialization.
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
        Timestamp key of the collection.
        """
        assert self.cfg.key_timestamp is not None
        assert self.cfg.key_timestamp in self.cfg.trajectory_keys
        return self.cfg.key_timestamp

    @property
    def timestamps(self) -> NDArray[np.int64 | np.float64]:
        """
        Timestamps of the collection.
        """
        return self[self.key_timestamp]

    @property
    def is_sorted(self) -> np.bool_:
        """
        Whether the collection is sorted by timestamps.
        """
        timestamps = self.timestamps
        # trajectories do not allow duplicate timestamps, so >= is valid
        return np.all(timestamps[1:] >= timestamps[:-1])

    def sort(self, copy: bool = True) -> Self:
        """
        Sort the collection by timestamps.

        Parameters
        ----------
        copy: bool, optional
            Whether to copy the collection before sorting.

        Returns
        -------
        Self
            Sorted collection.

        Raises
        ------
        ValueError
            If the collection is is a view of another, or has views of itself.
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
        Convert timestamps (start, stop, inclusive) to slice.

        Parameters
        ----------
        start: int | float
            Start timestamp.

        stop: int | float
            Stop timestamp.

        Returns
        -------
        slice
            Slice for the specified timestamps.

        Raises
        ------
        ValueError
            If the collection is empty or not sorted.
        OutOfInterval
            If the specified timestamps are out of the interval of the collection.
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
        Slice the collection by timestamps.

        Parameters
        ----------
        start: int | float
            Start timestamp, inclusive.

        stop: int | float
            Stop timestamp, inclusive.

        Returns
        -------
        Self
            Sliced collection (view).
        """
        return self[self._window_to_slice(start, stop)]
