from typing import Self, Optional, Mapping

import numpy as np
from numpy.typing import NDArray

from .. import config
from . import utils
from .collection import InstanceCollection


class TimestampedInstanceCollection(InstanceCollection):
    def __init__(
        self,
        *,
        data: Optional[Mapping[str, NDArray]] = None,
        cfg: Optional[config.Config] = None,
    ) -> None:
        super().__init__(data=data, cfg=cfg)
        try:
            _ = self.key_timestamp
        except AssertionError:
            raise ValueError(
                "cannot initialize with undefined timestamp key (key has to be present in trajectory_keys)."
            )

    @property
    def key_timestamp(self) -> str:
        assert self.cfg.key_timestamp is not None
        assert self.cfg.key_timestamp in self.cfg.trajectory_keys
        return self.cfg.key_timestamp

    @property
    def timestamps(self) -> NDArray[np.int64 | np.float64]:
        return self[self.key_timestamp]

    @property
    def is_sorted(self) -> np.bool_:
        timestamps = self.timestamps
        # trajectories do not allow duplicate timestamps, so >= is valid
        return np.all(timestamps[1:] >= timestamps[:-1])

    def sort(self, copy: bool = True) -> Self:
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
        return self._init_copy(data=data)

    def _window_to_slice(
        self,
        start: int | float,
        stop: int | float,
    ) -> slice:
        if self.length == 0:
            raise utils.OutOfInterval(
                "window slicing requires non zero-length trajectory"
            )
        if not self.is_sorted:
            raise ValueError(
                "window slicing requires sorted trajectory, call sort first."
            )
        timestamps = self.timestamps
        first = timestamps.min()
        last = timestamps.max()
        if start < first:
            raise utils.OutOfInterval(
                f"start: {start} not in trajectory range: [{first} {last}]"
            )
        if stop > last:
            raise utils.OutOfInterval(
                f"stop: {stop} not in trajectory range: [{first} {last}]"
            )
        return utils.get_interval_slice(timestamps, start, stop)

    def slice_window(
        self,
        start: int | float,
        stop: int | float,
    ) -> Self:
        return self[self._window_to_slice(start, stop)]
