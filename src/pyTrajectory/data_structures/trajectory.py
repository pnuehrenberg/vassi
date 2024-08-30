from math import isclose
from typing import Mapping, Optional, Self

import numpy as np
from numpy.typing import NDArray

from .. import config, series_operations
from . import utils
from .timestamped_collection import TimestampedInstanceCollection


class Trajectory(TimestampedInstanceCollection):
    _timestep: Optional[int | float] = None

    def __init__(
        self,
        *,
        data: Optional[Mapping[str, NDArray]] = None,
        cfg: Optional[config.Config] = None,
        timestep: Optional[int | float] = None,
    ) -> None:
        super().__init__(data=data, cfg=cfg)
        self._timestep = timestep

    def _init_copy(
        self, *, data: Optional[dict[str, NDArray]], copy_config: bool = True
    ) -> Self:
        cfg = self.cfg
        if copy_config:
            cfg = cfg.copy()
        return type(self)(
            data=data,
            cfg=cfg,
            timestep=self._timestep,
        )

    @property
    def cfg(
        self,
    ) -> config.Config:
        return super().cfg

    def validate_data(  # type: ignore
        self,
        data: Mapping[str, utils.Value],
        *,
        allow_missing_keys: bool = False,
        try_broadcasting: bool = True,
        require_array_like=False,
    ) -> bool:
        return super().validate_data(
            data,
            allow_duplicated_timestamps=False,
            allow_missing_keys=allow_missing_keys,
            try_broadcasting=try_broadcasting,
            require_array_like=require_array_like,
        )

    @property
    def is_complete(self) -> bool:
        if self.length <= 1:
            return True
        timestamps = self.timestamps
        duration = timestamps.max() - timestamps.min()
        return isclose(duration, (self.length - 1) * self.timestep)

    @property
    def timestep(self) -> int | float:
        if self._timestep is not None:
            return self._timestep
        if self.cfg.timestep is not None:
            return self.cfg.timestep
        timestamps = self.timestamps
        unique_timesteps = np.unique(np.diff(timestamps))
        timestep = utils.greatest_common_denominator(unique_timesteps)
        is_int = issubclass(timestamps.dtype.type, np.integer)
        if is_int:
            timestep = int(timestep)
        return timestep

    @timestep.setter
    def timestep(self, timestep: Optional[int | float]) -> None:
        self._timestep = timestep

    def sample(
        self,
        timestamps: NDArray[np.int64 | np.float64],
        *,
        keep_dtype: bool = False,
        copy: bool = True,
    ) -> Self:
        if not self.is_sorted:
            raise AssertionError("can only sample sorted trajectory")
        if keep_dtype:
            timestamps = np.asarray(timestamps, dtype=self.timestamps.dtype)
        else:
            timestamps = np.asarray(timestamps)
        exclude = [self.key_timestamp]
        identity = None
        if (
            key_identity := self.cfg.key_identity
        ) is not None and key_identity in self.keys():
            identities = self[key_identity]
            if len(np.unique(identities)) > 1:
                raise AssertionError(
                    "can only sample trajectory with exactly one unique identity"
                )
            identity = (identities[0], identities.dtype)
            exclude.append(key_identity)
        data = {
            key: series_operations.sample(
                self[key],
                self.timestamps,
                timestamps,
                keep_dtype=keep_dtype,
            )
            for key in self.keys(exclude=exclude)
        }
        data[self.key_timestamp] = timestamps
        if key_identity is not None and identity is not None:
            data[key_identity] = np.repeat(identity[0], timestamps.shape[0]).astype(
                identity[1]
            )
        if not copy:
            self.data = data
            return self
        return self._init_copy(data=data)

    def interpolate(
        self,
        timestep: int | float | None = None,
        *,
        copy: bool = True,
    ) -> Self:
        timestamps = self.timestamps
        first = timestamps.min()
        last = timestamps.max()
        if timestep is None:
            timestep = self.timestep
        length = 1 + (last - first) / timestep
        if not isclose(length, np.round(length)):
            raise ValueError(
                f"timestep should result in an integer trajectory length and not: {length}"
            )
        timestamps = np.linspace(
            first,
            last,
            int(np.round(length)),
        )
        trajectory = self.sample(timestamps, copy=copy)
        trajectory.timestep = timestep
        return trajectory

    def slice_window(
        self,
        start: int | float,
        stop: int | float,
        *,
        copy: bool = True,
        interpolate: bool = True,
        interpolation_timestep: int | float | None = None,
    ) -> Self:
        if not interpolate:
            window_view = super().slice_window(start, stop)
            if copy:
                return window_view.copy()
            return window_view
        if not copy:
            raise ValueError("cannot slice window as view with interpolate=True")
        slice_key = super()._window_to_slice(start, stop)
        if self[slice_key.start][self.key_timestamp] > start:
            slice_key = slice(
                max(0, slice_key.start - 1),
                slice_key.stop,
            )
        if slice_key.stop == 0:
            slice_key = slice(
                slice_key.start,
                self.length,
            )
        elif self[slice_key.stop - 1][self.key_timestamp] < stop:
            slice_key = slice(
                slice_key.start,
                min(self.length, slice_key.stop + 1),
            )
        trajectory_window = self[slice_key]
        trajectory_window.timestep = interpolation_timestep
        if not trajectory_window.is_complete:
            trajectory_window = trajectory_window.interpolate(
                timestep=interpolation_timestep
            )
        if isclose(
            float(trajectory_window[0][self.key_timestamp]),
            start,
        ) and isclose(
            float(trajectory_window[-1][self.key_timestamp]),
            stop,
        ):
            return trajectory_window
        return trajectory_window.slice_window(
            start,
            stop,
            interpolate=False,
            interpolation_timestep=interpolation_timestep,
        )

    # def select(self, timestamp: int | float, *, copy: bool = True) -> instance.Instance:  # type: ignore
    #     selection = super().select(timestamp)
    #     if len(selection) == 0:
    #         raise ValueError(
    #             f"{timestamp} not in timestamps. Use Trajectory.sample([{timestamp}]) instead."
    #         )
    #     elif len(selection) == 1:
    #         return selection[0]
    #     # a trajectory cannot have duplicate timestamps
    #     raise AssertionError
