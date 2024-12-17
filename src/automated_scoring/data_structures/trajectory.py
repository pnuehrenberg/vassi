from math import isclose
from typing import Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from .. import config, series_operations
from . import utils
from .timestamped_collection import TimestampedInstanceCollection


class Trajectory(TimestampedInstanceCollection):
    """
    Trajectory data structure.

    Parameters
    ----------
    data: Mapping[str, NDArray], optional
        Data of the instances.

    cfg: Config, optional
        Configuration of the instances.

    timestep: int | float, optional
        Timestep of the trajectory.

    validate_on_init: bool, optional
        Whether to validate the data on initialization.
    """
    _timestep: Optional[int | float] = None

    def __init__(
        self,
        *,
        data: Optional[Mapping[str, NDArray]] = None,
        cfg: Optional[config.Config] = None,
        timestep: Optional[int | float] = None,
        validate_on_init: bool = True,
    ) -> None:
        super().__init__(data=data, cfg=cfg, validate_on_init=validate_on_init)
        self._timestep = timestep

    def _init_other(
        self,
        *,
        data: Optional[dict[str, NDArray]],
        copy_config: bool = False,
        validate_on_init: bool = False,
    ) -> "Trajectory":
        """
        Initialize a new instance of the same type.

        Parameters
        ----------
        data: Mapping[str, NDArray]
            Data of the new collection.
        copy_config: bool, optional
            Whether to copy the configuration.
        validate_on_init: bool, optional
            Whether to validate the data on initialization.

        Returns
        -------
        Trajectory
            The new instance.
        """
        cfg = self.cfg
        if copy_config:
            cfg = cfg.copy()
        return type(self)(
            data=data,
            cfg=cfg,
            timestep=self._timestep,
            validate_on_init=validate_on_init,
        )

    @property
    def cfg(
        self,
    ) -> config.Config:
        """
        Configuration of the trajectory.
        """
        return super().cfg

    def validate_data(  # type: ignore
        self,
        data: Mapping[str, utils.Value],
        *,
        allow_missing_keys: bool = False,
        try_broadcasting: bool = True,
        require_array_like=False,
    ) -> bool:
        """
        Validates data to be compatible with the trajectory.

        Parameters
        ----------
        data: Mapping[str, NDArray | int | float]
            Data to validate.

        allow_missing_keys: bool, optional
            Whether to allow missing keys.

        try_broadcasting: bool, optional
            Whether to try broadcasting the data.

        require_array_like: bool, optional
            Whether to require the data to be array-like.

        Returns
        -------
        bool
            Whether the data is valid.
        """
        return super().validate_data(
            data,
            allow_duplicated_timestamps=False,
            allow_missing_keys=allow_missing_keys,
            try_broadcasting=try_broadcasting,
            require_array_like=require_array_like,
        )

    @property
    def is_complete(self) -> bool:
        """
        Whether the trajectory is complete.
        """
        if self.length <= 1:
            return True
        timestamps = self.timestamps
        duration = timestamps.max() - timestamps.min()
        return isclose(duration, (self.length - 1) * self.timestep)

    @property
    def timestep(self) -> int | float:
        """
        Timestep of the trajectory.

        If the timestep is not specified in the configuration or the trajectory,
        it is determined by the smallest common denominator of the differences of
        the timestamps.
        """
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
    ) -> "Trajectory":
        """
        Sample the trajectory at the specified timestamps using linear interpolation.

        Parameters
        ----------
        timestamps: NDArray[int | float]
            Timestamps to sample at.

        keep_dtype: bool, optional
            Whether to keep the data type of the timestamps.

        copy: bool, optional
            Whether to copy the trajectory before sampling.

        Returns
        -------
        Trajectory
            Sampled trajectory.
        """
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
        return self._init_other(data=data)

    def get_interpolated_length(self, timestep: Optional[int | float] = None) -> int:
        """
        Calculate the length of the trajectory after interpolation.

        Parameters
        ----------
        timestep: int | float, optional
            Timestep to use for interpolation.

        Returns
        -------
        int
            Length of the interpolated trajectory.

        Raises
        ------
        ValueError
            If the timestep does not result in an integer trajectory length.
        """
        if timestep is None:
            timestep = self.timestep
        interpolated_length = 1 + (self.timestamps[-1] - self.timestamps[0]) / timestep
        if not isclose(interpolated_length, np.round(interpolated_length)):
            raise ValueError(
                f"timestep should result in an integer trajectory length and not: {interpolated_length}"
            )
        return int(np.round(interpolated_length))

    def interpolate(
        self,
        timestep: int | float | None = None,
        *,
        copy: bool = True,
    ) -> "Trajectory":
        """
        Interpolate the trajectory with the specified timestep.

        Parameters
        ----------
        timestep: int | float, optional
            Timestep to use for interpolation.

        copy: bool, optional
            Whether to copy the trajectory before interpolation.

        Returns
        -------
        Trajectory
            Interpolated trajectory.
        """
        interpolated_length = self.get_interpolated_length(timestep)
        timestamps = np.linspace(
            self.timestamps[0],
            self.timestamps[-1],
            interpolated_length,
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
    ) -> "Trajectory":
        """
        Slice a timewindow of the trajectory.

        Parameters
        ----------
        start: int | float
            Start of the timewindow.
        stop: int | float
            End of the timewindow.
        copy: bool
            Whether to return a copy of the window.
        interpolate: bool
            Whether to interpolate the window.
        interpolation_timestep: int | float | None
            If interpolate is True, the timestep to interpolate to.

        Returns
        -------
        Trajectory
            A sliced timewindow of the trajectory.
        """
        if not interpolate:
            window_view = super().slice_window(start, stop)
            if copy:
                return window_view.copy()
            return window_view
        if not copy:
            raise ValueError("cannot slice window as view with interpolate=True")
        slice_key = super()._window_to_slice(start, stop)
        if self[self.key_timestamp][slice_key.start] > start:
            slice_key = slice(
                max(0, slice_key.start - 1),
                slice_key.stop,
            )
        if slice_key.stop == 0:
            slice_key = slice(
                slice_key.start,
                self.length,
            )
        elif self[self.key_timestamp][slice_key.stop - 1] < stop:
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
