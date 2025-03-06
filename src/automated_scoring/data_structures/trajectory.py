from collections.abc import Mapping
from math import isclose
from typing import Optional, Self

import numpy as np
from numpy.typing import NDArray

from .. import config, series_operations
from . import utils
from .timestamped_collection import TimestampedInstanceCollection


class Trajectory(TimestampedInstanceCollection):
    """
    Represents a trajectory, a specialized TimestampedInstanceCollection with a defined timestep.

    This class extends `TimestampedInstanceCollection` and adds functionality specific to trajectories, such as interpolation and sampling at specific timestamps. It also manages a timestep attribute.

    Parameters
    ----------
    data : Mapping of str and numpy.ndarray, optional
        A dictionary containing the trajectory data (default is None).
    cfg : config.Config, optional
        The configuration object (default is None).
    timestep : int or float, optional
        The timestep of the trajectory (default is None).
    validate_on_init : bool, optional
        Whether to validate the data during initialization (default is True).
    """

    _timestep: Optional[int | float]

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
    ) -> Self:
        """
        Initializes a new Trajectory object, optionally copying the configuration.

        Parameters
        ----------
        data : Optional[dict[str, NDArray]]
            The data to initialize the trajectory with.
        copy_config : bool, optional
            Whether to copy the configuration, defaults to False.
        validate_on_init : bool, optional
            Whether to validate the trajectory data during initialization, defaults to False.

        Returns
        -------
        Trajectory
            A new Trajectory object.
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

    def validate_data(  # type: ignore
        self,
        data: Mapping[str, utils.Value],
        *,
        allow_missing_keys: bool = False,
        try_broadcasting: bool = True,
        require_array_like=False,
    ) -> bool:
        """
        Validates the input data against the expected data structure.

        Parameters
        ----------
        data : Mapping[str, utils.Value]
            The data to validate.
        allow_missing_keys : bool, optional
            Whether to allow missing keys in the data, defaults to False.
        try_broadcasting : bool, optional
            Whether to attempt broadcasting of the data, defaults to True.
        require_array_like : bool, optional
            Whether to require array-like data structures, defaults to False.

        Returns
        -------
        bool
            True if the data is valid, False otherwise.
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
        Checks if the trajectory is complete based on its length, timestamps, and timestep.

        Returns
        -------
        bool
            True if the trajectory is complete, False otherwise.
        """
        if self.length <= 1:
            return True
        timestamps = self.timestamps
        duration = timestamps.max() - timestamps.min()
        return isclose(duration, (self.length - 1) * self.timestep)

    @property
    def timestep(self) -> int | float:
        """
        Gets or sets the timestep of the trajectory.

        Parameters
        ----------
        timestep : int or float, optional
            The timestep value to set. Defaults to None.

        Returns
        -------
        int or float
            The timestep of the trajectory.
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
    ) -> Self:
        """
        Samples the trajectory at the given timestamps.

        This method samples the trajectory data at the specified timestamps, interpolating values where necessary.

        Parameters
        ----------
        timestamps : numpy.ndarray[numpy.int64 | numpy.float64]
            The timestamps at which to sample the trajectory.
        keep_dtype : bool, optional
            Whether to preserve the original data type of the timestamps. Defaults to False.
        copy : bool, optional
            Whether to return a new Trajectory object or modify the existing one. Defaults to True.

        Returns
        -------
        Trajectory
            A new Trajectory object containing the sampled data.

        Raises
        ------
        AssertionError
            If the trajectory is not sorted or if it contains more than one unique identity.
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
        Calculates the interpolated length of the trajectory based on a given timestep.

        Parameters
        ----------
        timestep : int or float, optional
            The time step to use for interpolation; defaults to the trajectory's timestep if not provided.

        Returns
        -------
        int
            The interpolated length (number of instances) of the trajectory.

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
    ) -> Self:
        """
        Interpolates the trajectory to a new timestep.

        This method interpolates the trajectory data to a new timestep, effectively resampling the trajectory at a different rate. The interpolation is performed using linear interpolation.

        Parameters
        ----------
        timestep : int | float | None, optional
            The desired timestep for the interpolated trajectory. If None, the original timestep is used. Defaults to None.
        copy : bool, optional
            Whether to create a copy of the trajectory data. Defaults to True.

        Returns
        -------
        Trajectory
            The interpolated trajectory.
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
    ) -> Self:
        """
        Slices a window of the trajectory, optionally interpolating to a fixed timestep.

        Parameters
        ----------
        start : int | float
            The start time of the window.
        stop : int | float
            The stop time of the window.
        copy : bool, optional
            Whether to return a copy of the window, defaults to True.
        interpolate : bool, optional
            Whether to interpolate the window to a fixed timestep, defaults to True.
        interpolation_timestep : int | float | None, optional
            The timestep to use for interpolation, defaults to None.

        Returns
        -------
        Trajectory
            A new Trajectory object representing the sliced window.

        Raises
        ------
        ValueError
            If `interpolate` is True and `copy` is False.
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
