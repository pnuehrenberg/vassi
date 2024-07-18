"""A module for representing and manipulating trajectory data, implementing the Trajectory class."""

import numpy as np

import pyTrajectory.config
import pyTrajectory.series_operations
import pyTrajectory.instance

from typing import overload, TypeVar, Self, Any
from numpy.typing import NDArray

from math import isclose, gcd
from fractions import Fraction
from functools import reduce
from copy import deepcopy


class OutOfInterval(Exception):
    """Indicates that a trajectory was sampled outside of the trajectory interval between the first and last timestamp."""

    pass


TrajectoryType = TypeVar("Trajectory")


def validate_keys(
    data: dict[str, Any],
    data_reference: dict[str, Any] | TrajectoryType,
) -> None:
    """Validate data keys against a reference data or trajectory.

    Parameters
    ----------
    data : dict[str, Any]
        The data with keys to validate.
    data_reference : dict[str, Any] | TrajectoryType
        The reference data or trajectory.

    Raises
    ------
    KeyError
        If the keys do not match.

    Examples
    --------
    >>> validate_keys(
    ...     {"position": None, "time": None}, {"position": None, "time": None}
    ... )

    >>> validate_keys(
    ...     {"position": None, "frame_idx": None}, {"position": None, "time": None}
    ... )
    Traceback (most recent call last):
        ...
    KeyError: 'data keys do not match'

    >>> validate_keys(
    ...     {"position": None, "position": None},
    ...     {"position": None, "time": None, "posture": None},
    ... )
    Traceback (most recent call last):
        ...
    KeyError: 'data keys do not match'

    >>> validate_keys(
    ...     {"position": None, "time": None},
    ...     Trajectory(data={"position": None, "time": None}),
    ... )

    """
    if set(data.keys()) != set(data_reference.keys()):
        raise KeyError("data keys do not match")


def validate_data_lengths(
    data: dict[str, NDArray[Any] | int | float | None],
) -> int | None:
    """Validate data value lengths.

    Parameters
    ----------
    data : dict[str, NDArray[Any] | int | float | None]
        The data for which the lengths should be validated.

    Returns
    -------
    int | None
        The validated data length, or None if all data values are None.

    Raises
    ------
    ValueError
        If some non-None values have mismatched lengths.

    Examples
    --------
    >>> data = {
    ...     "time": np.array([0, 1, 2]),
    ...     "position": np.array([[0, 0], [1, 1], [2, 2]]),
    ...     "posture": None,
    ... }
    >>> validate_data_lengths(data)
    3

    >>> data = {
    ...     "time": np.array([0, 1, 2]),
    ...     "position": np.array([[0, 0], [1, 1]]),
    ...     "posture": None,
    ... }
    >>> validate_data_lengths(data)
    Traceback (most recent call last):
        ...
    ValueError: non-None values mave mismatched lengths

    >>> validate_data_lengths({"position": None}) is None
    True
    >>> validate_data_lengths({"position": None, "time": None}) is None
    True
    """
    lengths = set(
        [
            len(value)
            for value in data.values()
            if value is not None
            and not isinstance(value, int | float | np.integer | np.floating)
        ]
    )
    if len(lengths) > 1:
        raise ValueError("non-None values mave mismatched lengths")
    if len(lengths) == 0:
        return None
    return lengths.pop()


def validate_timestamps(
    timestamps: NDArray[int | float] | None,
) -> None:
    """Validate trajectory timestamps.

    Parameters
    ----------
    timestamps : NDArray[int | float]
        The timestamps to check. Should be a 1-dimensional numpy array without duplicate values.

    Raises
    ------
    ValueError
        If the timestamps are not 1-dimensional or contain duplicated values.

    Examples
    --------
    >>> timestamps = np.array([0, 1, 2])
    >>> validate_timestamps(timestamps)

    >>> timestamps = np.array([[0], [1], [2]])
    >>> validate_timestamps(timestamps)
    Traceback (most recent call last):
        ...
    ValueError: timestamps are not 1-dimensional

    >>> timestamps = np.array([0, 1, 1])
    >>> validate_timestamps(timestamps)
    Traceback (most recent call last):
        ...
    ValueError: data contains duplicated timestamps
    """
    if timestamps is None:
        return
    if timestamps.ndim != 1:
        raise ValueError("timestamps are not 1-dimensional")
    counts = np.unique(timestamps, return_counts=True)[1]
    if np.any(counts > 1):
        raise ValueError("data contains duplicated timestamps")


def greatest_common_denominator(
    values: list[int | float] | NDArray[int | float], return_inverse: bool = True
):
    """Find the greatest common denominator for a list of ints or floats.

    https://stackoverflow.com/questions/44587875/find-common-factor-to-convert-list-of-floats-to-list-of-integers

    Parameters
    ----------
    values : list[int | float] | NDArray[int | float]
        The values for which to find the greatest common denominator.
    return_inverse : bool, optional
        Whether to return the inverse of the denominator (default).

    Returns
    -------
    float
        The greatest common denominator or the inverse.

    Examples
    --------
    >>> greatest_common_denominator([2, 4, 8])
    1.0
    >>> greatest_common_denominator([0.1, 1, 0.2])
    0.1
    >>> greatest_common_denominator([0.3333334, 0.3333333, 0.6666667])
    0.3333333333333333
    >>> # numbers should be precise enough to get expected result
    >>> greatest_common_denominator([0.3334, 0.3333, 0.6667])
    0.0001
    >>> values = np.linspace(0, 1, 12)
    >>> values[1] *= 2
    >>> values[-1] *= 4
    >>> greatest_common_denominator(values)
    0.09090909090909091
    """
    denominators = [Fraction(value).limit_denominator().denominator for value in values]
    common_denominator = reduce(lambda a, b: a * b // gcd(a, b), denominators)
    return 1 / common_denominator


class Trajectory:
    """A class representing a trajectory consisting of timestamped data points.

    The Trajectory class provides a convenient way to store and manipulate time-series data
    with associated timestamps. It supports operations like sampling, interpolation,
    slicing, and sorting of trajectory data.

    Parameters
    ----------
    data : dict, optional
        A dictionary containing the trajectory data. The keys represent different
        data types (e.g., "position", "posture"), and the values are NumPy arrays
        representing the corresponding data points or None.
    cfg : pyTrajectory.config.Config, optional
        An instance of the TrajectoryConfig class specifying configuration options
        for the trajectory.
    timestep : int | float, optional
        The timestep of the trajectory timestamps. If not provided, the timestep is
        inferred from the data.

    Attributes
    ----------
    data : dict
        A dictionary containing the trajectory data.
    cfg : pyTrajectory.config.TrajectoryConfig
        The trajectory configuration.
    timestep : int | float
        The timestep of the trajectory timestamps.

    Examples
    --------
    >>> import pyTrajectory.config
    >>> cfg = pyTrajectory.config.cfg.copy()
    >>> cfg.trajectory_keys = ("position", "time")

    Create a trajectory with position and time data

    >>> data = {
    ...     "position": np.array([[0, 0], [1, 1], [2, 2]]),
    ...     "time": np.array([0, 1, 2]),
    ... }
    >>> trajectory = Trajectory(data=data, cfg=cfg)

    Accessing trajectory data

    >>> trajectory["position"]
    array([[0, 0],
           [1, 1],
           [2, 2]])
    >>> trajectory["time"]
    array([0, 1, 2])

    Interpolating the trajectory

    >>> interpolated_trajectory = trajectory.interpolate()

    Sampling the trajectory at specific timestamps

    >>> sampled_trajectory = trajectory.sample(np.array([0.5, 1.5, 2.5]))

    Slicing the trajectory window between start and stop timestamps

    >>> window = trajectory.slice_window(start=0, stop=1)

    Sorting the trajectory by timestamps

    >>> sorted_trajectory = trajectory.sort()

    Copying the trajectory

    >>> copied_trajectory = trajectory.copy()

    Getting the timestep of the trajectory

    >>> timestep = trajectory.timestep

    Checking if the trajectory is complete

    >>> is_complete = trajectory.is_complete

    Checking if the timestamps are sorted

    >>> is_sorted = trajectory.is_sorted

    """

    def __init__(
        self,
        data: dict[str, NDArray[Any] | None] | None = None,
        cfg: pyTrajectory.config.Config | None = None,
        timestep: int | float | None = None,
    ) -> None:
        """Construct a trajectory object.

        Parameters
        ----------
        data : dict[str, NDArray[Any] | None], optional
            The trajectory data to use, defaults to None.
        cfg : pyTrajectory.config.Config | None, optional
            The configuration to use, defaults to None.
        timestep : int | float | None, optional
            The trajectory timestep, defaults to None.

        See Also
        --------
        Trajectory.data
            Set or get the trajectory data.
        Trajectory.cfg
            Set or get the trajectory configuration.
        Trajectory.timestep
            Set, get or infer the trajectory timestep.

        Examples
        --------
        >>> import pyTrajectory.config
        >>> cfg = pyTrajectory.config.cfg
        >>> custom_cfg = cfg.copy()
        >>> custom_cfg.trajectory_keys = ("time", "position", "posture")
        >>> data = {
        ...     "time": np.array([0, 1, 2]),
        ...     "position": np.array([[0, 0], [1, 1], [2, 2]]),
        ...     "posture": None,
        ... }
        >>> trajectory = Trajectory(data=data, cfg=custom_cfg, timestep=0.1)
        """
        self._timestep = timestep
        self._cfg = cfg
        self._data = {key: None for key in self.keys()}
        if data is not None:
            self.data = data

    @property
    def cfg(
        self,
    ) -> pyTrajectory.config.Config:
        """Configuration object for the trajectory.

        If None, the default configuration object from pyTrajectory.config is used.

        Parameters
        ----------
        cfg : pyTrajectory.config.Config | None
            Configuration object for the trajectory. If None, the default configuration
            object from pyTrajectory.config is used.

        Returns
        -------
        cfg : pyTrajectory.config.Config | None
            The configuration of the trajectory.

        Example
        -------
        >>> import pyTrajectory.config
        >>> trajectory = Trajectory()

        >>> trajectory.cfg = None
        >>> trajectory.cfg is pyTrajectory.config.cfg
        True

        >>> cfg = pyTrajectory.config.cfg.copy()
        >>> trajectory.cfg = cfg
        >>> trajectory.cfg is cfg
        True
        """
        if self._cfg is None:
            return pyTrajectory.config.cfg
        return self._cfg

    @cfg.setter
    def cfg(
        self,
        cfg: pyTrajectory.config.Config | None,
    ):
        """Setter for the cfg property."""
        self._cfg = cfg

    def keys(
        self,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Return a list of keys present in the trajectory data.

        Parameters
        ----------
        exclude : list[str] | None, optional
            List of keys to exclude, by default None

        Returns
        -------
        list[str]
            List of keys present in the trajectory data

        Example
        -------
        >>> import pyTrajectory.config
        >>> cfg = pyTrajectory.config.cfg

        >>> trajectory = Trajectory()
        >>> keys = trajectory.keys()
        >>> set(keys) == set(cfg.trajectory_keys)
        True

        >>> cfg = pyTrajectory.config.cfg.copy()
        >>> cfg.trajectory_keys = ("time", "position", "posture")
        >>> trajectory = Trajectory(cfg=cfg)
        >>> keys = trajectory.keys(exclude=["posture"])
        >>> "posture" not in keys
        True
        """
        if exclude is None:
            exclude = []
        return [key for key in self.cfg.trajectory_keys if key not in exclude]

    def values(
        self,
        exclude: list[str] | None = None,
        copy: bool = True,
    ) -> list[NDArray[Any] | None]:
        """Return a list of values corresponding to the keys present in the trajectory data.

        Parameters
        ----------
        exclude : list[str] | None, optional
            List of keys to exclude, by default None
        copy : bool, optional
            Whether to return copies of the values, by default True

        Returns
        -------
        list[NDArray[Any] | None]
            List of values present in the trajectory data

        Example
        -------
        >>> import pyTrajectory.config
        >>> cfg = pyTrajectory.config.cfg

        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> values = trajectory.values(copy=False)  # read-only!
        >>> len(values) == len(cfg.trajectory_keys)
        True
        >>> timestamps = values[0]
        >>> timestamps[-1] += 1
        Traceback (most recent call last):
            ...
        ValueError: assignment destination is read-only

        >>> values = trajectory.values(exclude=["position"], copy=True)
        >>> len(values) == len(trajectory.keys()) - 1
        True
        >>> timestamps = values[0]
        >>> timestamps[-1] += 1
        >>> np.array_equal(timestamps, trajectory["time"])
        False
        """
        return [self.get_value(key, copy=copy) for key in self.keys(exclude=exclude)]

    def items(
        self,
        exclude: list[str] | None = None,
        copy: bool = True,
    ) -> list[tuple[str, NDArray[Any] | None]]:
        """Return a list of (key, value) tuples corresponding to the keys present in the trajectory data.

        Parameters
        ----------
        exclude : list[str] | None, optional
            List of keys to exclude, by default None
        copy : bool, optional
            Whether to return copies of the values, by default True

        Returns
        -------
        list[tuple[str, NDArray[Any] | None]]
            List of (key, value) tuples present in the trajectory data

        Example
        -------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> items = trajectory.items(copy=False)  # read-only!
        >>> set([key for key, value in items]) == {"time", "position"}
        True
        >>> data = {key: value for key, value in items}
        >>> data["time"][-1] += 1
        Traceback (most recent call last):
            ...
        ValueError: assignment destination is read-only

        >>> items = trajectory.items(exclude=["position"], copy=True)
        >>> len(items) == len(trajectory.keys()) - 1
        True
        >>> data = {key: value for key, value in items}
        >>> timestamps = data["time"]
        >>> timestamps[-1] += 1
        >>> np.array_equal(timestamps, trajectory["time"])
        False
        """
        keys = self.keys(exclude=exclude)
        values = self.values(exclude=exclude, copy=copy)
        return [(key, value) for key, value in zip(keys, values)]

    @property
    def length(self) -> int:
        """Return the length of the trajectory.

        Returns
        -------
        int
            Length of the trajectory

        Example
        -------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> trajectory.length == len(trajectory["time"]) == len(trajectory["position"])
        True

        >>> trajectory["position"] = None
        >>> trajectory.length == len(trajectory["time"])
        True
        """
        lengths = [len(value) for value in self.values(copy=False) if value is not None]
        if len(lengths) == 0:
            return 0
        return lengths[0]  # data is validated, so no need to look at all lengths

    def __len__(self) -> int:
        """Return the length of the trajectory.

        Returns
        -------
        int
            Length of the trajectory

        Example
        -------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> len(trajectory) == trajectory.length
        True

        >>> trajectory["position"] = None
        >>> len(trajectory) == trajectory.length
        True
        """
        return self.length

    def validate_data(self, data: dict[str, NDArray[Any] | None]) -> bool:
        """Validate the provided trajectory data.

        This method checks if the keys in the provided data match the keys expected
        in the trajectory object. It also ensures that the lengths of the data arrays
        are consistent with the existing trajectory data, and that timestamps are unique.

        Parameters
        ----------
        data : dict[str, NDArray[Any] | None]
            The trajectory data to be validated.

        Returns
        -------
        bool
            Whether the data is valid with respect to the trajectory.

        Raises
        ------
        KeyError
            If any key in the provided data does not match the keys expected in the
            trajectory object.
        ValueError
            If the lengths of the data arrays are inconsistent with the existing
            trajectory data.

        Example
        -------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> len(trajectory)
        3
        >>> set(trajectory.keys()) == {"time", "position"}
        True

        Valid data has the same length as the trajectory object and only defined keys. It
        can also contain None values or only a subset of defined keys.

        >>> data = {
        ...     "position": np.array([[0, 0], [1, 1], [2, 2]]),
        ...     "time": np.array([1, 2, 3]),
        ... }
        >>> trajectory.validate_data(data)
        True

        >>> data = {"position": None, "time": np.array([1, 2, 3])}
        >>> trajectory.validate_data(data)
        True

        >>> data = {"position": np.array([[0, 0], [1, 1], [2, 2]])}
        >>> trajectory.validate_data(data)
        True

        If all defined keys are used, also data with a different length is valid. If only
        a subset is used, a ValueError will be raised when data and trajectory length do not match.

        >>> data = {
        ...     "position": np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
        ...     "time": np.array([1, 2, 3, 4]),
        ... }
        >>> trajectory.validate_data(data)
        True

        >>> data = {"time": np.array([10, 11, 12, 13])}
        >>> trajectory.validate_data(data)
        Traceback (most recent call last):
            ...
        ValueError: data length (4) does not match trajectory length (3)

        Using undefined keys raises a KeyError:

        >>> data = {
        ...     "location": np.array([[0, 0], [1, 1], [2, 2]]),
        ...     "time": np.array([1, 2, 3]),
        ... }
        >>> trajectory.validate_data(data)
        Traceback (most recent call last):
            ...
        KeyError: 'key: location is not defined'

        Inconsistent data lengths and duplicated timestamps raise ValueErrors.

        >>> data = {"position": np.array([[0, 0], [1, 1]]), "time": np.array([1, 2, 3])}
        >>> trajectory.validate_data(data)
        Traceback (most recent call last):
            ...
        ValueError: non-None values mave mismatched lengths

        >>> data = {
        ...     "position": np.array([[0, 0], [1, 1], [2, 2]]),
        ...     "time": np.array([1, 2, 2]),
        ... }
        >>> trajectory.validate_data(data)
        Traceback (most recent call last):
            ...
        ValueError: data contains duplicated timestamps
        """
        if data is None:
            return True
        for key in list(set(data.keys()).difference(self.keys())):
            raise KeyError(f"key: {key} is not defined")
        if self.cfg.key_timestamp in data.keys():
            validate_timestamps(data[self.cfg.key_timestamp])
        data_length = validate_data_lengths(data)
        try:
            validate_keys(data, self)
            return (
                True  # if all defined keys are used, allow trajectory to be overwritten
            )
        except KeyError:
            pass
        # otherwise, data length should match trajectory length
        length = self.length
        if data_length is None:
            return True  # allow setting keys to None (when all values in data are None)
        if data_length != length:
            raise ValueError(
                f"data length ({data_length}) does not match trajectory length ({length})"
            )  #
        return True

    @property
    def data(self) -> dict[str, NDArray[Any] | None]:
        """Get or set a dictionary containing the trajectory data.

        Parameters
        ----------
        data : dict[str, NDArray[Any] | None]
            The trajectory data to set.

        Returns
        -------
        dict[str, NDArray[Any] | None]
            The trajectory data.

        Raises
        ------
        KeyError
            If any key in the provided data does not match the keys expected in the
            trajectory object.
        ValueError
            If the lengths of the data arrays are inconsistent with the existing
            trajectory data or if the data contains duplicated timestamps.

        See Also
        --------
        Trajectory.validate_data : Validate the provided trajectory data.

        Example
        -------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        Accessing the trajectory data:

        >>> trajectory.data["time"]
        array([1, 2, 4])
        >>> trajectory.data["position"]
        array([[1, 1],
               [2, 2],
               [4, 4]])

        Modifying the returned data will not update the trajectory:

        >>> modified_data = trajectory.data
        >>> modified_data["time"] = np.array([0, 1, 2])
        >>> trajectory.data["time"]
        array([1, 2, 4])

        >>> trajectory = Trajectory()
        >>> trajectory.data = data
        >>> trajectory.data["time"]
        array([1, 2, 4])
        >>> trajectory.data["position"]
        array([[1, 1],
               [2, 2],
               [4, 4]])

        Attempting to set invalid data:

        >>> invalid_data = {
        ...     "time": data["time"],
        ...     "posture": [[[0, 0]], [[1, 1]], [[2, 2]]],
        ... }
        >>> trajectory.data = invalid_data
        Traceback (most recent call last):
            ...
        KeyError: 'data keys do not match'
        """
        return {key: value for key, value in self.items()}

    @data.setter
    def data(self, data: dict[str, NDArray[Any] | None]) -> None:
        """Set the trajectory data."""
        validate_keys(data, self)
        self.validate_data(data)
        self._data = {key: None for key in self.keys()}
        if data is None:
            return
        for key, value in data.items():
            self.set_value(
                key,
                value,
                validate=False,
            )  # no need to validate again

    def set_value(
        self,
        key: str,
        value: NDArray[Any] | None,
        validate: bool = True,
    ) -> None:
        """Set the value for a specific key in the trajectory data.

        Parameters
        ----------
        key : str
            The key corresponding to the data type.
        value : ArrayLike | None
            The value to be assigned to the key.
        validate : bool, optional
            Whether to validate the provided data, by default True

        See Also
        --------
        Trajectory.__setitem__ : Set values using indexing notation and slicing.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        This is to demonstrate the method, however, the __setitem__ method should be preferred.

        >>> trajectory.set_value("position", np.array([[0, 0], [1, 1], [2, 2]]))
        >>> trajectory["position"]
        array([[0, 0],
               [1, 1],
               [2, 2]])

        >>> trajectory.set_value("position", np.array([[0, 0], [1, 1], [2, 2], [3, 3]]))
        Traceback (most recent call last):
            ...
        ValueError: data length (4) does not match trajectory length (3)

        >>> trajectory.set_value("time", np.array([0, 0, 1]))
        Traceback (most recent call last):
            ...
        ValueError: data contains duplicated timestamps

        >>> trajectory.set_value("position", None)
        >>> trajectory["position"] is None
        True

        Use broadcasting to set a single value.

        >>> trajectory["position"] = np.array([[0, 0], [1, 1], [2, 2]])
        >>> trajectory.set_value("position", 1)
        >>> trajectory["position"]
        array([[1, 1],
               [1, 1],
               [1, 1]])

        Or any other broadcasting. Note that data validation is not possible in most cases.

        >>> trajectory.set_value("position", [1, 2], validate=False)
        >>> trajectory["position"]
        array([[1, 2],
               [1, 2],
               [1, 2]])

        The preferred solution with data validation is using slicing syntax with the set_value_slice method.
        Aquivalent to the example above, but with data validation.

        >>> trajectory.set_value_slice(slice(None), "position", [1, 2], validate=True)
        >>> trajectory["position"]
        array([[1, 2],
               [1, 2],
               [1, 2]])
        """
        if validate:
            self.validate_data({key: value})
        if value is None:
            self._data[key] = value
            return
        if isinstance(value, np.ndarray):
            if not value.flags.owndata:
                value = value.copy()
            self._data[key] = value
        elif self._data[key] is not None:
            self._data[key].flags.writeable = True
            self._data[key][:] = value
        else:
            raise NotImplementedError(
                "broadcasting is only possible on non-None values"
            )
        self._data[key].flags.writeable = False

    def set_value_slice(
        self,
        slice_key: slice,
        key: str,
        value: NDArray[Any] | None | int | float,
        validate: bool = True,
    ) -> None:
        """Set a slice of values for a specific key in the trajectory data.

        Parameters
        ----------
        slice_key : slice
            The slice object specifying the indices to be replaced.
        key : str
            The key corresponding to the data type.
        value : ArrayLike | None | int | float
            The value to be assigned to the slice. Can use broadcasting.
        validate : bool, optional
            Whether to validate the provided data, by default True

        Raises
        ------
        ValueError
            If the value does not match the slice (either None or appropriate array-like object) or slice shape.

        See Also
        --------
        Trajectory.__setitem__ : Set values using indexing notation and slicing.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        This is to demonstrate the method, however, the __setitem__ method should be preferred.

        >>> trajectory.set_value_slice(
        ...     slice(1, 3), "position", np.array([[0, 0], [1, 1]])
        ... )
        >>> trajectory["position"]
        array([[1, 1],
               [0, 0],
               [1, 1]])

        >>> trajectory.set_value_slice(slice(0, 2), "position", None)
        Traceback (most recent call last):
            ...
        ValueError: value should match slice (either None or appropriate array-like)

        >>> trajectory.set_value_slice(
        ...     slice(0, 2), "position", np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        ... )
        Traceback (most recent call last):
            ...
        ValueError: could not broadcast input array from shape (4,2) into shape (2,2)

        >>> trajectory["position"] = None
        >>> trajectory.set_value_slice(slice(0, 2), "position", None)

        Use broadcasting to set slice to a single value or any other broadcasting.

        >>> trajectory["position"] = np.array([[0, 0], [1, 1], [2, 2]])
        >>> trajectory.set_value_slice(slice(None), "position", 1)
        >>> trajectory["position"]
        array([[1, 1],
               [1, 1],
               [1, 1]])

        >>> trajectory.set_value_slice(slice(1, None), "position", [1, 2])
        >>> trajectory["position"]
        array([[1, 1],
               [1, 2],
               [1, 2]])
        """
        is_value = value is not None
        has_value = self.get_value(key) is not None
        if is_value is not has_value:
            raise ValueError(
                "value should match slice (either None or appropriate array-like)"
            )
        data = {key: self.get_value(key, copy=True)}
        if data[key] is not None:
            data[key][slice_key] = value
        if validate:
            self.validate_data(data)
        if self._data[key] is None:
            return
        if isinstance(value, np.ndarray) and not value.flags.owndata:
            value = value.copy()
        self._data[key].flags.writeable = True
        self._data[key][slice_key] = value
        self._data[key].flags.writeable = False

    def set_instance(self, idx: int, instance: pyTrajectory.instance.Instance):
        """Set an instance at a specified index.

        Parameters
        ----------
        idx : int
            At which index to set the instance.
        instance : pyTrajectory.instance.Instance
            The instance to set

        Examples
        --------
        >>> from pyTrajectory.instance import Instance
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> instance = Instance(position=[3, 3], time=3)
        >>> trajectory.set_instance(-1, instance)
        >>> trajectory["time"]
        array([1, 2, 3])
        >>> trajectory["position"]
        array([[1, 1],
               [2, 2],
               [3, 3]])

        Data is always validated.

        >>> instance = Instance(position=[3, 3], time=2)
        >>> trajectory.set_instance(-1, instance)
        Traceback (most recent call last):
            ...
        ValueError: data contains duplicated timestamps

        >>> instance = Instance(position=[[3, 3], [4, 4]], time=3)
        >>> trajectory.set_instance(-1, instance)
        Traceback (most recent call last):
            ...
        ValueError: could not broadcast input array from shape (2,2) into shape (2,)
        """
        data = self.data
        for key in self.keys():
            if (data[key] is None) is not (instance[key] is None):
                raise ValueError(
                    "instance value should match trajectory value (either None or appropriate type)"
                )
            if instance[key] is None:
                continue
            data[key][idx] = instance[key]
        self.validate_data(data)
        for key in self.keys():
            value = instance[key]
            if value is None:
                continue
            if isinstance(value, np.ndarray) and not value.flags.owndata:
                value = value.copy()
            self._data[key].flags.writeable = True
            self._data[key][idx] = value
            self._data[key].flags.writeable = False

    def get_value(
        self,
        key: str,
        copy: bool = False,
    ) -> NDArray[Any] | None:
        """Get the value for a specific key in the trajectory data.

        Parameters
        ----------
        key : str
            The key corresponding to the data type.
        copy : bool, optional
            Whether to return a copy of the value, by default False

        Returns
        -------
        NDArray[Any] | None
            The value corresponding to the specified key. If `copy` is True, a copy
            of the value is returned.

        See Also
        --------
        Trajectory.__getitem__ : Access values using indexing notation and slicing.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[0, 0], [1, 1], [2, 2]]),
        ...     "time": np.array([1, 2, 3]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> trajectory.get_value("position")
        array([[0, 0],
               [1, 1],
               [2, 2]])
        >>> positions = trajectory.get_value("position", copy=True)
        >>> positions[-1] = [3, 3]
        >>> np.array_equal(positions, trajectory["position"])
        False

        By default, return trajectory value as read-only and not as a copy.

        >>> positions = trajectory.get_value("position")
        >>> positions[-1] = [3, 3]
        Traceback (most recent call last):
            ...
        ValueError: assignment destination is read-only
        """
        value = self._data[key]
        if value is None:
            return value
        if copy:
            return value.copy()
        return value

    def get_value_slice(
        self,
        slice_key: slice,
        key: str,
        copy: bool = False,
    ) -> NDArray[Any] | None:
        """Get a slice of the value for a specific key in the trajectory data.

        Parameters
        ----------
        slice_key : slice
            The slice object specifying the start, stop, and step for the slice.
        key : str
            The key corresponding to the data type.
        copy : bool, optional
            Whether to return a copy of the sliced value, by default False.

        Returns
        -------
        NDArray[Any] | None
            The sliced value corresponding to the specified key. If `copy` is True,
            a copy of the sliced value is returned.

        See Also
        --------
        Trajectory.__getitem__ : Access values using indexing notation and slicing.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[0, 0], [1, 1], [2, 2]]),
        ...     "time": np.array([1, 2, 3]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> trajectory.get_value_slice(slice(0, 2), "position")
        array([[0, 0],
               [1, 1]])

        >>> positions_slice = trajectory.get_value_slice(
        ...     slice(0, 2), "position", copy=True
        ... )
        >>> positions_slice[-1] = [3, 3]
        >>> np.array_equal(positions_slice, trajectory["position"][:2])
        False

        >>> positions_slice = trajectory.get_value_slice(
        ...     slice(0, 2), "position", copy=False
        ... )
        >>> positions_slice[-1] = [3, 3]
        Traceback (most recent call last):
            ...
        ValueError: assignment destination is read-only

        >>> trajectory["position"] = None
        >>> trajectory.get_value_slice(slice(0, 2), "position") is None
        True
        """
        value = self.get_value(key, copy=copy)
        if value is None:
            return value
        return value[slice_key]

    @overload
    def __getitem__(self, key: str) -> NDArray[Any] | None:
        # single key
        ...

    @overload
    def __getitem__(
        self, key: list[str] | tuple[str, ...]
    ) -> list[NDArray[Any] | None]:
        # multiple keys
        ...

    @overload
    def __getitem__(self, key: slice) -> Self:
        # trajectory slice
        ...

    @overload
    def __getitem__(self, key: tuple[slice, str]) -> NDArray[Any] | None:
        # single key with slice
        ...

    @overload
    def __getitem__(
        self, key: tuple[slice, list[str] | tuple[str, ...]]
    ) -> dict[str, NDArray[Any] | None]:
        # multiple keys with slice
        ...

    @overload
    def __getitem__(self, key: int | np.integer) -> pyTrajectory.instance.Instance:
        # trajectory index
        ...

    def __getitem__(
        self, key: Any
    ) -> (
        NDArray[Any]
        | None
        | dict[str, NDArray[Any] | None]
        | list[NDArray[Any] | None]
        | Self
    ):
        """Access trajectory data with indexing and slicing.

        Parameters
        ----------
        key : Any
            The key or keys specifying the data to be set or updated. It can be:

            - (str) - A trajectory key.
            - (list[str] | tuple[str, ...]) - Multiple trajectory keys.
            - (slice) - Retrieves a slice of the trajectory.
            - (tuple[slice, str]) - A slice and one trajectory key.
            - (tuple[slice, list[str] | tuple[str, ...]]) - A slice and multiple trajectory keys.
            - (int) - An index to retrieve the corresponding trajectory instance.

        Returns
        -------
        value : Any
            The retrieved trajectory data, value, sliced data or value, or a slice or instance of the trajectory.

            - (NDArray[Any] | None) - The value for one key (read-only).
            - (list[NDArray[Any] | None]) - The values for multiple keys (read-only).
            - (Self) - A trajectory containing the sliced data (copy).
            - (NDArray[Any] | None) - The sliced value for one key (read-only).
            - (dict[str, NDArray[Any] | None]) - The sliced data for multiple keys (read-only).
            - (pyTrajectory.instance.Instance) - The trajectory instance at a given index (copy).

        Raises
        ------
        NotImplementedError
            If the provided key does not match any of the described options.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> trajectory["position"]
        array([[1, 1],
               [2, 2],
               [4, 4]])
        >>> trajectory["position"] += 1
        Traceback (most recent call last):
            ...
        ValueError: output array is read-only

        >>> positions, timestamps = trajectory[("position", "time")]
        >>> positions
        array([[1, 1],
               [2, 2],
               [4, 4]])
        >>> timestamps
        array([1, 2, 4])
        >>> timestamps[0] = 0
        Traceback (most recent call last):
            ...
        ValueError: assignment destination is read-only

        >>> sliced_trajectory = trajectory[:2]
        >>> sliced_trajectory["position"]
        array([[1, 1],
               [2, 2]])
        >>> sliced_trajectory["time"]
        array([1, 2])

        >>> trajectory[0:2, "position"]
        array([[1, 1],
               [2, 2]])
        >>> trajectory[0:2, "time"]
        array([1, 2])
        >>> sliced_positions = trajectory[0:2, "position"]
        >>> sliced_positions[0] = [0, 0]
        Traceback (most recent call last):
            ...
        ValueError: assignment destination is read-only

        >>> sliced_data = trajectory[1:3, ("position", "time")]
        >>> sliced_data["position"]
        array([[2, 2],
               [4, 4]])
        >>> sliced_data["time"]
        array([2, 4])
        >>> sliced_data["time"][-1] += 1
        Traceback (most recent call last):
            ...
        ValueError: assignment destination is read-only

        >>> instance = trajectory[1]
        >>> instance["position"]
        array([2, 2])
        >>> instance["time"]
        2
        >>> instance["time"] += 1
        >>> instance["time"]
        3
        """
        if isinstance(key, str):
            # single key
            return self.get_value(key)
        if isinstance(key, tuple | list) and not isinstance(key[0], slice):
            # multiple keys
            return [self.get_value(key) for key in key]
        if isinstance(key, slice):
            # trajectory slice
            return type(self)(
                data={_key: self.get_value_slice(key, _key) for _key in self.keys()},
                cfg=self.cfg,
                timestep=self._timestep,
            )
        if (
            isinstance(key, tuple)
            and isinstance(key[0], slice)
            and isinstance(key[1], str)
        ):
            # single key with slice
            return self.get_value_slice(*key)
        if (
            isinstance(key, tuple)
            and isinstance(key[0], slice)
            and isinstance(key[1], tuple | list)
        ):
            # multiple keys with slice
            return {_key: self.get_value_slice(key[0], _key) for _key in key[1]}
        if issubclass(type(key), int | np.integer):
            # trajectory index
            instance_data = {
                _key: (deepcopy(value[key]) if value is not None else None)
                for _key, value in self.items(copy=False)
            }
            return pyTrajectory.instance.Instance(cfg=self.cfg, **instance_data)
        raise NotImplementedError

    @overload
    def __setitem__(self, key: str, value: NDArray[Any] | None) -> None:
        # single key, single value
        ...

    @overload
    def __setitem__(
        self, key: list[str] | tuple[str, ...], value: NDArray[Any] | None
    ) -> None:
        # multiple keys, single value
        ...

    @overload
    def __setitem__(
        self,
        key: list[str] | tuple[str, ...],
        value: list[NDArray[Any] | None],
    ) -> None:
        # multiple keys and corresponding values
        ...

    @overload
    def __setitem__(
        self,
        key: slice,
        value: TrajectoryType,
    ) -> None:
        # trajectory slice
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[slice, str],
        value: NDArray[Any] | None,
    ) -> None:
        # single key with slice
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[slice, list[str] | tuple[str, ...]],
        value: NDArray[Any] | None,
    ) -> None:
        # multiple keys with slice, single value
        ...

    @overload
    def __setitem__(
        self,
        key: int,
        value: pyTrajectory.instance.Instance,
    ) -> None:
        # instance at index
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[slice, list[str] | tuple[str, ...]],
        value: list[NDArray[Any] | None],
    ) -> None:
        # multiple keys with slice, corresponding values
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set or update trajectory data using various indexing and slicing methods.

        Parameters
        ----------
        key : Any
            The key or keys specifying the data to be set or updated. It can be:

            - (str) - A trajectory key.
            - (list[str] | tuple[str, ...]) - Multiple trajectory keys.
            - (slice) - A slice to set values on a slice.
            - (tuple[slice, str]) - A slice and a trajectory key to set a sliced value.
            - (tuple[slice, list[str] | tuple[str, ...]]) - A slice and multiple trajectory key to set values on a slice.

        value : Any
            The value or values to assign to the specified keys. It can be:

            - (NDArray[Any] | None) - Set a single value to one or multiple keys, or to a slice of one or multiple keys.
            - (list[NDArray[Any] | None]) - Set multiple values to multiple keys, or to a slice of multiple keys.
            - TrajectoryType - Set a slice of the trajectory.

        Returns
        -------
        None
            This method modifies the trajectory data in place.

        Raises
        ------
        NotImplementedError
            If the provided key and value combinations do not match any of the described options.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        Setting a single value:

        >>> trajectory["time"] = np.array([0, 2, 4])
        >>> trajectory["time"]
        array([0, 2, 4])

        Setting multiple keys to a single value. If all keys are provided, the trajectory can
        be completely overwritten (e.g. have a different length afterwards):

        >>> trajectory[("time", "position")] = np.array([0, 1, 2, 3, 4])
        >>> trajectory["time"]
        array([0, 1, 2, 3, 4])
        >>> trajectory["position"]  # Not that this also changed the dimensionality
        array([0, 1, 2, 3, 4])

        Setting multiple keys to corresponding values.

        >>> trajectory[("time", "position")] = [
        ...     np.array([0, 1, 2, 3]),
        ...     np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
        ... ]
        >>> trajectory["time"]
        array([0, 1, 2, 3])
        >>> trajectory["position"]
        array([[0, 0],
               [1, 1],
               [2, 2],
               [3, 3]])

        Direct broadcasting of single numeric values without using slicing is also allowed. Broadcasting arrays without
        slicing does not work, because it is ambiguous (could be invalid data).

        >>> trajectory[("time", "position")] = [np.array([0, 1, 4, 5]), 2]
        >>> np.all(trajectory["position"][:, :] == 2)
        True

        >>> trajectory[("time", "position")] = [
        ...     np.array([0, 1, 4, 5]),
        ...     np.array([0, 1]),
        ... ]
        Traceback (most recent call last):
            ...
        ValueError: non-None values mave mismatched lengths

        >>> # Use of slice enforces correct broadcasting.
        >>> trajectory[:, ("time", "position")] = [
        ...     np.array([0, 1, 4, 5]),
        ...     np.array([0, 1]),
        ... ]
        >>> trajectory["position"]
        array([[0, 1],
               [0, 1],
               [0, 1],
               [0, 1]])

        This is equivalent to:

        >>> trajectory["time"] = np.array([0, 1, 4, 5])
        >>> trajectory[:, "position"] = np.array([0, 1])
        >>> trajectory["position"]
        array([[0, 1],
               [0, 1],
               [0, 1],
               [0, 1]])

        Setting a slice of the trajectory.

        >>> short_trajectory = Trajectory(
        ...     data={
        ...         "time": np.array([10, 11]),
        ...         "position": np.array([[10, 10], [11, 11]]),
        ...     }
        ... )
        >>> trajectory[-2:] = short_trajectory
        >>> trajectory["time"]
        array([ 0,  1, 10, 11])
        >>> trajectory["position"]
        array([[ 0,  1],
               [ 0,  1],
               [10, 10],
               [11, 11]])

        Keep in mind that setting the data must always result in valid trajectory data.

        >>> trajectory[1:3] = short_trajectory
        Traceback (most recent call last):
            ...
        ValueError: data contains duplicated timestamps

        Setting a slice for a single key. Note that if broadcasting should be used, assign either with
        a numpy array or single numeric value (float or int).

        >>> # although this works, use np.array([1, 3]) for consistency (see below examples)
        >>> trajectory[:, "position"] = [1, 3]
        >>> trajectory[:, "position"] = np.array([1, 3])
        >>> trajectory["position"]
        array([[1, 3],
               [1, 3],
               [1, 3],
               [1, 3]])

        >>> trajectory[2:, "position"] = 2
        >>> trajectory["position"]
        array([[1, 3],
               [1, 3],
               [2, 2],
               [2, 2]])

        Setting a slice for multiple keys with the same value.

        >>> cfg = cfg.copy()
        >>> cfg.trajectory_keys = ("time", "position", "posture")
        >>> data = trajectory.data
        >>> data["posture"] = np.array(
        ...     [[[1, 1], [2, 2]], [[2, 2], [3, 3]], [[3, 3], [4, 4]], [[4, 4], [5, 5]]]
        ... )
        >>> trajectory = Trajectory(data=data, cfg=cfg)

        >>> trajectory[:, ("position", "posture")] = 1
        >>> np.all(trajectory["position"][:, :] == 1)
        True
        >>> np.all(trajectory["posture"][:, :, :] == 1)
        True

        The use of a numpy array is required when multiple keys should be set by broadcasting the same value
        when it is not a single, numeric value (float or int).

        >>> trajectory[:, ("position", "posture")] = np.array([1, 2])
        >>> np.all(trajectory["position"][:, 0] == 1)
        True
        >>> np.all(trajectory["position"][:, 1] == 2)
        True
        >>> np.all(trajectory["posture"][:, :, 0] == 1)
        True
        >>> np.all(trajectory["posture"][:, :, 1] == 2)
        True

        In contrast, the following example broadcasts single numeric values to both keys, by setting a
        slice for multiple keys with corresponding values.

        >>> trajectory[:, ("position", "posture")] = [1, 2]
        >>> np.all(trajectory["position"][:, :] == 1)
        True
        >>> np.all(trajectory["posture"][:, :, :] == 2)
        True

        >>> trajectory[:, ("position", "posture")] = [1, 2]
        >>> np.all(trajectory["position"][:, :] == 1)
        True
        >>> np.all(trajectory["posture"][:, :, :] == 2)
        True

        Here, appropriate broadcasting is also possible:

        >>> trajectory[:2, ("position", "posture")] = [
        ...     np.array([0, 1]),
        ...     np.array([1, 2]),
        ... ]
        >>> trajectory["position"]
        array([[0, 1],
               [0, 1],
               [1, 1],
               [1, 1]])
        >>> np.all(trajectory["posture"][:2, :, 0] == 1)
        True
        >>> np.all(trajectory["posture"][:2, :, 1] == 2)
        True
        >>> np.all(trajectory["posture"][2:] == 2)  # old values, set above
        True

        >>> from pyTrajectory.instance import Instance
        >>> instance = Instance(
        ...     cfg=trajectory.cfg,
        ...     time=15,
        ...     position=[0, 0],
        ...     posture=[[-1, -1], [-2, -2]],
        ... )
        >>> trajectory[-1] = instance
        """

        def is_value(value):
            if value is None or isinstance(
                value, (np.ndarray, int, float, np.integer, np.floating)
            ):
                return True
            return False

        if isinstance(key, str) and is_value(value):
            # single key, single value
            self.set_value(key, value, validate=True)
            return
        if (
            isinstance(key, tuple | list)
            and isinstance(key[0], str)
            and is_value(value)
        ):
            # multiple keys, single value
            self.validate_data({_key: value for _key in key})
            for _key in key:
                self.set_value(
                    _key,
                    value,
                    validate=False,
                )  # no need to validate again
            return
        if isinstance(key, slice) and issubclass(type(value), type(self)):
            # trajectory slice
            validate_keys(value.data, self)
            data = self.data
            for _key in value.keys():
                data[_key][key] = value[_key]
            self.validate_data(data)
            for _key in value.keys():
                self.set_value_slice(
                    key,
                    _key,
                    value[_key],
                    validate=False,
                )  # no need to validate again
            return
        if (
            isinstance(key, tuple)
            and isinstance(key[0], slice)
            and isinstance(key[1], str)
        ):
            # single key with slice
            data = {key[1]: self.get_value(key[1], copy=True)}
            data[key[1]][key[0]] = value
            self.validate_data(data)
            self.set_value_slice(
                *key, value, validate=False
            )  # no need to validate again
            return
        if (
            isinstance(key, tuple)
            and isinstance(key[0], slice)
            and isinstance(key[1], tuple | list)
            and is_value(value)
        ):
            # multiple keys with slice, single value
            data = {_key: self.get_value(_key, copy=True) for _key in key[1]}
            for _key in key[1]:
                data[_key][key[0]] = value
            self.validate_data(data)
            for _key in key[1]:
                self.set_value_slice(
                    key[0],
                    _key,
                    value,
                    validate=False,
                )  # no need to validate again
            return
        if (
            isinstance(key, tuple)
            and isinstance(key[0], slice)
            and isinstance(key[1], tuple | list)
            and isinstance(value, list)
        ):
            # multiple keys with slice, corresponding values
            data = {_key: self.get_value(_key, copy=True) for _key in key[1]}
            for _key, _value in zip(key[1], value):
                data[_key][key[0]] = _value
            self.validate_data(data)
            for _key, _value in zip(key[1], value):
                self.set_value_slice(
                    key[0],
                    _key,
                    _value,
                    validate=False,
                )  # no need to validate again
            return
        if isinstance(key, tuple | list) and isinstance(value, list):
            # multiple keys and corresponding values
            self.validate_data({_key: _value for _key, _value in zip(key, value)})
            for _key, _value in zip(key, value):
                self.set_value(
                    _key,
                    _value,
                    validate=False,
                )  # no need to validate again
            return
        if isinstance(key, int | np.integer) and isinstance(
            value, pyTrajectory.instance.Instance
        ):
            self.set_instance(key, value)  # always validates
            return
        raise NotImplementedError

    @property
    def timestep(self) -> int | float:
        """The timestep of the trajectory timestamps.

        If the timestep property is not explicitly set directly or via the config (both None), this returns
        the inverse of the greatest common denominator of unique timesteps (delta timestamps).
        Otherwise, the timestep of the trajectory is prioritized over the config.
        Note, this is not necessarily the smallest timestep (e.g. when the smallest timestep is 2),
        but usually 1 (for all timesteps that are dividable by 1).

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> trajectory.timestep
        1

        >>> trajectory["time"] = np.array([1.0, 2.0, 4.0])
        >>> trajectory.timestep
        1.0

        # this is 1 and not 2, as for integer timesteps, the denominators is always 1.
        >>> trajectory["time"] = np.array([1.0, 3.0, 11.0])
        >>> trajectory.timestep
        1.0

        >>> data = {
        ...     "position": np.repeat([1, 2], 100).reshape(-1, 2),
        ...     "time": np.linspace(0, 1, 100),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> trajectory.timestep
        0.010101010101010102
        >>> trajectory.timestep == 1 / 99
        True

        >>> timesteps = np.diff(data["time"])
        >>> indices = np.random.choice(np.arange(len(timesteps)), 30, replace=False)
        >>> min_random_timestep = 1
        >>> max_random_timestep = 20
        >>> random_timesteps = np.random.randint(
        ...     min_random_timestep, max_random_timestep + 1, len(indices)
        ... )
        >>> timesteps[indices] *= random_timesteps
        >>> data["time"] = np.concatenate([[0], np.cumsum(timesteps)])
        >>> trajectory = Trajectory(data=data)
        >>> trajectory.timestep
        0.010101010101010102
        >>> trajectory.timestep == 1 / 99
        True

        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data, timestep=0.1)
        >>> trajectory.timestep
        0.1

        >>> trajectory.cfg.timestep = 0.01
        >>> trajectory.timestep  # timestep is prioritized over cfg.timestep
        0.1

        >>> trajectory.timestep = None  # fall back to cfg.timestep
        >>> trajectory.timestep
        0.01

        >>> trajectory.cfg.timestep = None  # fall back to infering from timestamps
        >>> trajectory.timestep
        1

        Setting an incorrect timestep is possible

        >>> trajectory.timestep = 2
        >>> trajectory.timestep
        2
        >>> duration = trajectory[-1]["time"] - trajectory[0]["time"]
        >>> # duration should be dividable by timestep
        >>> duration % trajectory.timestep == 0
        False

        Setting None should be preferred in when in doubt

        >>> trajectory.timestep = None
        >>> trajectory.timestep
        1
        >>> duration = trajectory[-1]["time"] - trajectory[0]["time"]
        >>> # duration should be dividable by timestep
        >>> duration % trajectory.timestep == 0
        True
        """
        if self._timestep is not None:
            return self._timestep
        if self.cfg.timestep is not None:
            return self.cfg.timestep
        unique_timesteps = np.unique(np.diff(self[self.cfg.key_timestamp]))
        timestep = greatest_common_denominator(unique_timesteps)
        is_int = issubclass(self[self.cfg.key_timestamp].dtype.type, np.integer)
        if is_int:
            timestep = int(timestep)
        return timestep

    @timestep.setter
    def timestep(self, timestep: int | float | None) -> None:
        """Set the timestep of the trajectory. Note that this is not validated."""
        self._timestep = timestep

    @property
    def is_sorted(self) -> bool:
        """Whether the timestamps increase monotonically.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> trajectory.is_sorted
        True

        >>> instance = trajectory[-1]
        >>> instance["time"] = 0
        >>> trajectory[-1] = instance
        >>> trajectory["time"]
        array([1, 2, 0])
        >>> trajectory.is_sorted
        False
        """
        return (np.diff(self[self.cfg.key_timestamp]) > 0).all()

    @property
    def is_complete(self) -> bool:
        """Whether the trajectory is complete.

        A trajectory is complete, when the duration (max(timestamps) - min(timestamps))
        equals (length - 1) * timestep. Nothing to interpolate.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> trajectory.is_complete
        False

        >>> instance = trajectory[-1]
        >>> instance["time"] = 3
        >>> trajectory[-1] = instance
        >>> trajectory.is_complete
        True
        """
        timestamps = self[self.cfg.key_timestamp]
        duration = timestamps.max() - timestamps.min()
        return isclose(duration, (self.length - 1) * self.timestep)

    def copy(self) -> TrajectoryType:
        """Return a copy of the trajectory.

        Returns
        -------
        trajectory : TrajectoryType
            A copy of the trajectory

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [2, 2], [4, 4]]),
        ...     "time": np.array([1, 2, 4]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> trajectory_copy = trajectory.copy()
        >>> trajectory is trajectory_copy
        False

        >>> trajectory[:, "position"] = np.array([1, 2])
        >>> np.array_equal(trajectory["position"], trajectory_copy["position"])
        False
        """
        return type(self)(
            data=self.data,
            cfg=self.cfg,
            timestep=self.timestep,
        )

    def sort(self, copy: bool = True) -> TrajectoryType:
        """Sort the trajectory, so that timestamps are monotonically increasing.

        Parameters
        ----------
        copy :  bool, optional
            Whether to sort the trajectory in place, or to return a sorted copy (default).

        Returns
        -------
        TrajectoryType
            The sorted trajectory (either a copy, or the trajectory).

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [3, 3], [2, 2]]),
        ...     "time": np.array([1, 3, 2]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> trajectory.is_sorted
        False

        >>> sorted_trajectory = trajectory.sort()
        >>> sorted_trajectory is trajectory  # by default, copy
        False
        >>> trajectory.is_sorted
        False
        >>> sorted_trajectory.is_sorted
        True
        >>> sorted_trajectory["time"]
        array([1, 2, 3])
        >>> sorted_trajectory["position"]
        array([[1, 1],
               [2, 2],
               [3, 3]])

        >>> _ = trajectory.sort(copy=False)  # sort in place
        >>> trajectory.is_sorted
        True
        >>> trajectory["time"]
        array([1, 2, 3])
        >>> trajectory["position"]
        array([[1, 1],
               [2, 2],
               [3, 3]])
        """
        if self.is_sorted:
            if not copy:
                return self
            return type(self)(data=self.data)
        sort_idx = np.argsort(self[self.cfg.key_timestamp])
        data = {key: value[sort_idx] for key, value in self.items()}
        if not copy:
            self.data = data
            return self
        return type(self)(data=data)

    def sample(
        self,
        timestamps: NDArray[int | float],
        copy: bool = True,
    ) -> TrajectoryType:
        """Sample the trajectory at given timestamps.

        Parameters
        ----------
        timestamps : NDArray[int | float]
            At which timestamps to sample
        copy : bool, optional
            Whether to sample the trajectory in place, or to return a sampled copy (default).

        Returns
        -------
        TrajectoryType
                The sampled trajectory (either a copy, or the trajectory).

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [3, 3], [2, 2]]),
        ...     "time": np.array([1, 3, 2]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> trajectory.sample(np.array([0.5, 1.5]))
        Traceback (most recent call last):
            ...
        AssertionError: can only sample sorted trajectory

        >>> trajectory = trajectory.sort()
        >>> sampled_trajectory = trajectory.sample(np.array([0.5, 1.5]))
        >>> sampled_trajectory["time"]
        array([0.5, 1.5])
        >>> # note that the array dtype is enforced
        >>> sampled_trajectory["position"]
        array([[1, 1],
               [1, 1]])

        If timestamps outside the trajectory interval (between first and last timestamp)
        are sampled, data is not extrapolated but first or last trajectory instance, respectively.
        Otherwise, data is linearly interpolated.

        >>> data = {
        ...     "position": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        ...     "time": np.array([1, 2, 3]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> sampled_trajectory = trajectory.sample(np.array([0.5, 1.5]))
        >>> sampled_trajectory["time"]
        array([0.5, 1.5])
        >>> sampled_trajectory["position"]
        array([[1. , 1. ],
               [1.5, 1.5]])

        Trajectory sampling should result in valid trajectories.

        >>> sampled_trajectory = trajectory.sample(np.array([1.5, 1.5]))
        Traceback (most recent call last):
            ...
        ValueError: data contains duplicated timestamps

        >>> sampled_trajectory = trajectory.sample(np.array([0.5, 1.5]))
        >>> sampled_trajectory is trajectory
        False
        >>> sampled_trajectory = trajectory.sample(np.array([0.5, 1.5]), copy=False)
        >>> sampled_trajectory is trajectory
        True
        """
        if not self.is_sorted:
            raise AssertionError("can only sample sorted trajectory")
        data = {
            key: pyTrajectory.series_operations.sample_series(
                self[key],
                self[self.cfg.key_timestamp],
                timestamps,
            )
            for key in self.keys(exclude=[self.cfg.key_timestamp])
        }
        data[self.cfg.key_timestamp] = timestamps
        if not copy:
            self.data = data
            return self
        return type(self)(data=data)

    def interpolate(
        self,
        timestep: int | float | None = None,
        copy: bool = True,
    ) -> TrajectoryType:
        """Interpolate the trajectory.

        Parameters
        ----------
        timestep : int | float | None, optional
            The timestep used for interpolation. If None (default), the trajectory timestep will be used.
        copy : bool, optional
            Whether to interpolate the trajectory in place, or to return a sampled copy (default).

        Returns
        -------
        TrajectoryType
            The interpolated trajectory (either a copy, or the trajectory).


        See Also
        --------
        Trajectory.timestep
            Set, get or infer the trajectory timestep.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [3, 3], [2, 2]]),
        ...     "time": np.array([1, 3, 2]),
        ... }
        >>> trajectory = Trajectory(data=data)

        >>> trajectory.interpolate()
        Traceback (most recent call last):
            ...
        AssertionError: can only sample sorted trajectory

        >>> data = {
        ...     "position": np.array([[1, 1], [3, 3]]),
        ...     "time": np.array([1, 3]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> trajectory.timestep
        1

        >>> trajectory.interpolate()["time"]
        array([1, 2, 3])
        >>> trajectory.interpolate(timestep=0.5)["time"]
        array([1. , 1.5, 2. , 2.5, 3. ])
        >>> trajectory.interpolate(timestep=2 / 3)["time"]
        array([1.        , 1.66666667, 2.33333333, 3.        ])

        The timestep should result in a integer trajectory length.

        >>> trajectory.interpolate(timestep=0.75)["time"]
        Traceback (most recent call last):
            ...
        ValueError: timestep should result in an integer trajectory length and not: 3.6666666666666665

        >>> interpolated_trajectory = trajectory.interpolate()
        >>> interpolated_trajectory is trajectory
        False
        >>> interpolated_trajectory = trajectory.interpolate(copy=False)
        >>> interpolated_trajectory is trajectory
        True
        """
        timestamps = self[self.cfg.key_timestamp]
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
        if isclose(timestep, 1) and isclose(first, np.round(first)):
            timestamps = np.round(timestamps).astype(int)
        trajectory = self.sample(timestamps, copy=copy)
        trajectory.timestep = timestep
        return trajectory

    def slice_window(
        self,
        start: int | float,
        stop: int | float,
        interpolate: bool = True,
        interpolation_timestep: int | float | None = None,
    ) -> TrajectoryType:
        """Get a window of the trajectory between a start and a stop timestamp (both inclusive).

        Parameters
        ----------
        start : int | float
            The start timestamp of the trajectory window
        stop : int | float
            The stop timestamp of the trajectory window
        interpolate : bool, optional
            Whether to interpolate the trajectory window between start and stop. This ensures both
            start and stop are included in the window. Windows are interpolated by default.
        interpolation_timestep : int | float | None, optional
            With which timestep the window should be interpolated. If None (default), the trajectory timestep will be used.

        Returns
        -------
        TrajectoryType
            The trajectory window.

        Raises
        ------
        OutOfInterval
            When the start or stop timestamp are outside of the trajectory interval.

        Examples
        --------
        >>> data = {
        ...     "position": np.array([[1, 1], [10, 10]]),
        ...     "time": np.array([1, 10]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> trajectory_window = trajectory.slice_window(2, 5)
        >>> len(trajectory_window) == 1 + (5 - 2) / trajectory.timestep
        True
        >>> trajectory_window["time"]
        array([2, 3, 4, 5])

        >>> trajectory_window = trajectory.slice_window(2, 5, interpolate=False)
        >>> trajectory_window["time"]
        array([], dtype=int64)

        >>> data = {
        ...     "position": np.array([[1, 1], [3, 3], [10, 10]]),
        ...     "time": np.array([1, 3, 10]),
        ... }
        >>> trajectory = Trajectory(data=data)
        >>> trajectory_window = trajectory.slice_window(2, 5, interpolate=False)
        >>> trajectory_window["time"]
        array([3])

        >>> trajectory_window = trajectory.slice_window(
        ...     2, 5, interpolation_timestep=0.5
        ... )
        >>> len(trajectory_window) == 1 + (5 - 2) / 0.5
        True
        >>> trajectory_window["time"]
        array([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])

        >>> trajectory.timestep = None
        >>> trajectory.cfg = trajectory.cfg.copy()
        >>> trajectory.cfg.timestep = 0.5
        >>> trajectory_window = trajectory.slice_window(
        ...     2, 5, interpolation_timestep=None
        ... )
        >>> len(trajectory_window) == 1 + (5 - 2) / 0.5
        True
        """
        key_timestamp = self.cfg.key_timestamp
        timestamps = self[self.cfg.key_timestamp]
        first = timestamps.min()
        last = timestamps.max()
        if start < first:
            raise OutOfInterval(
                f"start: {start} not in trajectory range: [{first} {last}]"
            )
        if stop > last:
            raise OutOfInterval(
                f"stop: {stop} not in trajectory range: [{first} {last}]"
            )
        slice_key = slice(
            np.argwhere(timestamps >= start).ravel()[0],
            np.argwhere(timestamps <= stop).ravel()[-1] + 1,
        )
        if not interpolate:
            return self[slice_key]
        if self[slice_key.start][key_timestamp] > start:
            slice_key = slice(
                max(
                    0,
                    slice_key.start - 1,
                ),
                slice_key.stop,
            )
        if self[slice_key.stop - 1][key_timestamp] < stop:
            slice_key = slice(
                slice_key.start,
                min(
                    self.length,
                    slice_key.stop + 1,
                ),
            )
        trajectory_window = self[slice_key]
        trajectory_window.timestep = interpolation_timestep
        if not trajectory_window.is_complete:
            trajectory_window = trajectory_window.interpolate(
                timestep=interpolation_timestep
            )
        if isclose(
            trajectory_window[0][key_timestamp],
            start,
        ) and isclose(
            trajectory_window[-1][key_timestamp],
            stop,
        ):
            return trajectory_window
        return trajectory_window.slice_window(
            start,
            stop,
            interpolate=False,
            interpolation_timestep=interpolation_timestep,
        )


if __name__ == "__main__":
    import doctest

    import numpy as np
    import pyTrajectory.config

    cfg = pyTrajectory.config.cfg
    cfg.trajectory_keys = (
        "time",
        "position",
    )
    cfg.key_timestamp = "time"

    doctest.testmod(extraglobs={"cfg": cfg})
