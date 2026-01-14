from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Self, overload, override

import numpy as np
from numpy.dtypes import StringDType

from ..config import Config
from ..config import cfg as CFG
from ._type_checking import (
    confirm_int_str,
    confirm_int_str_iterable,
    confirm_slice_str,
    confirm_slice_str_iterable,
    confirm_str_iterable,
    confirm_value,
    confirm_value_iterable,
)
from .base import ConfiguredData
from .instance import Instance
from .utils import (
    MultipleValues,
    Value,
    validate_keys,
    validate_timestamps,
    validated_length,
    writeable,
)


class InstanceCollection(ConfiguredData):
    """
    The base class to represent an unordered collection of instances, possibly from more than one animal.

    Implements the :code:`__getitem__` and :code:`__setitem__` methods to provide indexing, slicing, and dictionary-like access to the data.

    Parameters:
        data: A dictionary containing the data for the collection.
        cfg: The configuration object.
        validate_on_init: Whether to validate the data during initialization.
    """

    _view_of: list[Self]
    _views: list[Self]
    _validate: bool
    _length: int

    def __init__(
        self,
        *,
        data: Mapping[str, np.ndarray] | None = None,
        cfg: Config | None = None,
        validate_on_init: bool = True,
    ) -> None:
        self._validate = True
        self._view_of = []
        self._views = []
        if cfg is None:
            cfg = CFG.copy()
        self._cfg: Config | None = cfg
        self._length = 0
        if data is not None:
            with self.validate(validate_on_init):
                self.data = dict(data)

    @contextmanager
    def validate(self, validate: bool) -> Iterator[None]:
        """
        Yields a context where data validation is enabled or disabled.

        Parameters:
            validate: Whether to enable validation.
        """
        _validate = self._validate
        self._validate = validate
        try:
            yield
        except Exception as e:
            raise e
        finally:
            self._validate = _validate

    @property
    def length(self) -> int:
        """Returns the number of instances in the collection."""
        return self._length

    def __len__(self) -> int:
        return self.length

    def validate_data(
        self,
        data: Mapping[str, Value],
        *,
        allow_duplicated_timestamps: bool = True,
        allow_missing_keys: bool = False,
        try_broadcasting: bool = True,
        require_array_like: bool = False,
    ) -> bool:
        """
        Validates the input data against the specified requirements.

        Parameters:
            data: The data to validate.
            allow_duplicated_timestamps: Whether to allow duplicated timestamps.
            allow_missing_keys: Whether to allow missing keys.
            try_broadcasting: Whether to try broadcasting.
            require_array_like: Whether to require array-like values.

        Returns:
            bool: If the data passed validation.

        Raises:
            ValueError: If the data fails any of the validation checks, such as key mismatches, length mismatches, or invalid timestamp or identity data types.
        """
        complete_keys = validate_keys(
            data.keys(), self.keys(), allow_missing=allow_missing_keys
        )
        if require_array_like and any(
            [
                (not isinstance(value, Iterable) or isinstance(value, str))
                for value in data.values()
            ]
        ):
            raise ValueError("all values are required to be array-like.")
        length = validated_length(*data.values())
        if (
            not allow_duplicated_timestamps
            and (key_timestamp := self.cfg.key_timestamp) is not None
            and key_timestamp in data
        ):
            _ = validate_timestamps(data[key_timestamp])
        if (
            require_array_like
            and (key_identity := self.cfg.key_identity) is not None
            and key_identity in data
            and (
                not isinstance((identities := data[key_identity]), np.ndarray)
                or not (
                    isinstance(identities.dtype, StringDType)
                    or np.issubdtype(identities.dtype, int)
                )
            )
        ):
            raise ValueError(
                f"value for {key_identity} should be a numpy array of type int or numpy.dtypes.StringDType"
            )
        if complete_keys:
            # complete override
            return True
        if length is None or length == self.length:
            # partial override
            return True
        if not try_broadcasting:
            raise ValueError("length mismatch")
        return True

    @property
    @override
    def data(self) -> dict[str, np.ndarray]:
        """
        Property that returns the underlying data dictionary.
        The property can be set to update the data if the data passes validation.
        """
        return super().data

    @data.setter
    def data(self, data: dict[str, np.ndarray]) -> None:
        if self._validate:
            _ = self.validate_data(data, require_array_like=True)
        self._data: dict[str, np.ndarray] | None = {
            key: value for key, value in data.items()
        }
        self._length = len(next(iter(data.values())))

    def init_other(
        self,
        *,
        data: dict[str, np.ndarray],
        copy_config: bool = False,
        validate_on_init: bool = False,
    ) -> Self:
        """
        Initializes a new collection from provided data with the same configuration.

        Parameters:
            data: A dictionary containing the data for the new InstanceCollection.
            copy_config: Whether to copy or use the configuration from the current instance.
            validate_on_init: Whether to validate the data during initialization.

        Returns:
            The new collection.
        """
        cfg = self.cfg
        if copy_config:
            cfg = cfg.copy()
        return type(self)(
            data=data,
            cfg=cfg,
            validate_on_init=validate_on_init,
        )

    def copy(self, *, copy_config: bool = False) -> Self:
        """
        Copies the collection.

        Parameters:
            copy_config: Whether to copy the configuration.

        Returns:
            the copied collection.
        """
        return self.init_other(
            data=self.data, copy_config=copy_config, validate_on_init=False
        )

    def _set_value(
        self,
        key: str,
        value: Value,
        at: slice | int | np.integer,
    ) -> None:
        """
        Sets a value in the underlying data structure at a specified location (index or slice).

        This method allows modification of the stored data. If the data is a view, the base data is also modified. No data validation is performed.

        Parameters:
            key: The key associated with the value to be set.
            value: The value to set.
            at: The index or slice where the value should be set.

        Raises:
            ValueError: If the underlying data structure is not initialized.
        """
        if self._data is None:
            raise ValueError("not initialized")
        _value = self._get_value(key, copy=False)
        with writeable(
            *[base._get_value(key, copy=False) for base in self._view_of],
            _value,
        ):
            _value[at] = value

    @overload
    def __getitem__(self, key: str) -> np.ndarray:
        # single key
        ...

    @overload
    def __getitem__(self, key: tuple[str, ...] | list[str]) -> tuple[np.ndarray, ...]:
        # multiple keys
        ...

    @overload
    def __getitem__(self, key: slice) -> Self:
        # slice
        ...

    @overload
    def __getitem__(self, key: tuple[slice, str]) -> np.ndarray:
        # slice single key
        ...

    @overload
    def __getitem__(
        self, key: tuple[slice, tuple[str, ...] | list[str]]
    ) -> dict[str, np.ndarray]:
        # slice multiple keys
        ...

    @overload
    def __getitem__(self, key: int | np.integer) -> Instance:
        # trajectory index
        ...

    @overload
    def __getitem__(self, key: tuple[int | np.integer, str]) -> Value:
        # trajectory index with key
        ...

    @overload
    def __getitem__(
        self, key: tuple[int | np.integer, tuple[str, ...] | list[str]]
    ) -> dict[str, Value]:
        # trajectory index with multiple keys
        ...

    def __getitem__(
        self,
        key: (
            str
            | tuple[str, ...]
            | list[str]
            | slice
            | tuple[slice, str]
            | tuple[slice, tuple[str, ...] | list[str]]
            | int
            | np.integer
            | tuple[int | np.integer, str]
            | tuple[int | np.integer, tuple[str, ...] | list[str]]
        ),
    ) -> (
        np.ndarray
        | tuple[np.ndarray, ...]
        | Self
        | dict[str, np.ndarray]
        | Instance
        | Value
        | dict[str, Value]
    ):
        if isinstance(key, str):
            # single key
            return self._get_value(key)
        if confirm_str_iterable(key):
            # multiple keys
            return tuple(self._get_value(_key) for _key in key)
        if isinstance(key, slice):
            # slice
            view = self.init_other(
                data={_key: self._get_value(_key)[key] for _key in self.keys()}
            )
            self._views.append(view)
            view._view_of.append(self)
            return view
        if confirm_slice_str(key):
            # slice single key
            return self._get_value(key[1])[key[0]]
        if confirm_slice_str_iterable(key):
            # slice multiple keys
            return {_key: self._get_value(_key)[key[0]] for _key in key[1]}
        if isinstance(key, int | np.integer):
            # instance at index
            instance_data = {_key: value[key] for _key, value in self.items(copy=False)}
            return Instance(cfg=self.cfg, **instance_data, from_scalars=True)
        if confirm_int_str(key):
            # instance value at index
            return self._get_value(key[1], copy=True)[key[0]]
        if confirm_int_str_iterable(key):
            # instance values at index
            return {_key: self._get_value(_key, copy=True)[key[0]] for _key in key[1]}
        raise KeyError(f"unsupported key of type ({type(key)})")

    @overload
    def __setitem__(self, key: str, value: Value) -> None:
        # single key, single value
        ...

    @overload
    def __setitem__(self, key: tuple[str, ...] | list[str], value: Value) -> None:
        # multiple keys, single value
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[str, ...] | list[str],
        value: MultipleValues,
    ) -> None:
        # multiple keys and corresponding values
        ...

    @overload
    def __setitem__(
        self,
        key: slice,
        value: Self,
    ) -> None:
        # slice
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[slice, str],
        value: Value,
    ) -> None:
        # single key with slice
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[slice, tuple[str, ...] | list[str]],
        value: Value,
    ) -> None:
        # multiple keys with slice, single value
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[slice, tuple[str, ...] | list[str]],
        value: MultipleValues,
    ) -> None:
        # multiple keys with slice, corresponding values
        ...

    @overload
    def __setitem__(
        self,
        key: int,
        value: Instance,
    ) -> None:
        # instance at index
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[int, str],
        value: Value,
    ) -> None:
        # instance value at index
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[int, tuple[str, ...] | list[str]],
        value: Value,
    ) -> None:
        # instance values at index
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[int, tuple[str, ...] | list[str]],
        value: MultipleValues,
    ) -> None:
        # instance values at index, corresponding values
        ...

    def __setitem__(
        self,
        # single key, single value
        # multiple keys, single value
        # multiple keys and corresponding values
        # slice
        # single key with slice
        # multiple keys with slice, single value
        # multiple keys with slice, corresponding values
        # instance at index
        # instance value at index
        # instance values at index
        # instance values at index, corresponding values
        key: (
            str
            | tuple[str, ...]
            | list[str]
            | slice
            | tuple[slice, str]
            | tuple[slice, tuple[str, ...] | list[str]]
            | int
            | tuple[int, str]
            | tuple[int, tuple[str, ...] | list[str]]
        ),
        value: (Value | MultipleValues | Self | Instance),
    ) -> None:
        # single value
        if confirm_value(value):
            if isinstance(key, str):
                # single key, single value
                if self._validate:
                    data = self.data
                    data[key][:] = value
                    _ = self.validate_data(data)
                self._set_value(key, value, slice(None))
                return
            if confirm_str_iterable(key):
                # multiple keys, single value
                if self._validate:
                    data = self.data
                    for _key in key:
                        data[_key][:] = value
                    _ = self.validate_data(data)
                for _key in key:
                    self._set_value(_key, value, slice(None))
                return
            if confirm_slice_str(key):
                # single key with slice
                if self._validate:
                    data = self.data
                    data[key[1]][key[0]] = value
                    _ = self.validate_data(data)
                self._set_value(key[1], value, key[0])
                return
            if confirm_slice_str_iterable(key):
                # multiple keys with slice, single value
                if self._validate:
                    data = self.data
                    for _key in key[1]:
                        data[_key][key[0]] = value
                    _ = self.validate_data(data)
                for _key in key[1]:
                    self._set_value(_key, value, key[0])
                return
            if confirm_int_str(key):
                # instance value at index
                if self._validate:
                    data = self.data
                    data[key[1]][key[0]] = value
                    _ = self.validate_data(data)
                self._set_value(key[1], value, key[0])
                return
            if confirm_int_str_iterable(key):
                # instance values at index
                if self._validate:
                    data = self.data
                    for _key in key[1]:
                        data[_key][key[0]] = value
                    _ = self.validate_data(data)
                for _key in key[1]:
                    self._set_value(_key, value, key[0])
                return
            raise ValueError(f"Invalid key type for single value: {type(key)}")
        # corresponding values
        if confirm_value_iterable(value):
            if confirm_str_iterable(key):
                # multiple keys and corresponding values
                if self._validate:
                    data = self.data
                    for _key, _value in zip(key, value):
                        data[_key][:] = _value
                    _ = self.validate_data(data)
                for _key, _value in zip(key, value):
                    self._set_value(_key, _value, slice(None))
                return
            if confirm_slice_str_iterable(key) or confirm_int_str_iterable(key):
                # multiple keys with slice, corresponding values
                # instance values at index, corresponding values
                if self._validate:
                    data = self.data
                    for _key, _value in zip(key[1], value):
                        data[_key][key[0]] = _value
                    _ = self.validate_data(data)
                for _key, _value in zip(key[1], value):
                    self._set_value(_key, _value, key[0])
                return
            if isinstance(key, slice) and isinstance(value, InstanceCollection):
                # slice
                _data = value.data
                if self._validate:
                    _ = self.validate_data(_data)
                    data = self.data
                    for _key, _value in _data.items():
                        data[_key][key] = _value
                    _ = self.validate_data(data)
                for _key, _value in _data.items():
                    self._set_value(_key, _value, key)
                return
            if isinstance(key, int | np.integer) and isinstance(value, Instance):
                # instance at index
                if self._validate:
                    data = self.data
                    for _key in value.keys():
                        data[_key][key] = value[_key]
                    _ = self.validate_data(data)
                for _key in value.keys():
                    self._set_value(_key, value[_key], key)
                return
            raise ValueError(f"Invalid key type for corresponding values: {type(key)}")
        raise ValueError(
            f"unsupported key value pair of types: {type(key)}, {type(value)}"
        )

    def select_index(self, index: np.ndarray) -> Self:
        """
        Selects a subset of the data based on the provided index.
        Allows indexing as in :code:`numpy`, and will return a view if possible.

        Parameters:
            index: The indices to select.

        Returns:
            The selected collection.
        """
        # advanced indexing with boolean array always triggers copy, not view
        # # TODO check index (at least 1d numpy array if int, else boolean with length of collection)
        # # otherwise coerce to array with appropriate type or raise
        return self.init_other(
            data={key: value[index] for key, value in self.items(copy=False)}
        )

    def select(
        self,
        *,
        timestamp: Value | None = None,
        identity: Value | None = None,
    ) -> Self:
        """
        Selects a subset of instances based on timestamp and/or identity.

        Parameters:
            timestamp: The timestamp value to filter by.
            identity: The identity value to filter by.

        Returns:
            The selected collection.

        Raises:
            ValueError: If timestamp/identity selection is attempted without the corresponding keys defined in the configuration.
        """
        if self._data is None:
            raise ValueError("not initialized")
        selection = np.ones(self.length, dtype=bool)
        if timestamp is not None:
            if self.cfg.key_timestamp is None:
                raise ValueError(
                    "timestamp select only possible for collections with defined timestamps"
                )
            timestamps = self[self.cfg.key_timestamp]
            selection = selection & np.isin(
                timestamps, np.asarray(timestamp, dtype=timestamps.dtype)
            )
        if identity is not None:
            if self.cfg.key_identity is None:
                raise ValueError(
                    "identity select only possible for collections with defined identities"
                )
            identities = self[self.cfg.key_identity]
            selection = selection & np.isin(
                identities, np.asarray(identity, dtype=identities.dtype)
            )
        if selection.all():
            return self.copy()
        selection = np.argwhere(selection).ravel()
        # advanced indexing with boolean array always triggers copy, not view
        return self.init_other(
            data={key: value[selection] for key, value in self.items(copy=False)}
        )

    @classmethod
    def concatenate(
        cls,
        *collections: Self,
        copy_config: bool = False,
        validate: bool = True,
    ) -> Self:
        """
        Concatenates multiple collections into a single one.

        Parameters:
            collections: The collections to concatenate.
            copy_config: Whether to copy the configuration of the first collection.
            validate: Whether to validate equality of configurations and concatenated data on the resulting collection.

        Returns:
            The concatenated collection.

        Raises:
            AssertionError: If no collections are provided, or if the configurations of the collections are not equal when validation is enabled.
            ValueError: If the identity types of the collections are not compatible.
        """
        if len(collections) == 0:
            raise AssertionError("need at least one collection to concatenate")
        cfg = collections[0].cfg
        if copy_config:
            cfg = cfg.copy()
        if validate and not all(
            [cfg == collection.cfg for collection in collections[1:]]
        ):
            raise AssertionError(
                "can only concatenate collections with equal configurations"
            )
        data = [collection.data for collection in collections]
        identity_dtype = None
        if cfg.key_identity is not None and cfg.key_identity in cfg.trajectory_keys:
            dtypes = [collection[cfg.key_identity].dtype for collection in collections]
            if all([np.issubdtype(dtype, int) for dtype in dtypes]):
                identity_dtype = int
            elif all(
                [
                    (isinstance(dtype, StringDType) or np.issubdtype(dtype, str))
                    for dtype in dtypes
                ]
            ):
                identity_dtype = StringDType()
            else:
                raise ValueError(
                    "can only concatenate collections with equal identity type."
                )
        data = {
            key: np.concatenate(
                [_data[key] for _data in data],
                axis=0,
                dtype=identity_dtype if key == cfg.key_identity else None,
            )
            for key in data[0].keys()
        }
        return cls(data=data, cfg=cfg, validate_on_init=validate)
