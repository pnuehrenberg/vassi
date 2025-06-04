from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Mapping, Optional, Self, overload

import numpy as np
from numpy.dtypes import StringDType  # type: ignore

#
from .. import config
from . import _type_checking as type_checking
from . import instance, utils
from .base import ConfiguredData


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

    def __init__(
        self,
        *,
        data: Optional[Mapping[str, np.ndarray]] = None,
        cfg: Optional[config.Config] = None,
        validate_on_init: bool = True,
    ) -> None:
        self._validate = True
        self._view_of = []
        self._views = []
        if cfg is None:
            cfg = config.cfg.copy()
        self._cfg = cfg
        if data is not None:
            with self.validate(validate_on_init):
                self.data = data

    @contextmanager
    def validate(self, validate: bool) -> Generator:
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
        if self._data is None:
            return 0
        if not self._validate:
            return len(self[self.keys()[0]])
        length = utils.validated_length(*self.values(copy=False))
        assert length is not None
        return length

    def __len__(self) -> int:
        return self.length

    def validate_data(
        self,
        data: Mapping[str, utils.Value],
        *,
        allow_duplicated_timestamps: bool = True,
        allow_missing_keys: bool = False,
        try_broadcasting: bool = True,
        require_array_like=False,
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
        complete_keys = utils.validate_keys(
            data.keys(), self.keys(), allow_missing=allow_missing_keys
        )
        if require_array_like and any(
            [
                (not isinstance(value, Iterable) or isinstance(value, str))
                for value in data.values()
            ]
        ):
            raise ValueError("all values are required to be array-like.")
        length = utils.validated_length(*data.values())
        if (
            not allow_duplicated_timestamps
            and (key_timestamp := self.cfg.key_timestamp) is not None
            and key_timestamp in data
        ):
            utils.validate_timestamps(data[key_timestamp])
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
    def data(self) -> dict[str, np.ndarray]:
        """
        Property that returns the underlying data dictionary.
        The property can be set to update the data if the data passes validation.
        """
        return super().data

    @data.setter
    def data(self, data: Mapping[str, np.ndarray]) -> None:
        if self._validate:
            self.validate_data(data, require_array_like=True)
        self._data = {key: value for key, value in data.items()}

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
        value: utils.Value,
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
        with utils.writeable(
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
    def __getitem__(self, key: int | np.integer) -> instance.Instance:
        # trajectory index
        ...

    @overload
    def __getitem__(self, key: tuple[int | np.integer, str]) -> utils.Value:
        # trajectory index with key
        ...

    @overload
    def __getitem__(
        self, key: tuple[int | np.integer, tuple[str, ...] | list[str]]
    ) -> dict[str, utils.Value]:
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
        | instance.Instance
        | utils.Value
        | dict[str, utils.Value]
    ):
        if isinstance(key, str):
            # single key
            return self._get_value(key)
        valid, _key = type_checking.is_str_iterable(key)
        if valid:
            # multiple keys
            return tuple(self._get_value(_key) for _key in _key)
        if isinstance(key, slice):
            # slice
            view = self.init_other(
                data={_key: self._get_value(_key)[key] for _key in self.keys()}
            )
            self._views.append(view)
            view._view_of.append(self)
            return view
        valid, _key = type_checking.is_slice_str(key)
        if valid:
            # slice single key
            return self._get_value(_key[1])[_key[0]]
        valid, _key = type_checking.is_slice_str_iterable(key)
        if valid:
            # slice multiple keys
            return {__key: self._get_value(__key)[_key[0]] for __key in _key[1]}
        if isinstance(key, int | np.integer):
            # instance at index
            instance_data = {
                _key: (value[key] if value is not None else None)
                for _key, value in self.items(copy=False)
            }
            return instance.Instance(cfg=self.cfg, **instance_data, from_scalars=True)
        valid, _key = type_checking.is_int_str(key)
        if valid:
            # instance value at index
            return self._get_value(_key[1], copy=True)[_key[0]]
        valid, _key = type_checking.is_int_str_iterable(key)
        if valid:
            # instance values at index
            return {
                __key: self._get_value(__key, copy=True)[_key[0]] for __key in _key[1]
            }
        raise KeyError(f"unsupported key of type ({type(key)})")

    @overload
    def __setitem__(self, key: str, value: utils.Value) -> None:
        # single key, single value
        ...

    @overload
    def __setitem__(self, key: tuple[str, ...] | list[str], value: utils.Value) -> None:
        # multiple keys, single value
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[str, ...] | list[str],
        value: utils.MultipleValues,
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
        value: utils.Value,
    ) -> None:
        # single key with slice
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[slice, tuple[str, ...] | list[str]],
        value: utils.Value,
    ) -> None:
        # multiple keys with slice, single value
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[slice, tuple[str, ...] | list[str]],
        value: utils.MultipleValues,
    ) -> None:
        # multiple keys with slice, corresponding values
        ...

    @overload
    def __setitem__(
        self,
        key: int,
        value: instance.Instance,
    ) -> None:
        # instance at index
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[int, str],
        value: utils.Value,
    ) -> None:
        # instance value at index
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[int, tuple[str, ...] | list[str]],
        value: utils.Value,
    ) -> None:
        # instance values at index
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[int, tuple[str, ...] | list[str]],
        value: utils.MultipleValues,
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
        value: (utils.Value | utils.MultipleValues | Self | instance.Instance),
    ) -> None:
        # single value
        valid_value, _value = type_checking.is_value(value)
        if valid_value and isinstance(key, str):
            # single key, single value
            if self._validate:
                data = self.data
                data[key][:] = _value
                self.validate_data(data)
            self._set_value(key, _value, slice(None))
            return
        valid_key, _key = type_checking.is_str_iterable(key)
        if valid_value and valid_key:
            # multiple keys, single value
            if self._validate:
                data = self.data
                for __key in _key:
                    data[__key][:] = _value
                self.validate_data(data)
            for __key in _key:
                self._set_value(__key, _value, slice(None))
            return
        valid_key, _key = type_checking.is_slice_str(key)
        if valid_value and valid_key:
            # single key with slice
            if self._validate:
                data = self.data
                data[_key[1]][_key[0]] = _value
                self.validate_data(data)
            self._set_value(_key[1], _value, _key[0])
            return
        valid_key, _key = type_checking.is_slice_str_iterable(key)
        if valid_value and valid_key:
            # multiple keys with slice, single value
            if self._validate:
                data = self.data
                for __key in _key[1]:
                    data[__key][_key[0]] = _value
                self.validate_data(data)
            for __key in _key[1]:
                self._set_value(__key, _value, _key[0])
            return
        valid_key, _key = type_checking.is_int_str(key)
        if valid_value and valid_key:
            # instance value at index
            if self._validate:
                data = self.data
                data[_key[1]][_key[0]] = _value
                self.validate_data(data)
            self._set_value(_key[1], _value, _key[0])
            return
        valid_key, _key = type_checking.is_int_str_iterable(key)
        if valid_value and valid_key:
            # instance values at index
            if self._validate:
                data = self.data
                for __key in _key[1]:
                    data[__key][_key[0]] = _value
                self.validate_data(data)
            for __key in _key[1]:
                self._set_value(__key, _value, _key[0])
            return
        # corresponding values
        valid_value, _value = type_checking.is_value_iterable(value)
        valid_key, _key = type_checking.is_str_iterable(key)
        if valid_value and valid_key:
            # multiple keys and corresponding values
            if self._validate:
                data = self.data
                for __key, __value in zip(_key, _value):
                    data[__key][:] = __value
                self.validate_data(data)
            for __key, __value in zip(_key, _value):
                self._set_value(__key, __value, slice(None))
            return
        valid_key, _key = type_checking.is_slice_str_iterable(key)
        if not valid_key:
            valid_key, _key = type_checking.is_int_str_iterable(key)
        if valid_value and valid_key:
            # multiple keys with slice, corresponding values
            # instance values at index, corresponding values
            if self._validate:
                data = self.data
                for __key, __value in zip(_key[1], _value):
                    data[__key][_key[0]] = __value
                self.validate_data(data)
            for __key, __value in zip(_key[1], _value):
                self._set_value(__key, __value, _key[0])
            return
        if isinstance(key, slice) and isinstance(value, InstanceCollection):
            # slice
            _data = value.data
            if self._validate:
                self.validate_data(_data)
                data = self.data
                for _key, _value in _data.items():
                    data[_key][key] = _value
                self.validate_data(data)
            for _key, _value in _data.items():
                self._set_value(_key, _value, key)
            return
        if isinstance(key, int | np.integer) and isinstance(value, instance.Instance):
            # instance at index
            if self._validate:
                data = self.data
                for _key in value.keys():
                    data[_key][key] = value[_key]
                self.validate_data(data)
            for _key in value.keys():
                self._set_value(_key, value[_key], key)
            return
        raise ValueError(
            f"unsupported key value pair of types ({type(key)} {type(value)})"
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
        timestamp: Optional[utils.Value] = None,
        identity: Optional[utils.Value] = None,
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
