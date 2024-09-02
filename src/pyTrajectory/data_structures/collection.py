from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Mapping, Optional, Self, overload

import numpy as np
from numpy.dtypes import StringDType  # type: ignore
from numpy.typing import NDArray

from .. import config
from . import instance, utils
from .base import ConfiguredData


class InstanceCollection(ConfiguredData):
    _view_of: list["InstanceCollection"]
    _views: list["InstanceCollection"]
    _validate: bool = True

    def __init__(
        self,
        *,
        data: Optional[Mapping[str, NDArray]] = None,
        cfg: Optional[config.Config] = None,
        validate_on_init: bool = True,
    ) -> None:
        self._view_of = []
        self._views = []
        if cfg is not None:
            self._cfg = cfg
        if data is not None:
            with self.validate(validate_on_init):
                self.data = data

    @contextmanager
    def validate(self, validate: bool) -> Generator:
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
    def data(self) -> dict[str, NDArray]:
        return super().data

    @data.setter
    def data(self, data: Mapping[str, NDArray]) -> None:
        if self._validate:
            self.validate_data(data, require_array_like=True)
        self._data = {key: value for key, value in data.items()}

    def _init_other(
        self,
        *,
        data: dict[str, NDArray],
        copy_config: bool = False,
        validate_on_init: bool = False,
    ) -> Self:
        cfg = self.cfg
        if copy_config:
            cfg = cfg.copy()
        return type(self)(
            data=data,
            cfg=cfg,
            validate_on_init=validate_on_init,
        )

    def copy(self, *, copy_config: bool = False) -> Self:
        return self._init_other(
            data=self.data, copy_config=copy_config, validate_on_init=False
        )

    def _set_value(
        self,
        key: str,
        value: utils.Value,
        at: slice | int | np.integer,
    ) -> None:
        if self._data is None:
            raise ValueError("not initialized")
        _value = self._get_value(key, copy=False)
        with utils.writeable(
            *[base._get_value(key, copy=False) for base in self._view_of],
            _value,
        ):
            _value[at] = value

    @overload
    def __getitem__(self, key: None) -> NDArray:
        # single key
        ...

    @overload
    def __getitem__(self, key: str) -> NDArray:
        # single key
        ...

    @overload
    def __getitem__(self, key: utils.MultipleStrings) -> tuple[NDArray, ...]:
        # multiple keys
        ...

    @overload
    def __getitem__(self, key: slice) -> Self:
        # slice
        ...

    @overload
    def __getitem__(self, key: tuple[slice, str]) -> NDArray:
        # slice single key
        ...

    @overload
    def __getitem__(
        self, key: tuple[slice, utils.MultipleStrings]
    ) -> dict[str, NDArray]:
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
        self, key: tuple[int | np.integer, utils.MultipleStrings]
    ) -> dict[str, utils.Value]:
        # trajectory index with multiple keys
        ...

    def __getitem__(
        self,
        key: (
            None
            | str
            | utils.MultipleStrings
            | slice
            | tuple[slice, str]
            | tuple[slice, utils.MultipleStrings]
            | int
            | np.integer
            | tuple[int | np.integer, str]
            | tuple[int | np.integer, utils.MultipleStrings]
        ),
    ) -> (
        NDArray
        | tuple[NDArray, ...]
        | Self
        | NDArray
        | dict[str, NDArray]
        | instance.Instance
        | utils.Value
        | dict[str, utils.Value]
    ):
        if key is None:
            raise KeyError
        if isinstance(key, str):
            # single key
            return self._get_value(key)
        valid, _key = utils.is_str_iterable(key)
        if valid:
            # multiple keys
            return tuple(self._get_value(_key) for _key in _key)
        if isinstance(key, slice):
            # slice
            view = self._init_other(
                data={_key: self._get_value(_key)[key] for _key in self.keys()}
            )
            self._views.append(view)
            view._view_of.append(self)
            return view
        valid, _key = utils.is_slice_str(key)
        if valid:
            # slice single key
            return self._get_value(_key[1])[_key[0]]
        valid, _key = utils.is_slice_str_iterable(key)
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
        valid, _key = utils.is_int_str(key)
        if valid:
            # instance value at index
            return self._get_value(_key[1], copy=True)[_key[0]]
        valid, _key = utils.is_int_str_iterable(key)
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
    def __setitem__(self, key: utils.MultipleStrings, value: utils.Value) -> None:
        # multiple keys, single value
        ...

    @overload
    def __setitem__(
        self,
        key: utils.MultipleStrings,
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
        key: tuple[slice, utils.MultipleStrings],
        value: utils.Value,
    ) -> None:
        # multiple keys with slice, single value
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[slice, utils.MultipleStrings],
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
        key: tuple[int, utils.MultipleStrings],
        value: utils.Value,
    ) -> None:
        # instance values at index
        ...

    @overload
    def __setitem__(
        self,
        key: tuple[int, utils.MultipleStrings],
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
            | utils.MultipleStrings
            | utils.MultipleStrings
            | slice
            | tuple[slice, str]
            | tuple[slice, utils.MultipleStrings]
            | tuple[slice, utils.MultipleStrings]
            | int
            | tuple[int, str]
            | tuple[int, utils.MultipleStrings]
            | tuple[int, utils.MultipleStrings]
        ),
        value: (
            utils.Value
            | utils.Value
            | utils.MultipleValues
            | Self
            | utils.Value
            | utils.Value
            | utils.MultipleValues
            | instance.Instance
            | utils.Value
            | utils.Value
            | utils.MultipleValues
        ),
    ) -> None:
        # single value
        valid_value, _value = utils.is_value(value)
        if valid_value and isinstance(key, str):
            # single key, single value
            if self._validate:
                data = self.data
                data[key][:] = _value
                self.validate_data(data)
            self._set_value(key, _value, slice(None))
            return
        valid_key, _key = utils.is_str_iterable(key)
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
        valid_key, _key = utils.is_slice_str(key)
        if valid_value and valid_key:
            # single key with slice
            if self._validate:
                data = self.data
                data[_key[1]][_key[0]] = _value
                self.validate_data(data)
            self._set_value(_key[1], _value, _key[0])
            return
        valid_key, _key = utils.is_slice_str_iterable(key)
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
        valid_key, _key = utils.is_int_str(key)
        if valid_value and valid_key:
            # instance value at index
            if self._validate:
                data = self.data
                data[_key[1]][_key[0]] = _value
                self.validate_data(data)
            self._set_value(_key[1], _value, _key[0])
            return
        valid_key, _key = utils.is_int_str_iterable(key)
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
        valid_value, _value = utils.is_value_iterable(value)
        valid_key, _key = utils.is_str_iterable(key)
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
        valid_key, _key = utils.is_slice_str_iterable(key)
        if not valid_key:
            valid_key, _key = utils.is_int_str_iterable(key)
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
        if isinstance(key, slice) and isinstance(value, "InstanceCollection"):
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

    def select_index(self, index: NDArray) -> Self:
        # advanced indexing with boolean array always triggers copy, not view
        # # TODO check index (at least 1d numpy array if int, else boolean with length of collection)
        # # otherwise coerce to array with appropriate type or raise
        return self._init_other(
            data={key: value[index] for key, value in self.items(copy=False)}
        )

    def select(
        self,
        *,
        timestamp: Optional[utils.Value] = None,
        identity: Optional[utils.Value] = None,
    ) -> Self:
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
        return self._init_other(
            data={key: value[selection] for key, value in self.items(copy=False)}
        )

    @classmethod
    def concatenate(
        cls,
        *collections: "InstanceCollection",
        copy_config: bool = False,
        validate: bool = True,
    ) -> Self:
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
