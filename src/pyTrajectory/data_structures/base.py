from typing import Optional
from collections.abc import Iterable
from numpy.typing import NDArray


from .. import config

from . import utils


class ConfiguredData:
    _data: Optional[dict[str, NDArray]] = None
    _cfg: Optional[config.Config] = None

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def cfg(
        self,
    ) -> config.Config:
        if self._cfg is None:
            return config.cfg
        return self._cfg

    def keys(
        self,
        *,
        exclude: Optional[Iterable[str]] = None,
    ) -> tuple[str, ...]:
        if exclude is None:
            exclude = []
        return tuple(set(self.cfg.trajectory_keys).difference(exclude))

    def get_value(
        self,
        key: str,
        *,
        copy: bool = False,
    ) -> NDArray:
        utils.validate_keys([key], self.keys(), allow_missing=True)
        if self._data is None:
            raise ValueError("not initialized")
        value = self._data[key]
        if copy:
            return value.copy()
        else:
            value = value.view()
            value.flags.writeable = False
        return value

    def values(
        self,
        *,
        exclude: Optional[Iterable[str]] = None,
        copy: bool = True,
    ) -> tuple[NDArray, ...]:
        return tuple(
            self.get_value(key, copy=copy) for key in self.keys(exclude=exclude)
        )

    def items(
        self,
        *,
        exclude: Optional[Iterable[str]] = None,
        copy: bool = True,
    ) -> tuple[tuple[str, NDArray], ...]:
        keys = self.keys(exclude=exclude)
        values = self.values(exclude=exclude, copy=copy)
        return tuple((key, value) for key, value in zip(keys, values))

    @property
    def data(self) -> dict[str, NDArray]:
        return {key: value for key, value in self.items()}
