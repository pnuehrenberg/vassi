from collections.abc import Iterable
from typing import Optional

from numpy.typing import NDArray

from .. import config


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

    def _get_value(
        self,
        key: str,
        *,
        copy: bool = False,
    ) -> NDArray:
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
            self._get_value(key, copy=copy) for key in self.keys(exclude=exclude)
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
